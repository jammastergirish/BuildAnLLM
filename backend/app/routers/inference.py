"""Inference endpoints for loading models and generating text."""

from __future__ import annotations

import json
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.app.core.demo import block_if_demo
from backend.app.core.state import inference_registry
from backend.app.schemas.inference import DiagnosticsRequest, GenerateRequest, InferenceSessionRequest
from backend.app.services.checkpoints import resolve_checkpoint_path
from backend.app.services.inference import (
    build_diagnostics,
    generate_text,
    generate_text_stream,
    get_attention_map,
    get_layer_norms,
    get_logit_lens,
)
from backend.app.services.tokenizers import load_tokenizer_for_checkpoint
from pretraining.model.model_loader import load_model_from_checkpoint
from utils import get_device

router = APIRouter(prefix="/api/inference", dependencies=[Depends(block_if_demo)])


@router.post("/sessions")
async def create_session(request: InferenceSessionRequest) -> dict:
    checkpoint_path = resolve_checkpoint_path(request.checkpoint_id)
    device = get_device()
    model, cfg, checkpoint = load_model_from_checkpoint(str(checkpoint_path), device)

    tokenizer_type = checkpoint.get("tokenizer_type", "character")
    try:
        tokenizer = load_tokenizer_for_checkpoint(
            checkpoint_path, tokenizer_type, cfg.d_vocab
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_id = uuid4().hex
    session = _build_session(session_id, model, tokenizer, cfg, checkpoint)
    inference_registry.add(session)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6

    return {
        "session_id": session_id,
        "tokenizer_type": tokenizer_type,
        "param_count_m": round(param_count, 2),
        "cfg": cfg.to_dict(),
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    session = inference_registry.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    inference_registry.remove(session_id)
    return {"status": "deleted"}


@router.post("/sessions/{session_id}/generate")
async def generate(session_id: str, request: GenerateRequest) -> dict:
    session = inference_registry.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    device = next(session.model.parameters()).device
    text = generate_text(
        session.model,
        session.tokenizer,
        device,
        request.prompt,
        request.max_new_tokens,
        request.temperature,
        request.top_k,
        request.top_p,
    )
    return {
        "prompt": request.prompt,
        "generated_text": text,
        "new_characters": max(0, len(text) - len(request.prompt)),
    }


@router.post("/sessions/{session_id}/generate/stream")
async def generate_stream(session_id: str, request: GenerateRequest) -> StreamingResponse:
    session = inference_registry.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    device = next(session.model.parameters()).device
    token_stream = generate_text_stream(
        session.model,
        session.tokenizer,
        device,
        request.prompt,
        request.max_new_tokens,
        request.temperature,
        request.top_k,
        request.top_p,
    )

    def event_stream():
        yield _format_sse("start", {"prompt": request.prompt})
        try:
            for token in token_stream:
                yield _format_sse("token", {"token": token})
        except Exception as exc:
            yield _format_sse("error", {"message": str(exc)})
            return
        yield _format_sse("done", {"status": "ok"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/sessions/{session_id}/diagnostics")
async def diagnostics(session_id: str, request: DiagnosticsRequest) -> dict:
    session = inference_registry.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    device = next(session.model.parameters()).device
    diag_data = build_diagnostics(session.model, session.tokenizer, device, request.prompt)
    diag_id = uuid4().hex
    diag_data["session_id"] = session_id
    session.diagnostics[diag_id] = diag_data

    return {
        "diagnostic_id": diag_id,
        "token_ids": diag_data["token_ids"],
        "token_labels": diag_data["token_labels"],
    }


@router.get("/diagnostics/{diag_id}/attention")
async def attention_map(
    diag_id: str,
    layer: int = Query(..., ge=0),
    head: int = Query(..., ge=0),
) -> dict:
    session, diag_data = _find_diagnostics(diag_id)
    if diag_data is None:
        raise HTTPException(status_code=404, detail="Diagnostics not found")

    patterns = diag_data["attention_patterns"]
    if layer >= len(patterns):
        raise HTTPException(status_code=400, detail="Layer out of range")
    if head >= patterns[layer].shape[1]:
        raise HTTPException(status_code=400, detail="Head out of range")

    attn = get_attention_map(diag_data, layer, head)
    return {
        "attention": attn,
        "token_labels": diag_data["token_labels"],
        "layer": layer,
        "head": head,
    }


@router.get("/diagnostics/{diag_id}/logit-lens")
async def logit_lens(
    diag_id: str,
    position: int = Query(..., ge=0),
    top_k: int = Query(5, ge=1, le=10),
) -> dict:
    session, diag_data = _find_diagnostics(diag_id)
    if diag_data is None:
        raise HTTPException(status_code=404, detail="Diagnostics not found")

    if position >= len(diag_data["token_ids"]):
        raise HTTPException(status_code=400, detail="Position out of range")

    results = get_logit_lens(
        diag_data,
        session.model,
        session.tokenizer,
        position,
        top_k=top_k,
    )
    return {
        "position": position,
        "token_label": diag_data["token_labels"][position],
        "layers": results,
    }


@router.get("/diagnostics/{diag_id}/layer-norms")
async def layer_norms(diag_id: str) -> dict:
    _, diag_data = _find_diagnostics(diag_id)
    if diag_data is None:
        raise HTTPException(status_code=404, detail="Diagnostics not found")

    return {"layers": get_layer_norms(diag_data)}


def _build_session(session_id: str, model, tokenizer, cfg, checkpoint):
    from backend.app.core.jobs import InferenceSession

    return InferenceSession(session_id, model, tokenizer, cfg, checkpoint)


def _find_diagnostics(diag_id: str):
    for session in inference_registry.list().values():
        if diag_id in session.diagnostics:
            return session, session.diagnostics[diag_id]
    return None, None


def _format_sse(event: str, data: dict) -> str:
    payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"
