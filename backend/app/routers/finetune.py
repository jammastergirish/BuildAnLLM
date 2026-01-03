"""Fine-tuning job endpoints."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.app.core.demo import block_if_demo
from backend.app.core.jobs import TrainingJob
from backend.app.core.state import job_registry
from backend.app.schemas.training import AttentionRequest, FinetuneJobPayload, InspectRequest, JobStatusResponse, JobStepRequest
from backend.app.services.checkpoints import resolve_checkpoint_path
from backend.app.services.tokenizers import load_tokenizer_for_checkpoint
from backend.app.services.training_inspect import build_attention_map, build_sft_inspect
from config import PositionalEncoding
from finetuning.data.sft_dataset import SFTDataset
from finetuning.training.finetuning_args import FinetuningArgs
from finetuning.training.sft_trainer import SFTTrainer
from pretraining.model.model_loader import load_model_from_checkpoint
from pretraining.model.utils import extend_positional_embeddings
from utils import get_device

router = APIRouter(prefix="/api/finetune", dependencies=[Depends(block_if_demo)])


def _parse_payload(payload: str) -> FinetuneJobPayload:
    try:
        return FinetuneJobPayload.model_validate_json(payload)
    except AttributeError:
        return FinetuneJobPayload.parse_raw(payload)


def _prepare_csv(upload: Optional[UploadFile]) -> Path:
    if upload is not None:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp.write(upload.file.read())
        temp.close()
        return Path(temp.name)
    default_path = Path("finetuning.csv")
    if not default_path.exists():
        raise HTTPException(status_code=400, detail="finetuning.csv not found")
    return default_path


@router.post("/jobs", response_model=JobStatusResponse)
async def create_job(
    payload: str = Form(...),
    data_file: Optional[UploadFile] = File(default=None),
) -> JobStatusResponse:
    parsed = _parse_payload(payload)
    checkpoint_path = resolve_checkpoint_path(parsed.checkpoint_id)

    device = get_device()
    model, cfg, checkpoint = load_model_from_checkpoint(str(checkpoint_path), device)
    model.train()

    model_max_length = cfg.n_ctx if hasattr(cfg, "n_ctx") else 256
    if parsed.max_length > model_max_length:
        uses_learned_pos = (
            hasattr(cfg, "positional_encoding")
            and cfg.positional_encoding == PositionalEncoding.LEARNED
        )
        if uses_learned_pos:
            if hasattr(model, "pos_embed") and model.pos_embed is not None:
                extend_positional_embeddings(model.pos_embed, parsed.max_length)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot extend pos embeddings beyond {model_max_length}",
                )

    if parsed.use_lora:
        from finetuning.peft.lora_utils import convert_model_to_lora

        model = convert_model_to_lora(
            model,
            rank=parsed.lora_rank,
            alpha=parsed.lora_alpha,
            dropout=parsed.lora_dropout,
            target_modules=parsed.lora_target_modules,
        )
        model = model.to(device)

    tokenizer_type = checkpoint.get("tokenizer_type", "character")
    try:
        tokenizer = load_tokenizer_for_checkpoint(
            checkpoint_path, tokenizer_type, cfg.d_vocab
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    csv_path = _prepare_csv(data_file)
    try:
        dataset = SFTDataset(
            csv_path=str(csv_path),
            tokenizer=tokenizer,
            max_length=parsed.max_length,
            mask_prompt=parsed.mask_prompt,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    X_train, Y_train, masks_train = dataset.get_train_data()
    X_val, Y_val, masks_val = dataset.get_val_data()

    checkpoint_dir = checkpoint_path.parent
    if checkpoint_dir.name == "sft":
        checkpoint_dir = checkpoint_dir.parent
    sft_dir = checkpoint_dir / "sft"
    sft_dir.mkdir(exist_ok=True)

    training_args = FinetuningArgs(
        batch_size=parsed.training.batch_size,
        epochs=parsed.training.epochs,
        max_steps_per_epoch=parsed.training.max_steps_per_epoch,
        lr=parsed.training.learning_rate,
        weight_decay=parsed.training.weight_decay,
        save_dir=str(sft_dir),
        save_interval=parsed.training.save_interval,
        eval_iters=parsed.training.eval_iters,
        use_lora=parsed.use_lora,
        lora_rank=parsed.lora_rank,
        lora_alpha=parsed.lora_alpha,
        lora_dropout=parsed.lora_dropout,
        lora_target_modules=parsed.lora_target_modules,
    )

    trainer = SFTTrainer(
        model,
        training_args,
        X_train,
        Y_train,
        masks_train,
        X_val,
        Y_val,
        masks_val,
        device,
        eval_interval=parsed.training.eval_interval,
        tokenizer_type=tokenizer_type,
    )
    trainer.tokenizer = tokenizer

    job_id = uuid4().hex
    job = TrainingJob(
        job_id=job_id,
        kind="finetune",
        trainer=trainer,
        eval_interval=parsed.training.eval_interval,
        save_interval=parsed.training.save_interval,
    )
    job_registry.add(job)
    job.start(paused=not parsed.auto_start)

    return JobStatusResponse(**job._status_payload())


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str) -> JobStatusResponse:
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    payload = job._status_payload()
    payload["error"] = job.error
    return JobStatusResponse(**payload)


@router.post("/jobs/{job_id}/step")
async def step_job(job_id: str, request: JobStepRequest) -> dict:
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.step >= job.trainer.max_iters:
        raise HTTPException(status_code=400, detail="Job already completed")
    return {"metrics": job.step_once(include_batch=request.include_batch)}


@router.post("/jobs/{job_id}/inspect")
async def inspect_job(job_id: str, request: InspectRequest) -> dict:
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        payload = build_sft_inspect(
            job,
            sample_index=request.sample_index,
            max_tokens=request.max_tokens,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return payload


@router.post("/jobs/{job_id}/attention")
async def attention_job(job_id: str, request: AttentionRequest) -> dict:
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        payload = build_attention_map(
            job,
            sample_index=request.sample_index,
            layer=request.layer,
            head=request.head,
            max_tokens=request.max_tokens,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return payload


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str) -> dict:
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.pause()
    return job._status_payload()


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str) -> dict:
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.resume()
    return job._status_payload()


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.cancel()
    return job._status_payload()


@router.get("/jobs/{job_id}/events")
async def stream_events(job_id: str) -> StreamingResponse:
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    def event_stream():
        yield _format_sse("status", job._status_payload())
        while True:
            event = job.events.get(timeout=1.0)
            if event:
                yield _format_sse(event.event, event.data)
                if event.event in {"done", "error"}:
                    break
            else:
                if job.status in {"completed", "error", "canceled"}:
                    yield _format_sse("done", job._status_payload())
                    break
                yield ": ping\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _format_sse(event: str, data: dict) -> str:
    payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"
