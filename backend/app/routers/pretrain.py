"""Pre-training job endpoints."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.app.core.demo import block_if_demo
from backend.app.core.jobs import TrainingJob
from backend.app.core.state import job_registry
from backend.app.schemas.training import AttentionRequest, InspectRequest, JobStatusResponse, JobStepRequest, PretrainJobPayload
from backend.app.services.model_config import build_model_config
from backend.app.services.training_inspect import build_attention_map, build_pretrain_inspect
from pretraining.data.dataset import TransformerDataset
from pretraining.model.model import TransformerModel
from pretraining.training.training_args import TransformerTrainingArgs
from pretraining.training.trainer import TransformerTrainer
from utils import get_device

router = APIRouter(prefix="/api/pretrain", dependencies=[Depends(block_if_demo)])


def _parse_payload(payload: str) -> PretrainJobPayload:
    try:
        return PretrainJobPayload.model_validate_json(payload)
    except AttributeError:
        return PretrainJobPayload.parse_raw(payload)


def _read_training_text(
    upload: Optional[UploadFile],
    training_text_paths: Optional[list[str]] = None,
) -> str:
    """Read training text from upload, specified paths, or default to Orwell."""
    texts: list[str] = []

    # Add uploaded file content if provided
    if upload is not None:
        texts.append(upload.file.read().decode("utf-8"))

    # Add content from selected data source paths
    if training_text_paths:
        for path_str in training_text_paths:
            path = Path(path_str)
            if path.exists():
                texts.append(path.read_text(encoding="utf-8"))

    # If we have any text, return it combined
    if texts:
        return "\n\n".join(texts)

    # Fall back to default Orwell text
    training_path = Path("input_data/pretraining/orwell.txt")
    if not training_path.exists():
        raise HTTPException(status_code=400, detail="Default pretraining text not found")
    return training_path.read_text(encoding="utf-8")


@router.post("/jobs", response_model=JobStatusResponse)
async def create_job(
    payload: str = Form(...),
    training_file: Optional[UploadFile] = File(default=None),
) -> JobStatusResponse:
    parsed = _parse_payload(payload)
    text = _read_training_text(training_file, parsed.training_text_paths)

    cfg = build_model_config(parsed.model_config_data)
    try:
        dataset = TransformerDataset(text, cfg, tokenizer_type=parsed.tokenizer_type)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    cfg = dataset.cfg

    device = get_device()
    model = TransformerModel(cfg, use_einops=parsed.use_einops).to(device)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = Path("checkpoints") / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    training_args = TransformerTrainingArgs(
        batch_size=parsed.training.batch_size,
        epochs=parsed.training.epochs,
        max_steps_per_epoch=parsed.training.max_steps_per_epoch,
        lr=parsed.training.learning_rate,
        weight_decay=parsed.training.weight_decay,
        eval_iters=parsed.training.eval_iters,
        save_dir=str(save_dir),
        save_interval=parsed.training.save_interval,
    )

    trainer = TransformerTrainer(
        model,
        training_args,
        *dataset.get_train_data(),
        *dataset.get_val_data(),
        device=device,
        eval_interval=parsed.training.eval_interval,
        tokenizer_type=parsed.tokenizer_type,
        tokenizer=dataset.tokenizer,
    )

    job_id = uuid4().hex
    job = TrainingJob(
        job_id=job_id,
        kind="pretrain",
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
        payload = build_pretrain_inspect(
            job,
            sample_index=request.sample_index,
            max_tokens=request.max_tokens,
            top_k=request.top_k,
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


# Define available pretraining data sources with metadata
PRETRAINING_DATA_SOURCES = {
    "Charles Dickens Collection": {
        "filename": "input_data/pretraining/dickens.txt",
        "language": "English",
        "script": "Latin",
    },
    "Homer's Iliad": {
        "filename": "input_data/pretraining/iliad.txt",
        "language": "Greek",
        "script": "Greek",
    },
    "Isaac Newton's Principia": {
        "filename": "input_data/pretraining/principia.txt",
        "language": "Latin",
        "script": "Latin",
    },
    "Vyasa's Bhagavad Gita": {
        "filename": "input_data/pretraining/gita.txt",
        "language": "Sanskrit",
        "script": "Devanagari",
    },
    "George Orwell Collection": {
        "filename": "input_data/pretraining/orwell.txt",
        "language": "English",
        "script": "Latin",
    },
    "William Shakespeare Collection": {
        "filename": "input_data/pretraining/shakespeare.txt",
        "language": "English",
        "script": "Latin",
    },
    "Oscar Wilde Collection": {
        "filename": "input_data/pretraining/wilde.txt",
        "language": "English",
        "script": "Latin",
    },
    "Muhammad al-Khwarizmi's Al-Jabr": {
        "filename": "input_data/pretraining/aljbr.txt",
        "language": "Arabic",
        "script": "Arabic",
    },
    "Marcel Proust's Swann's Way": {
        "filename": "input_data/pretraining/proust.txt",
        "language": "French",
        "script": "Latin",
    },
    "Miguel de Cervantes's Don Quixote": {
        "filename": "input_data/pretraining/donquixote.txt",
        "language": "Spanish",
        "script": "Latin",
    },
}


@router.get("/data-sources")
async def get_data_sources() -> dict:
    """Return available pretraining data sources with their stats."""
    sources = []
    for name, info in PRETRAINING_DATA_SOURCES.items():
        path = Path(info["filename"])
        if path.exists():
            content = path.read_text(encoding="utf-8")
            words = len(content.split())
            chars = len(content)
            sources.append({
                "name": name,
                "filename": info["filename"],
                "language": info["language"],
                "script": info["script"],
                "words": words,
                "chars": chars,
            })
    return {"sources": sources}
