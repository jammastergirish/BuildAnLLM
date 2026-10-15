"""Reinforcement learning (GRPO) endpoints.

A session holds one model being trained, a frozen copy of where it started, and
an optimiser. The page drives it one round at a time: each round samples a
group of answers per problem, scores them, and (when training) applies one
update. Nothing runs in the background, so the page can show every round.
"""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pretraining.model.model_loader import load_model_from_checkpoint
from rl.arithmetic import format_answer, format_prompt
from rl.grpo import GRPOTrainer
from utils import get_device

from backend.app.core.demo import block_if_demo
from backend.app.core.state import rl_registry
from backend.app.schemas.rl import RLEvalRequest, RLRoundRequest, RLSessionRequest
from backend.app.services.checkpoints import resolve_checkpoint_path
from backend.app.services.tokenizers import load_tokenizer_for_checkpoint

router = APIRouter(prefix="/api/rl", dependencies=[Depends(block_if_demo)])


class RLSession:
    def __init__(
        self,
        session_id: str,
        trainer: GRPOTrainer,
        checkpoint_path: Path,
        checkpoint: dict,
    ):
        self.session_id = session_id
        self.trainer = trainer
        self.checkpoint_path = checkpoint_path
        self.checkpoint = checkpoint
        self.created_at = time.time()


def tokenizer_problem(tokenizer) -> str | None:
    """Why this tokenizer cannot do the task, or None if it can.

    The task needs digits, '+', '?', a newline and the <answer> tags. A
    tokenizer trained only on a novel may have no tokens for some of these,
    and then the prompt and the tags come back garbled.
    """
    sample = format_prompt(42, 35) + format_answer(77)
    try:
        decoded = tokenizer.decode(tokenizer.encode(sample))
    except Exception as exc:  # e.g. a character tokenizer missing '<'
        return f"The tokenizer cannot encode the arithmetic prompt ({exc})."
    if decoded != sample:
        return (
            "This checkpoint's tokenizer cannot represent the arithmetic prompt: "
            f"{sample!r} comes back as {decoded!r}. Use a checkpoint trained with "
            "the GPT-2 (tiktoken) tokenizer."
        )
    return None


def _get(session_id: str) -> RLSession:
    session = rl_registry.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/sessions")
def create_session(request: RLSessionRequest) -> dict:
    try:
        checkpoint_path = resolve_checkpoint_path(request.checkpoint_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    device = get_device()
    model, cfg, checkpoint = load_model_from_checkpoint(str(checkpoint_path), device)

    tokenizer_type = checkpoint.get("tokenizer_type", "character")
    try:
        tokenizer = load_tokenizer_for_checkpoint(
            checkpoint_path, tokenizer_type, cfg.d_vocab
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    problem = tokenizer_problem(tokenizer)
    if problem:
        raise HTTPException(status_code=400, detail=problem)

    # Only one RL session at a time: each holds two copies of the model.
    for old in list(rl_registry.list()):
        rl_registry.remove(old)

    trainer = GRPOTrainer(
        model,
        tokenizer,
        device,
        tokenizer_type,
        lr=request.learning_rate,
        beta=request.beta,
    )
    session_id = uuid4().hex
    rl_registry.add(RLSession(session_id, trainer, checkpoint_path, checkpoint))
    return {
        "session_id": session_id,
        "tokenizer_type": tokenizer_type,
        "param_count_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "cfg": cfg.to_dict(),
        "is_finetuned": bool(checkpoint.get("is_finetuned", False)),
        "example_prompt": format_prompt(42, 35),
        "example_answer": format_answer(77),
    }


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    _get(session_id)
    rl_registry.remove(session_id)
    return {"status": "deleted"}


@router.post("/sessions/{session_id}/round")
def run_round(session_id: str, request: RLRoundRequest) -> dict:
    trainer = _get(session_id).trainer
    problem = None
    if request.a is not None and request.b is not None:
        problem = (request.a, request.b)
    try:
        return trainer.run_round(
            train=request.train,
            digits=request.digits,
            group=request.group,
            max_new=request.max_new_tokens,
            temperature=request.temperature,
            problems=request.problems,
            beta=request.beta,
            problem=problem,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sessions/{session_id}/evaluate")
def evaluate(session_id: str, request: RLEvalRequest) -> dict:
    trainer = _get(session_id).trainer
    return trainer.evaluate(
        digits=request.digits,
        samples=request.samples,
        temperature=request.temperature,
        max_new=request.max_new_tokens,
        which=request.which,
        held_out=request.held_out,
    )


@router.post("/sessions/{session_id}/reset")
def reset(session_id: str) -> dict:
    trainer = _get(session_id).trainer
    with trainer.lock:
        trainer.reset()
    return {"step": trainer.step}


@router.post("/sessions/{session_id}/save")
def save(session_id: str) -> dict:
    session = _get(session_id)
    run_dir = session.checkpoint_path.parent
    if run_dir.name in {"sft", "rl"}:
        run_dir = run_dir.parent
    with session.trainer.lock:
        path = session.trainer.save_checkpoint(str(run_dir / "rl"), session.checkpoint)
    return {"path": path, "step": session.trainer.step}
