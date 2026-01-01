"""Background job management for training workflows."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Any, Callable, Dict, Optional


@dataclass
class JobEvent:
    event: str
    data: Dict[str, Any]


class EventQueue:
    """Thread-safe event queue for SSE streams."""

    def __init__(self) -> None:
        self._queue: Queue[JobEvent] = Queue()

    def put(self, event: str, data: Dict[str, Any]) -> None:
        self._queue.put(JobEvent(event=event, data=data))

    def get(self, timeout: float = 1.0) -> Optional[JobEvent]:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None


@dataclass
class TrainingJob:
    """State for a training job (pretrain or finetune)."""

    job_id: str
    kind: str
    trainer: Any
    eval_interval: int
    save_interval: int
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    accumulated_time: float = 0.0
    status: str = "paused"
    step: int = 0
    error: Optional[str] = None
    events: EventQueue = field(default_factory=EventQueue)
    lock: threading.Lock = field(default_factory=threading.Lock)
    _run_event: threading.Event = field(default_factory=threading.Event)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None
    last_metrics: Optional[Dict[str, Any]] = None

    def start(self, paused: bool = False) -> None:
        if self._thread and self._thread.is_alive():
            return
        if paused:
            self._run_event.clear()
            self.status = "paused"
        else:
            self._run_event.set()
            self.status = "running"
            if self.started_at is None:
                self.started_at = time.time()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.events.put("status", self._status_payload())

    def pause(self) -> None:
        self._run_event.clear()
        if self.status not in {"completed", "error", "canceled"}:
            self.status = "paused"
        self._freeze_time()
        self.events.put("status", self._status_payload())

    def resume(self) -> None:
        if self.status in {"completed", "error", "canceled"}:
            return
        self._run_event.set()
        self.status = "running"
        if self.started_at is None:
            self.started_at = time.time()
        self.events.put("status", self._status_payload())

    def cancel(self) -> None:
        self._stop_event.set()
        self._run_event.set()
        if self.status not in {"completed", "error"}:
            self.status = "canceled"
        self._freeze_time()
        self.events.put("status", self._status_payload())

    def step_once(self, include_batch: bool = False) -> Dict[str, Any]:
        with self.lock:
            step_start = time.time()
            metrics = self.trainer.train_single_step()
            self.step += 1
            self.last_metrics = metrics
            if self.started_at is None:
                self.accumulated_time += time.time() - step_start
            payload = _serialize_metrics(metrics, include_batch=include_batch)
            payload["iter"] = self.step
            payload["max_iters"] = self.trainer.max_iters
            payload.update(self._timing_payload())
            log_line = _format_log_line(self.kind, self.step, self.trainer.max_iters, metrics, payload)
            self.events.put("log", {"message": log_line, "iter": self.step})
            print(log_line, flush=True)
            self.events.put("metrics", payload)
            if self.step >= self.trainer.max_iters and self.status != "canceled":
                self.trainer.save_checkpoint(self.trainer.max_iters, is_final=True)
                if hasattr(self.trainer, "save_loss_graph"):
                    self.trainer.save_loss_graph()
                self.status = "completed"
                self._freeze_time()
                self.events.put("done", self._status_payload())
            return payload

    def _run_loop(self) -> None:
        try:
            while not self._stop_event.is_set() and self.step < self.trainer.max_iters:
                if not self._run_event.is_set():
                    time.sleep(0.1)
                    continue
                with self.lock:
                    metrics = self.trainer.train_single_step()
                    self.step += 1
                    self.last_metrics = metrics
                    payload = _serialize_metrics(metrics, include_batch=False)
                    payload["iter"] = self.step
                    payload["max_iters"] = self.trainer.max_iters
                    payload.update(self._timing_payload())
                log_line = _format_log_line(self.kind, self.step, self.trainer.max_iters, metrics, payload)
                self.events.put("log", {"message": log_line, "iter": self.step})
                print(log_line, flush=True)
                self.events.put("metrics", payload)

                if self.eval_interval > 0 and self.step % self.eval_interval == 0:
                    losses = self.trainer.estimate_loss()
                    self.events.put("eval", {
                        "iter": self.step,
                        "train_loss": losses.get("train"),
                        "val_loss": losses.get("val"),
                    })

                if self.save_interval > 0 and self.step % self.save_interval == 0:
                    self.trainer.save_checkpoint(self.step)
                    self.events.put("checkpoint", {"iter": self.step})

            if self.step >= self.trainer.max_iters and self.status != "canceled":
                self.trainer.save_checkpoint(self.trainer.max_iters, is_final=True)
                if hasattr(self.trainer, "save_loss_graph"):
                    self.trainer.save_loss_graph()
                self.status = "completed"
                self._freeze_time()
                self.events.put("done", self._status_payload())
        except Exception as exc:  # pragma: no cover - defensive logging
            self.error = str(exc)
            self.status = "error"
            self._freeze_time()
            self.events.put("error", {"message": self.error})

    def _status_payload(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "iter": self.step,
            "max_iters": self.trainer.max_iters,
            "created_at": self.created_at,
        }

    def _elapsed_time(self) -> float:
        if self.started_at is None:
            return self.accumulated_time
        return self.accumulated_time + (time.time() - self.started_at)

    def _timing_payload(self) -> Dict[str, Any]:
        elapsed = self._elapsed_time()
        iter_per_sec = self.step / elapsed if elapsed > 0 else None
        eta_seconds = None
        if iter_per_sec and iter_per_sec > 0:
            eta_seconds = max(0.0, (self.trainer.max_iters - self.step) / iter_per_sec)
        return {
            "elapsed_time": elapsed,
            "iter_per_sec": iter_per_sec,
            "eta_seconds": eta_seconds,
        }

    def _freeze_time(self) -> None:
        if self.started_at is not None:
            self.accumulated_time += time.time() - self.started_at
            self.started_at = None


class JobRegistry:
    """In-memory registry for active training jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def add(self, job: TrainingJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get(self, job_id: str) -> Optional[TrainingJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def list(self) -> Dict[str, TrainingJob]:
        with self._lock:
            return dict(self._jobs)


class InferenceSession:
    """In-memory inference session."""

    def __init__(self, session_id: str, model, tokenizer, cfg, checkpoint):
        self.session_id = session_id
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.checkpoint = checkpoint
        self.created_at = time.time()
        self.diagnostics: Dict[str, Dict[str, Any]] = {}


class InferenceRegistry:
    """In-memory registry for inference sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, InferenceSession] = {}
        self._lock = threading.Lock()

    def add(self, session: InferenceSession) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def get(self, session_id: str) -> Optional[InferenceSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def list(self) -> Dict[str, InferenceSession]:
        with self._lock:
            return dict(self._sessions)



def _serialize_metrics(metrics: Dict[str, Any], include_batch: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "loss": metrics.get("loss"),
        "running_loss": metrics.get("running_loss"),
        "grad_norm": metrics.get("grad_norm"),
        "aux_loss": metrics.get("aux_loss"),
    }

    if include_batch:
        payload.update(_serialize_batch(metrics))

    return payload


def _serialize_batch(metrics: Dict[str, Any]) -> Dict[str, Any]:
    def _tensor_to_list(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    batch = {}
    for key in ("inputs", "targets", "masks"):
        if key in metrics:
            batch[key] = _tensor_to_list(metrics[key])
    return batch


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "-"
    total = int(seconds)
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {sec}s"
    if minutes > 0:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _format_log_line(
    kind: str,
    step: int,
    max_iters: int,
    metrics: Dict[str, Any],
    payload: Dict[str, Any],
) -> str:
    loss = metrics.get("loss")
    running = metrics.get("running_loss")
    grad = metrics.get("grad_norm")
    elapsed = payload.get("elapsed_time")
    eta = payload.get("eta_seconds")
    parts = [f"[{kind}] Iter {step}/{max_iters}"]
    if loss is not None:
        parts.append(f"loss={loss:.4f}")
    if running is not None:
        parts.append(f"avg={running:.4f}")
    if grad is not None:
        parts.append(f"grad={grad:.4f}")
    parts.append(f"elapsed={_format_duration(elapsed)}")
    if eta is not None:
        parts.append(f"eta={_format_duration(eta)}")
    return " ".join(parts)
