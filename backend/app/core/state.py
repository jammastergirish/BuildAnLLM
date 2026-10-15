"""Shared in-memory state for jobs and sessions."""

from backend.app.core.jobs import JobRegistry, InferenceRegistry

job_registry = JobRegistry()
inference_registry = InferenceRegistry()
# Same registry shape, keyed by session_id; holds RL (GRPO) sessions.
rl_registry = InferenceRegistry()
