"""Demo-mode helpers for disabling resource-intensive endpoints."""

from __future__ import annotations

import os

from fastapi import HTTPException, status


def is_demo_mode() -> bool:
    """Return True when demo mode is enabled via environment variable."""
    value = os.getenv("DEMO_MODE", "")
    return value.lower() in {"1", "true", "yes", "on"}


def block_if_demo() -> None:
    """Raise an HTTP error when demo mode is active."""
    if is_demo_mode():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is disabled in demo mode.",
        )
