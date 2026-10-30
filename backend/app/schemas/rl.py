"""Pydantic schemas for reinforcement learning (GRPO) endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RLSessionRequest(BaseModel):
    checkpoint_id: str
    learning_rate: float = 3e-5
    beta: float = 0.02


class RLRoundRequest(BaseModel):
    train: bool = True
    digits: int = Field(default=3, ge=1, le=4)
    group: int = Field(default=8, ge=2, le=16)
    problems: int = Field(default=8, ge=1, le=32)
    max_new_tokens: int = Field(default=16, ge=4, le=64)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    beta: float | None = None
    # A fixed question for sampling without training. Ignored when training.
    a: int | None = None
    b: int | None = None


class RLEvalRequest(BaseModel):
    digits: int = Field(default=3, ge=1, le=4)
    samples: int = Field(default=4, ge=1, le=16)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=16, ge=4, le=64)
    which: Literal["policy", "ref"] = "policy"
    held_out: bool = True
