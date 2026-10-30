"""Pydantic schemas for training endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PretrainTrainingParams(BaseModel):
    batch_size: int = 32
    epochs: int = 10
    max_steps_per_epoch: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    eval_interval: int = 500
    eval_iters: int = 10
    save_interval: int = 1000


class PretrainJobPayload(BaseModel):
    model_config_data: Dict[str, Any] = Field(alias="model_config")
    tokenizer_type: str = "bpe-tiktoken"
    use_einops: bool = True
    training: PretrainTrainingParams = Field(default_factory=PretrainTrainingParams)
    training_text_paths: Optional[List[str]] = None
    auto_start: bool = True

    model_config = ConfigDict(populate_by_name=True)


class FinetuneTrainingParams(BaseModel):
    batch_size: int = 4
    epochs: int = 3
    max_steps_per_epoch: int = 200
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    eval_interval: int = 50
    eval_iters: int = 50
    save_interval: int = 500


class FinetuneJobPayload(BaseModel):
    checkpoint_id: str
    max_length: int = 512
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    lora_target_modules: str = "all"
    training: FinetuneTrainingParams = Field(default_factory=FinetuneTrainingParams)
    auto_start: bool = True
    mask_prompt: bool = True
    # Bundled dataset to use when no CSV is uploaded: "default" or "arithmetic"
    dataset: str = "default"

    model_config = ConfigDict(populate_by_name=True)


class JobStepRequest(BaseModel):
    include_batch: bool = False

    model_config = ConfigDict(populate_by_name=True)


class InspectRequest(BaseModel):
    sample_index: int = 0
    max_tokens: int | None = None
    top_k: int = 10

    model_config = ConfigDict(populate_by_name=True)


class AttentionRequest(BaseModel):
    sample_index: int = 0
    layer: int
    head: int
    max_tokens: int | None = None

    model_config = ConfigDict(populate_by_name=True)


class JobStatusResponse(BaseModel):
    job_id: str
    kind: str
    status: str
    iter: int
    max_iters: int
    created_at: float
    error: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)
