"""Integration tests for FastAPI endpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from config import ModelConfig


class DummyTokenizer:
    def decode(self, token_ids):
        return f"T{token_ids[0]}"


class DummyUnembed:
    def __init__(self, d_model: int = 4, vocab_size: int = 10) -> None:
        self.W_U = torch.randn(d_model, vocab_size)


class DummyModel:
    def __init__(self, _cfg=None, use_einops: bool = True, d_model: int = 4, vocab_size: int = 10) -> None:
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.unembed = DummyUnembed(d_model, vocab_size)
        self.ln_f = torch.nn.Identity()
        self.pos_embed = None
        self.use_einops = use_einops

    def parameters(self):
        return iter([self._param])

    def to(self, device):
        return self

    def train(self):
        return self


class DummyDataset:
    def __init__(self, text: str, cfg: ModelConfig, tokenizer_type: str = "character") -> None:
        self.cfg = cfg
        self.tokenizer = DummyTokenizer()

    def get_train_data(self):
        X = torch.zeros((2, 2), dtype=torch.long)
        Y = torch.zeros((2, 2), dtype=torch.long)
        return X, Y

    def get_val_data(self):
        X = torch.zeros((2, 2), dtype=torch.long)
        Y = torch.zeros((2, 2), dtype=torch.long)
        return X, Y


class DummySFTDataset:
    def __init__(self, csv_path: str, tokenizer, max_length: int, mask_prompt: bool) -> None:
        self.tokenizer = tokenizer

    def get_train_data(self):
        X = torch.zeros((2, 2), dtype=torch.long)
        Y = torch.zeros((2, 2), dtype=torch.long)
        masks = torch.ones((2, 2), dtype=torch.long)
        return X, Y, masks

    def get_val_data(self):
        X = torch.zeros((2, 2), dtype=torch.long)
        Y = torch.zeros((2, 2), dtype=torch.long)
        masks = torch.ones((2, 2), dtype=torch.long)
        return X, Y, masks


class DummyTrainer:
    def __init__(self, model, args, *_, **__):
        self.model = model
        self.args = args
        self.max_iters = 2

    def train_single_step(self):
        return {
            "loss": 1.0,
            "running_loss": 1.0,
            "grad_norm": 0.1,
        }

    def estimate_loss(self):
        return {"train": 1.0, "val": 1.0}

    def save_checkpoint(self, step: int, is_final: bool = False):
        return None

    def save_loss_graph(self):
        return None


def _fake_build_diagnostics(model, tokenizer, device, prompt: str):
    seq_len = 3
    heads = 1
    d_model = model.unembed.W_U.shape[0]
    attention = torch.eye(seq_len).unsqueeze(0).unsqueeze(0).repeat(1, heads, 1, 1)
    layer_out = torch.randn(1, seq_len, d_model)
    return {
        "token_ids": [1, 2, 3],
        "token_labels": ["T1", "T2", "T3"],
        "attention_patterns": [attention],
        "layer_outputs": [layer_out],
    }


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    from backend.app.core.jobs import InferenceRegistry, JobRegistry, TrainingJob
    from backend.app.main import app
    import backend.app.routers.pretrain as pretrain
    import backend.app.routers.finetune as finetune
    import backend.app.routers.inference as inference

    monkeypatch.chdir(tmp_path)
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    dummy_checkpoint = checkpoints_dir / "dummy.pt"
    dummy_checkpoint.write_text("ckpt", encoding="utf-8")

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("prompt,response\nHi,Hello\n", encoding="utf-8")

    # Create sample pretraining data files for data sources endpoint
    pretraining_dir = tmp_path / "input_data" / "pretraining"
    pretraining_dir.mkdir(parents=True, exist_ok=True)
    (pretraining_dir / "orwell.txt").write_text("Sample George Orwell text for testing.", encoding="utf-8")
    (pretraining_dir / "dickens.txt").write_text("Sample Charles Dickens text for testing.", encoding="utf-8")
    (pretraining_dir / "shakespeare.txt").write_text("Sample William Shakespeare text for testing.", encoding="utf-8")
    (pretraining_dir / "wilde.txt").write_text("Sample Oscar Wilde text for testing.", encoding="utf-8")
    (pretraining_dir / "aljbr.txt").write_text("نموذج نص للاختبار", encoding="utf-8")
    (pretraining_dir / "proust.txt").write_text("Texte d'exemple pour les tests.", encoding="utf-8")
    (pretraining_dir / "donquixote.txt").write_text("Texto de ejemplo para pruebas.", encoding="utf-8")

    pretrain.job_registry = JobRegistry()
    finetune.job_registry = JobRegistry()
    inference.inference_registry = InferenceRegistry()

    def quiet_start(self, paused: bool = False) -> None:
        if paused:
            self.status = "paused"
        else:
            self.status = "running"
            if self.started_at is None:
                self.started_at = time.time()
        self.events.put("status", self._status_payload())

    monkeypatch.setattr(TrainingJob, "start", quiet_start)

    monkeypatch.setattr(pretrain, "_read_training_text", lambda upload, paths=None: "hello world")
    monkeypatch.setattr(pretrain, "TransformerDataset", DummyDataset)
    monkeypatch.setattr(pretrain, "TransformerModel", DummyModel)
    monkeypatch.setattr(pretrain, "TransformerTrainer", DummyTrainer)
    monkeypatch.setattr(pretrain, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pretrain, "build_pretrain_inspect", lambda *_, **__: {"ok": True})
    monkeypatch.setattr(pretrain, "build_attention_map", lambda *_, **__: {"attention": []})

    monkeypatch.setattr(finetune, "resolve_checkpoint_path", lambda _: dummy_checkpoint)
    monkeypatch.setattr(
        finetune,
        "load_model_from_checkpoint",
        lambda *_: (DummyModel(), ModelConfig.gpt_small(), {"tokenizer_type": "character"}),
    )
    monkeypatch.setattr(finetune, "load_tokenizer_for_checkpoint", lambda *_: DummyTokenizer())
    monkeypatch.setattr(finetune, "SFTDataset", DummySFTDataset)
    monkeypatch.setattr(finetune, "SFTTrainer", DummyTrainer)
    monkeypatch.setattr(finetune, "_prepare_csv", lambda _: csv_path)
    monkeypatch.setattr(finetune, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(finetune, "build_sft_inspect", lambda *_, **__: {"ok": True})
    monkeypatch.setattr(finetune, "build_attention_map", lambda *_, **__: {"attention": []})

    monkeypatch.setattr(inference, "resolve_checkpoint_path", lambda _: dummy_checkpoint)
    monkeypatch.setattr(
        inference,
        "load_model_from_checkpoint",
        lambda *_: (DummyModel(), ModelConfig.gpt_small(), {"tokenizer_type": "character"}),
    )
    monkeypatch.setattr(inference, "load_tokenizer_for_checkpoint", lambda *_: DummyTokenizer())
    monkeypatch.setattr(inference, "generate_text", lambda *_: "generated")
    monkeypatch.setattr(inference, "generate_text_stream", lambda *_: iter(["A", "B"]))
    monkeypatch.setattr(inference, "build_diagnostics", _fake_build_diagnostics)

    return TestClient(app)


@pytest.mark.integration
def test_root_and_health(api_client: TestClient):
    response = api_client.get("/")
    assert response.status_code == 200
    if "application/json" in response.headers.get("content-type", ""):
        assert response.json() == {"service": "transformer-backend"}
    else:
        assert response.text
    assert api_client.get("/api/health").json()["status"] == "ok"


@pytest.mark.integration
def test_system_info(api_client: TestClient):
    response = api_client.get("/api/system/info")
    assert response.status_code == 200
    assert "cpu" in response.json()


@pytest.mark.integration
def test_docs_endpoints(api_client: TestClient):
    payload = {"model_config": ModelConfig.gpt_small().to_dict()}
    response = api_client.post("/api/docs/model-code", json=payload)
    assert response.status_code == 200
    assert response.json()["snippets"]

    response = api_client.get("/api/docs/inference-code")
    assert response.status_code == 200
    assert response.json()["snippets"]

    response = api_client.post("/api/docs/finetuning-code", json={"use_lora": False})
    assert response.status_code == 200
    assert response.json()["snippets"]


@pytest.mark.integration
def test_tokenizer_endpoints(api_client: TestClient):
    response = api_client.get("/api/tokenizers/tiktoken/models")
    assert response.status_code == 200
    assert "models" in response.json()

    response = api_client.post(
        "/api/tokenizers/tiktoken/encode",
        json={"model": "gpt-4", "text": "Hello"},
    )
    assert response.status_code == 200
    assert response.json()["token_count"] >= 1


@pytest.mark.integration
def test_data_sources_endpoint(api_client: TestClient):
    """Test the GET /api/pretrain/data-sources endpoint."""
    response = api_client.get("/api/pretrain/data-sources")
    assert response.status_code == 200
    
    data = response.json()
    assert "sources" in data
    sources = data["sources"]
    
    # Should have multiple sources
    assert len(sources) >= 1
    
    # Each source should have required fields
    for source in sources:
        assert "name" in source
        assert "filename" in source
        assert "language" in source
        assert "script" in source
        assert "words" in source
        assert "chars" in source


@pytest.mark.integration
def test_data_sources_include_multilingual(api_client: TestClient):
    """Test that data sources include non-English sources."""
    response = api_client.get("/api/pretrain/data-sources")
    assert response.status_code == 200
    
    sources = response.json()["sources"]
    source_names = [s["name"] for s in sources]
    
    # Check for multilingual sources
    assert "Muhammad al-Khwarizmi" in source_names
    assert "Marcel Proust" in source_names
    assert "Miguel de Cervantes" in source_names


@pytest.mark.integration
def test_pretrain_job_with_training_text_paths(api_client: TestClient):
    """Test creating a pretrain job with specific training_text_paths."""
    payload = {
        "model_config": ModelConfig.gpt_small().to_dict(),
        "tokenizer_type": "character",
        "use_einops": True,
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "max_steps_per_epoch": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "eval_interval": 1,
            "eval_iters": 1,
            "save_interval": 10,
        },
        "training_text_paths": ["input_data/pretraining/orwell.txt"],
        "auto_start": False,
    }
    response = api_client.post("/api/pretrain/jobs", data={"payload": json.dumps(payload)})
    assert response.status_code == 200
    assert "job_id" in response.json()


@pytest.mark.integration
def test_pretrain_job_with_multiple_training_text_paths(api_client: TestClient):
    """Test creating a pretrain job with multiple training_text_paths for concatenation."""
    payload = {
        "model_config": ModelConfig.gpt_small().to_dict(),
        "tokenizer_type": "character",
        "use_einops": True,
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "max_steps_per_epoch": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "eval_interval": 1,
            "eval_iters": 1,
            "save_interval": 10,
        },
        "training_text_paths": [
            "input_data/pretraining/orwell.txt",
            "input_data/pretraining/dickens.txt",
            "input_data/pretraining/shakespeare.txt",
        ],
        "auto_start": False,
    }
    response = api_client.post("/api/pretrain/jobs", data={"payload": json.dumps(payload)})
    assert response.status_code == 200
    assert "job_id" in response.json()


@pytest.mark.integration
def test_pretrain_job_with_file_upload(api_client: TestClient):
    """Test creating a pretrain job with an uploaded training file."""
    payload = {
        "model_config": ModelConfig.gpt_small().to_dict(),
        "tokenizer_type": "character",
        "use_einops": True,
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "max_steps_per_epoch": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "eval_interval": 1,
            "eval_iters": 1,
            "save_interval": 10,
        },
        "auto_start": False,
    }
    # Create a simple text file for upload
    file_content = b"This is custom training text from an uploaded file."
    files = {"training_file": ("custom.txt", file_content, "text/plain")}
    response = api_client.post(
        "/api/pretrain/jobs",
        data={"payload": json.dumps(payload)},
        files=files,
    )
    assert response.status_code == 200
    assert "job_id" in response.json()


@pytest.mark.integration
def test_pretrain_job_with_upload_and_paths_combined(api_client: TestClient):
    """Test creating a pretrain job with both uploaded file and training_text_paths."""
    payload = {
        "model_config": ModelConfig.gpt_small().to_dict(),
        "tokenizer_type": "character",
        "use_einops": True,
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "max_steps_per_epoch": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "eval_interval": 1,
            "eval_iters": 1,
            "save_interval": 10,
        },
        "training_text_paths": ["input_data/pretraining/orwell.txt"],
        "auto_start": False,
    }
    # Create uploaded file
    file_content = b"Additional custom content from uploaded file."
    files = {"training_file": ("custom.txt", file_content, "text/plain")}
    response = api_client.post(
        "/api/pretrain/jobs",
        data={"payload": json.dumps(payload)},
        files=files,
    )
    assert response.status_code == 200
    assert "job_id" in response.json()


@pytest.mark.integration
def test_checkpoint_endpoints(api_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    import backend.app.routers.checkpoints as checkpoints

    monkeypatch.setattr(
        checkpoints,
        "list_checkpoints",
        lambda: [
            {
                "id": "checkpoints/dummy.pt",
                "path": "checkpoints/dummy.pt",
                "run_id": "run-1",
                "is_finetuned": False,
                "name": "dummy.pt",
                "iter": 1,
                "mtime": 0.0,
                "size_bytes": 10,
            }
        ],
    )

    response = api_client.get("/api/checkpoints")
    assert response.status_code == 200
    assert response.json()["checkpoints"]

    monkeypatch.setattr(checkpoints, "resolve_checkpoint_path", lambda _: Path("checkpoints/dummy.pt"))
    monkeypatch.setattr(
        checkpoints.torch,
        "load",
        lambda *_args, **_kwargs: {
            "cfg": ModelConfig.gpt_small().to_dict(),
            "tokenizer_type": "character",
            "iter_num": 1,
            "is_finetuned": False,
            "use_lora": False,
        },
    )
    response = api_client.get("/api/checkpoints/checkpoints/dummy.pt")
    assert response.status_code == 200
    assert response.json()["tokenizer_type"] == "character"


@pytest.mark.integration
def test_pretrain_job_flow(api_client: TestClient):
    payload = {
        "model_config": ModelConfig.gpt_small().to_dict(),
        "tokenizer_type": "character",
        "use_einops": True,
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "max_steps_per_epoch": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "eval_interval": 1,
            "eval_iters": 1,
            "save_interval": 10,
        },
        "auto_start": False,
    }
    response = api_client.post("/api/pretrain/jobs", data={"payload": json.dumps(payload)})
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    step = api_client.post(f"/api/pretrain/jobs/{job_id}/step", json={"include_batch": False})
    assert step.status_code == 200
    assert "metrics" in step.json()

    inspect_resp = api_client.post(
        f"/api/pretrain/jobs/{job_id}/inspect",
        json={"sample_index": 0, "max_tokens": 4, "top_k": 3},
    )
    assert inspect_resp.status_code == 200
    assert inspect_resp.json()["ok"] is True

    attention_resp = api_client.post(
        f"/api/pretrain/jobs/{job_id}/attention",
        json={"sample_index": 0, "layer": 0, "head": 0, "max_tokens": 4},
    )
    assert attention_resp.status_code == 200


@pytest.mark.integration
def test_finetune_job_flow(api_client: TestClient):
    payload = {
        "checkpoint_id": "checkpoints/dummy.pt",
        "max_length": 64,
        "use_lora": False,
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "max_steps_per_epoch": 2,
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "eval_interval": 1,
            "eval_iters": 1,
            "save_interval": 10,
        },
        "auto_start": False,
        "mask_prompt": True,
    }
    response = api_client.post("/api/finetune/jobs", data={"payload": json.dumps(payload)})
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    step = api_client.post(f"/api/finetune/jobs/{job_id}/step", json={"include_batch": False})
    assert step.status_code == 200
    assert "metrics" in step.json()

    inspect_resp = api_client.post(
        f"/api/finetune/jobs/{job_id}/inspect",
        json={"sample_index": 0, "max_tokens": 4},
    )
    assert inspect_resp.status_code == 200
    assert inspect_resp.json()["ok"] is True

    attention_resp = api_client.post(
        f"/api/finetune/jobs/{job_id}/attention",
        json={"sample_index": 0, "layer": 0, "head": 0, "max_tokens": 4},
    )
    assert attention_resp.status_code == 200


@pytest.mark.integration
def test_inference_session_flow(api_client: TestClient):
    response = api_client.post(
        "/api/inference/sessions",
        json={"checkpoint_id": "checkpoints/dummy.pt"},
    )
    assert response.status_code == 200
    session_id = response.json()["session_id"]

    response = api_client.post(
        f"/api/inference/sessions/{session_id}/generate",
        json={"prompt": "Hello", "max_new_tokens": 3},
    )
    assert response.status_code == 200
    assert response.json()["generated_text"] == "generated"

    stream_resp = api_client.post(
        f"/api/inference/sessions/{session_id}/generate/stream",
        json={"prompt": "Hello", "max_new_tokens": 3},
    )
    assert stream_resp.status_code == 200
    assert "event: token" in stream_resp.text

    diag_resp = api_client.post(
        f"/api/inference/sessions/{session_id}/diagnostics",
        json={"prompt": "Hello"},
    )
    assert diag_resp.status_code == 200
    diag_id = diag_resp.json()["diagnostic_id"]

    attention = api_client.get(
        f"/api/inference/diagnostics/{diag_id}/attention",
        params={"layer": 0, "head": 0},
    )
    assert attention.status_code == 200
    assert attention.json()["token_labels"]

    logit_lens = api_client.get(
        f"/api/inference/diagnostics/{diag_id}/logit-lens",
        params={"position": 0, "top_k": 3},
    )
    assert logit_lens.status_code == 200
    assert logit_lens.json()["layers"]

    norms = api_client.get(f"/api/inference/diagnostics/{diag_id}/layer-norms")
    assert norms.status_code == 200
    assert norms.json()["layers"]

    delete_resp = api_client.delete(f"/api/inference/sessions/{session_id}")
    assert delete_resp.status_code == 200


@pytest.mark.integration
def test_demo_mode_blocks_training_and_inference(api_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEMO_MODE", "true")

    payload = {
        "model_config": ModelConfig.gpt_small().to_dict(),
        "tokenizer_type": "character",
        "use_einops": True,
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "max_steps_per_epoch": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "eval_interval": 1,
            "eval_iters": 1,
            "save_interval": 10,
        },
        "auto_start": False,
    }
    response = api_client.post("/api/pretrain/jobs", data={"payload": json.dumps(payload)})
    assert response.status_code == 403
    assert response.json()["detail"] == "This endpoint is disabled in demo mode."

    finetune_payload = {
        "checkpoint_id": "checkpoints/dummy.pt",
        "max_length": 64,
        "use_lora": False,
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "max_steps_per_epoch": 2,
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "eval_interval": 1,
            "eval_iters": 1,
            "save_interval": 10,
        },
        "auto_start": False,
        "mask_prompt": True,
    }
    response = api_client.post("/api/finetune/jobs", data={"payload": json.dumps(finetune_payload)})
    assert response.status_code == 403

    response = api_client.post("/api/inference/sessions", json={"checkpoint_id": "checkpoints/dummy.pt"})
    assert response.status_code == 403

    assert api_client.get("/api/health").status_code == 200
