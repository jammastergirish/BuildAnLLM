"""Unit tests for data source selection functionality."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest
from fastapi import UploadFile

from backend.app.schemas.training import PretrainJobPayload


class TestPretrainJobPayloadSchema:
    """Tests for the PretrainJobPayload Pydantic model."""

    def test_training_text_paths_accepts_list(self):
        """Verify training_text_paths accepts a list of strings."""
        payload = PretrainJobPayload(
            model_config={
                "d_model": 64,
                "n_heads": 2,
                "n_layers": 2,
                "n_ctx": 64,
                "d_head": 32,
                "d_mlp": 128,
                "vocab_size": 100,
            },
            training_text_paths=["path/to/file1.txt", "path/to/file2.txt"],
        )
        assert payload.training_text_paths == ["path/to/file1.txt", "path/to/file2.txt"]

    def test_training_text_paths_defaults_to_none(self):
        """Verify training_text_paths defaults to None when not provided."""
        payload = PretrainJobPayload(
            model_config={
                "d_model": 64,
                "n_heads": 2,
                "n_layers": 2,
                "n_ctx": 64,
                "d_head": 32,
                "d_mlp": 128,
                "vocab_size": 100,
            },
        )
        assert payload.training_text_paths is None

    def test_training_text_paths_accepts_empty_list(self):
        """Verify training_text_paths accepts an empty list."""
        payload = PretrainJobPayload(
            model_config={
                "d_model": 64,
                "n_heads": 2,
                "n_layers": 2,
                "n_ctx": 64,
                "d_head": 32,
                "d_mlp": 128,
                "vocab_size": 100,
            },
            training_text_paths=[],
        )
        assert payload.training_text_paths == []


class TestReadTrainingText:
    """Tests for the _read_training_text function."""

    @pytest.fixture
    def mock_upload_file(self) -> UploadFile:
        """Create a mock UploadFile with test content."""
        content = b"Uploaded file content."
        file_obj = BytesIO(content)
        upload = MagicMock(spec=UploadFile)
        upload.file = file_obj
        return upload

    def test_read_from_single_path(self, temp_data_files: dict[str, Path], monkeypatch: pytest.MonkeyPatch):
        """Verify reading from a single specified path."""
        from backend.app.routers import pretrain
        
        # Import after patching to get fresh module
        path = str(temp_data_files["source1"])
        result = pretrain._read_training_text(None, [path])
        
        assert result == "Content from source one."

    def test_read_from_multiple_paths(self, temp_data_files: dict[str, Path]):
        """Verify reading from multiple paths combines with newlines."""
        from backend.app.routers import pretrain
        
        paths = [str(temp_data_files["source1"]), str(temp_data_files["source2"])]
        result = pretrain._read_training_text(None, paths)
        
        assert "Content from source one." in result
        assert "Content from source two." in result
        assert "\n\n" in result  # Combined with double newline

    def test_read_with_upload_and_paths(self, temp_data_files: dict[str, Path], mock_upload_file: UploadFile):
        """Verify both upload and paths are combined."""
        from backend.app.routers import pretrain
        
        paths = [str(temp_data_files["source1"])]
        result = pretrain._read_training_text(mock_upload_file, paths)
        
        assert "Uploaded file content." in result
        assert "Content from source one." in result

    def test_ignores_nonexistent_paths(self, temp_data_files: dict[str, Path]):
        """Verify graceful handling of non-existent paths (skips them)."""
        from backend.app.routers import pretrain
        
        paths = [
            str(temp_data_files["source1"]),
            "/nonexistent/path/to/file.txt",
            str(temp_data_files["source2"]),
        ]
        result = pretrain._read_training_text(None, paths)
        
        # Should contain both existing files, skip the missing one
        assert "Content from source one." in result
        assert "Content from source two." in result

    def test_empty_paths_list_falls_back_to_default(self, temp_data_files: dict[str, Path], monkeypatch: pytest.MonkeyPatch):
        """Verify empty paths list falls back to Orwell."""
        from backend.app.routers import pretrain
        
        monkeypatch.chdir(temp_data_files["orwell"].parent.parent.parent)
        
        result = pretrain._read_training_text(None, [])
        
        assert result == "Default Orwell text."

    def test_upload_only_works(self, mock_upload_file: UploadFile):
        """Verify upload file alone works without paths."""
        from backend.app.routers import pretrain
        
        result = pretrain._read_training_text(mock_upload_file, None)
        
        assert result == "Uploaded file content."


class TestPretrainingDataSources:
    """Tests for PRETRAINING_DATA_SOURCES metadata."""

    def test_all_sources_have_required_fields(self):
        """Verify all data sources have filename, language, and script."""
        from backend.app.routers.pretrain import PRETRAINING_DATA_SOURCES
        
        required_fields = {"filename", "language", "script"}
        
        for name, info in PRETRAINING_DATA_SOURCES.items():
            missing = required_fields - set(info.keys())
            assert not missing, f"Source '{name}' missing fields: {missing}"

