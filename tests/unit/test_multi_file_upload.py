"""Unit tests for multi-file upload functionality in pretraining."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import UploadFile


class TestMultiFileUploadConcatenation:
    """Tests for concatenating multiple uploaded files."""

    @pytest.fixture
    def temp_files(self, tmp_path: Path) -> dict[str, Path]:
        """Create temporary text files for testing."""
        files = {}

        file1 = tmp_path / "file1.txt"
        file1.write_text("Content from file one.", encoding="utf-8")
        files["file1"] = file1

        file2 = tmp_path / "file2.txt"
        file2.write_text("Content from file two.", encoding="utf-8")
        files["file2"] = file2

        file3 = tmp_path / "file3.txt"
        file3.write_text("Content from file three.", encoding="utf-8")
        files["file3"] = file3

        # Create default fallback
        pretraining_dir = tmp_path / "input_data" / "pretraining"
        pretraining_dir.mkdir(parents=True, exist_ok=True)
        orwell = pretraining_dir / "orwell.txt"
        orwell.write_text("Default fallback text.", encoding="utf-8")
        files["orwell"] = orwell

        return files

    def test_multiple_paths_concatenated_with_double_newline(self, temp_files: dict[str, Path]):
        """Verify multiple paths are concatenated with double newlines."""
        from backend.app.routers import pretrain

        paths = [str(temp_files["file1"]), str(temp_files["file2"]), str(temp_files["file3"])]
        result = pretrain._read_training_text(None, paths)

        # Check all content is present
        assert "Content from file one." in result
        assert "Content from file two." in result
        assert "Content from file three." in result

        # Check separator is double newline
        assert "\n\n" in result
        parts = result.split("\n\n")
        assert len(parts) == 3

    def test_concatenation_preserves_order(self, temp_files: dict[str, Path]):
        """Verify files are concatenated in the order provided."""
        from backend.app.routers import pretrain

        paths = [str(temp_files["file1"]), str(temp_files["file2"]), str(temp_files["file3"])]
        result = pretrain._read_training_text(None, paths)

        # Check order is preserved
        pos1 = result.find("file one")
        pos2 = result.find("file two")
        pos3 = result.find("file three")

        assert pos1 < pos2 < pos3

    def test_upload_combined_with_paths(self, temp_files: dict[str, Path]):
        """Verify uploaded file is combined with path-specified files."""
        from backend.app.routers import pretrain

        # Create mock upload
        upload_content = b"Uploaded content here."
        upload = MagicMock(spec=UploadFile)
        upload.file = BytesIO(upload_content)

        paths = [str(temp_files["file1"])]
        result = pretrain._read_training_text(upload, paths)

        # Both should be present
        assert "Uploaded content here." in result
        assert "Content from file one." in result

    def test_upload_appears_first_in_concatenation(self, temp_files: dict[str, Path]):
        """Verify uploaded file content appears before path-specified content."""
        from backend.app.routers import pretrain

        upload_content = b"UPLOAD_MARKER"
        upload = MagicMock(spec=UploadFile)
        upload.file = BytesIO(upload_content)

        paths = [str(temp_files["file1"])]
        result = pretrain._read_training_text(upload, paths)

        upload_pos = result.find("UPLOAD_MARKER")
        file_pos = result.find("Content from file one")

        assert upload_pos < file_pos


class TestMultiFileUploadEdgeCases:
    """Edge case tests for multi-file upload."""

    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        """Create temp directory with fallback."""
        pretraining_dir = tmp_path / "input_data" / "pretraining"
        pretraining_dir.mkdir(parents=True, exist_ok=True)
        orwell = pretraining_dir / "orwell.txt"
        orwell.write_text("Fallback text.", encoding="utf-8")
        return tmp_path

    def test_empty_file_included_in_concatenation(self, temp_dir: Path):
        """Verify empty files are handled gracefully."""
        from backend.app.routers import pretrain

        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        content_file = temp_dir / "content.txt"
        content_file.write_text("Has content.", encoding="utf-8")

        paths = [str(empty_file), str(content_file)]
        result = pretrain._read_training_text(None, paths)

        # Should contain the content from non-empty file
        assert "Has content." in result

    def test_unicode_content_preserved(self, temp_dir: Path):
        """Verify UTF-8 content is preserved through concatenation."""
        from backend.app.routers import pretrain

        # Create file with various Unicode characters
        unicode_file = temp_dir / "unicode.txt"
        unicode_content = "Hello 世界! Привет мир! مرحبا العالم! 🌍🚀"
        unicode_file.write_text(unicode_content, encoding="utf-8")

        result = pretrain._read_training_text(None, [str(unicode_file)])

        assert unicode_content in result

    def test_special_characters_preserved(self, temp_dir: Path):
        """Verify special characters like tabs and carriage returns are preserved."""
        from backend.app.routers import pretrain

        special_file = temp_dir / "special.txt"
        special_content = "Line1\tTabbed\nLine2\r\nWindows line\nLine3"
        special_file.write_text(special_content, encoding="utf-8")

        result = pretrain._read_training_text(None, [str(special_file)])

        assert "\t" in result
        assert "Tabbed" in result

    def test_single_file_no_separator(self, temp_dir: Path):
        """Verify single file doesn't have extra separators added."""
        from backend.app.routers import pretrain

        single_file = temp_dir / "single.txt"
        single_file.write_text("Just one file.", encoding="utf-8")

        result = pretrain._read_training_text(None, [str(single_file)])

        # Should not start or end with newlines
        assert result == "Just one file."


class TestMultiFileUploadValidation:
    """Tests for file upload validation behavior."""

    def test_nonexistent_paths_skipped(self, tmp_path: Path):
        """Verify non-existent paths are skipped without error."""
        from backend.app.routers import pretrain

        # Create one valid file
        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("Valid content.", encoding="utf-8")

        paths = [
            str(valid_file),
            "/nonexistent/path/to/file.txt",
            str(tmp_path / "also_missing.txt"),
        ]

        result = pretrain._read_training_text(None, paths)

        assert "Valid content." in result
        # Should not contain any indication of missing files

    def test_all_paths_nonexistent_returns_empty_or_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Verify behavior when all provided paths are non-existent."""
        from backend.app.routers import pretrain

        # Set up fallback
        pretraining_dir = tmp_path / "input_data" / "pretraining"
        pretraining_dir.mkdir(parents=True, exist_ok=True)
        (pretraining_dir / "orwell.txt").write_text("Fallback.", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        paths = ["/nonexistent1.txt", "/nonexistent2.txt"]
        result = pretrain._read_training_text(None, paths)

        # Should fall back to default
        assert result == "Fallback."

    def test_upload_only_no_paths(self):
        """Verify upload works when no paths are specified."""
        from backend.app.routers import pretrain

        upload_content = b"Upload only content."
        upload = MagicMock(spec=UploadFile)
        upload.file = BytesIO(upload_content)

        result = pretrain._read_training_text(upload, None)

        assert result == "Upload only content."

    def test_upload_with_empty_paths_list(self):
        """Verify upload works with empty paths list."""
        from backend.app.routers import pretrain

        upload_content = b"Upload with empty list."
        upload = MagicMock(spec=UploadFile)
        upload.file = BytesIO(upload_content)

        result = pretrain._read_training_text(upload, [])

        assert result == "Upload with empty list."
