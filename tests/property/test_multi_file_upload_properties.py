"""Property-based tests for multi-file upload functionality."""

from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest
from fastapi import UploadFile
from hypothesis import given, strategies as st, settings, assume


@pytest.mark.property
class TestMultiFileConcatenationProperties:
    """Property-based tests for file concatenation invariants."""

    @given(
        contents=st.lists(
            st.text(
                min_size=1,
                max_size=100,
                alphabet=st.characters(
                    blacklist_categories=("Cs", "Cc"),  # Exclude surrogates and control chars
                    blacklist_characters="\r",  # \r gets normalized to \n in text mode
                ),
            ),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=25)
    def test_concatenation_preserves_all_content(self, contents: List[str]):
        """Property: All file contents appear in the concatenated result."""
        from backend.app.routers import pretrain

        # Skip if any content is empty after stripping
        assume(all(c.strip() for c in contents))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            paths = []
            for i, content in enumerate(contents):
                file_path = tmp_path / f"file_{i}.txt"
                file_path.write_text(content, encoding="utf-8")
                paths.append(str(file_path))

            result = pretrain._read_training_text(None, paths)

            # Each content should appear in result
            for content in contents:
                assert content in result, f"Content '{content[:20]}...' not found in result"

    @given(
        contents=st.lists(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(
                    blacklist_categories=("Cs", "Cc"),
                    blacklist_characters="\r",
                ),
            ),
            min_size=2,
            max_size=4,
        )
    )
    @settings(max_examples=20)
    def test_concatenation_length_equals_sum_plus_separators(self, contents: List[str]):
        """Property: Result length equals sum of contents plus separators."""
        from backend.app.routers import pretrain

        assume(all(c for c in contents))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            paths = []
            for i, content in enumerate(contents):
                file_path = tmp_path / f"file_{i}.txt"
                file_path.write_text(content, encoding="utf-8")
                paths.append(str(file_path))

            result = pretrain._read_training_text(None, paths)

            # Expected length: sum of all content lengths + double newlines between
            expected_content_length = sum(len(c) for c in contents)
            num_separators = len(contents) - 1
            separator_length = 2  # "\n\n"
            expected_total = expected_content_length + (num_separators * separator_length)

            assert len(result) == expected_total

    @given(
        content=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(
                blacklist_categories=("Cs", "Cc"),
                blacklist_characters="\r",
            ),
        )
    )
    @settings(max_examples=30)
    def test_single_file_no_separator_overhead(self, content: str):
        """Property: Single file produces exact content without separators."""
        from backend.app.routers import pretrain

        assume(content)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / "single.txt"
            file_path.write_text(content, encoding="utf-8")

            result = pretrain._read_training_text(None, [str(file_path)])

            assert result == content

    @given(
        file_content=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                blacklist_categories=("Cs", "Cc"),
                blacklist_characters="\r",
            ),
        ),
        upload_content=st.binary(min_size=1, max_size=50),
    )
    @settings(max_examples=20)
    def test_upload_always_appears_before_paths(self, file_content: str, upload_content: bytes):
        """Property: Upload content always appears before path-specified content."""
        from backend.app.routers import pretrain

        assume(file_content)
        assume(upload_content)

        # Ensure unique markers
        try:
            upload_text = upload_content.decode("utf-8")
        except UnicodeDecodeError:
            upload_text = "UPLOAD_MARKER"
            upload_content = upload_text.encode("utf-8")

        assume(file_content not in upload_text)
        assume(upload_text not in file_content)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / "file.txt"
            file_path.write_text(file_content, encoding="utf-8")

            upload = MagicMock(spec=UploadFile)
            upload.file = BytesIO(upload_content)

            result = pretrain._read_training_text(upload, [str(file_path)])

            upload_pos = result.find(upload_text)
            file_pos = result.find(file_content)

            assert upload_pos >= 0, "Upload content not found"
            assert file_pos >= 0, "File content not found"
            assert upload_pos < file_pos, "Upload should appear before file content"


@pytest.mark.property
class TestFilePathProperties:
    """Property tests for file path handling."""

    @given(
        num_valid=st.integers(min_value=1, max_value=3),
        num_invalid=st.integers(min_value=0, max_value=3),
    )
    @settings(max_examples=15)
    def test_valid_files_always_included_regardless_of_invalid(
        self, num_valid: int, num_invalid: int
    ):
        """Property: Valid file content appears even when mixed with invalid paths."""
        from backend.app.routers import pretrain

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create valid files
            valid_paths = []
            valid_contents = []
            for i in range(num_valid):
                file_path = tmp_path / f"valid_{i}.txt"
                content = f"VALID_CONTENT_{i}"
                file_path.write_text(content, encoding="utf-8")
                valid_paths.append(str(file_path))
                valid_contents.append(content)

            # Create invalid paths (don't create the files)
            invalid_paths = [f"/nonexistent/path_{i}.txt" for i in range(num_invalid)]

            # Mix paths
            all_paths = valid_paths + invalid_paths

            result = pretrain._read_training_text(None, all_paths)

            # All valid content should be present
            for content in valid_contents:
                assert content in result


@pytest.mark.property
class TestCharacterEncodingProperties:
    """Property tests for character encoding handling."""

    @given(
        text=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "S", "Z"),
                blacklist_characters="\x00",
            ),
        )
    )
    @settings(max_examples=30)
    def test_unicode_text_survives_roundtrip(self, text: str):
        """Property: Any valid Unicode text survives file write/read cycle."""
        from backend.app.routers import pretrain

        assume(text.strip())

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / "unicode.txt"
            file_path.write_text(text, encoding="utf-8")

            result = pretrain._read_training_text(None, [str(file_path)])

            assert result == text

    @given(
        texts=st.lists(
            st.sampled_from([
                "English text",
                "中文文本",
                "日本語テキスト",
                "한국어 텍스트",
                "Текст на русском",
                "نص عربي",
                "טקסט בעברית",
                "ข้อความไทย",
                "🌍🎉🚀💡",
            ]),
            min_size=2,
            max_size=4,
        )
    )
    @settings(max_examples=15)
    def test_multilingual_concatenation_preserves_scripts(self, texts: List[str]):
        """Property: Multiple scripts in different files all preserved."""
        from backend.app.routers import pretrain

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            paths = []
            for i, text in enumerate(texts):
                file_path = tmp_path / f"lang_{i}.txt"
                file_path.write_text(text, encoding="utf-8")
                paths.append(str(file_path))

            result = pretrain._read_training_text(None, paths)

            for text in texts:
                assert text in result
