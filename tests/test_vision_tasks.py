"""Tests for vision_tasks.py — all inference calls are mocked to avoid loading models."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from config import TASK_PROMPTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_image(width: int = 100, height: int = 100) -> Image.Image:
    """Return a minimal RGB test image."""
    return Image.new("RGB", (width, height), (128, 128, 128))


# ---------------------------------------------------------------------------
# Sample inference outputs that mimic Florence-2 results
# ---------------------------------------------------------------------------

_SAMPLE_OD_OUTPUT = {
    "<OD>": {
        "bboxes": [[10, 20, 50, 60], [70, 80, 120, 140]],
        "labels": ["cat", "dog"],
    }
}

_SAMPLE_CAPTION_OUTPUT = {
    "<CAPTION>": "A cat sitting on a windowsill looking outside.",
}

_SAMPLE_DETAILED_CAPTION_OUTPUT = {
    "<DETAILED_CAPTION>": "A fluffy orange cat sitting on a wooden windowsill, gazing outside at a garden.",
}

_SAMPLE_MORE_DETAILED_CAPTION_OUTPUT = {
    "<MORE_DETAILED_CAPTION>": (
        "A fluffy orange tabby cat is sitting comfortably on a weathered wooden windowsill. "
        "The cat gazes intently outside at a lush green garden with blooming flowers."
    ),
}

_SAMPLE_OCR_OUTPUT = {
    "<OCR>": "Hello World\nLine 2 of text",
}

_SAMPLE_OCR_EMPTY_OUTPUT = {
    "<OCR>": "",
}

_SAMPLE_GROUNDING_OUTPUT = {
    "<CAPTION_TO_PHRASE_GROUNDING>": {
        "bboxes": [[15, 25, 55, 65]],
        "labels": ["red ball"],
    }
}

_SAMPLE_GROUNDING_EMPTY_OUTPUT = {
    "<CAPTION_TO_PHRASE_GROUNDING>": {
        "bboxes": [],
        "labels": [],
    }
}


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------

class TestDetectObjects:
    """Tests for ``detect_objects``."""

    @patch("vision_tasks.model_manager")
    def test_detect_objects_returns_expected_structure(
        self, mock_mm: MagicMock
    ) -> None:
        """Result must contain 'objects', 'count', and 'inference_time' keys."""
        mock_mm.run_inference.return_value = _SAMPLE_OD_OUTPUT

        from vision_tasks import detect_objects

        result = detect_objects(_create_test_image())

        assert isinstance(result, dict)
        assert "objects" in result
        assert "count" in result
        assert "inference_time" in result
        assert isinstance(result["objects"], list)
        assert isinstance(result["count"], int)
        assert result["count"] >= 0

    @patch("vision_tasks.model_manager")
    def test_detect_objects_handles_error(
        self, mock_mm: MagicMock
    ) -> None:
        """When inference raises an exception the function should handle it gracefully."""
        mock_mm.run_inference.side_effect = RuntimeError("GPU out of memory")

        from vision_tasks import detect_objects

        result = detect_objects(_create_test_image())

        assert isinstance(result, dict)
        assert "error" in result


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

class TestGenerateCaption:
    """Tests for ``generate_caption``."""

    @patch("vision_tasks.model_manager")
    def test_generate_caption_returns_expected_structure(
        self, mock_mm: MagicMock
    ) -> None:
        """Result must contain 'caption' and 'inference_time'."""
        mock_mm.run_inference.return_value = _SAMPLE_CAPTION_OUTPUT

        from vision_tasks import generate_caption

        result = generate_caption(_create_test_image())

        assert isinstance(result, dict)
        assert "caption" in result
        assert "inference_time" in result
        assert isinstance(result["caption"], str)
        assert len(result["caption"]) > 0

    @patch("vision_tasks.model_manager")
    def test_generate_caption_detail_levels(
        self, mock_mm: MagicMock
    ) -> None:
        """Each detail level should invoke the corresponding Florence-2 task prompt."""
        from vision_tasks import generate_caption

        detail_map: dict[str, dict[str, str]] = {
            "normal": _SAMPLE_CAPTION_OUTPUT,
            "detailed": _SAMPLE_DETAILED_CAPTION_OUTPUT,
            "more_detailed": _SAMPLE_MORE_DETAILED_CAPTION_OUTPUT,
        }

        expected_prompts: dict[str, str] = {
            "normal": TASK_PROMPTS.get("caption", "<CAPTION>"),
            "detailed": TASK_PROMPTS.get("detailed_caption", "<DETAILED_CAPTION>"),
            "more_detailed": TASK_PROMPTS.get(
                "more_detailed_caption", "<MORE_DETAILED_CAPTION>"
            ),
        }

        for level, mock_output in detail_map.items():
            mock_mm.reset_mock()
            mock_mm.run_inference.return_value = mock_output

            result = generate_caption(_create_test_image(), detail=level)

            assert isinstance(result, dict)
            assert "caption" in result
            # Verify the correct task prompt was passed to run_inference
            call_args = mock_mm.run_inference.call_args
            assert call_args is not None, (
                f"run_inference was not called for detail level '{level}'"
            )
            # The task prompt is typically the first or second positional arg,
            # or a keyword arg named 'task' / 'task_prompt'.
            all_args_str = str(call_args)
            prompt = expected_prompts[level]
            assert prompt in all_args_str, (
                f"Expected task prompt '{prompt}' for level '{level}' "
                f"but call was: {all_args_str}"
            )


# ---------------------------------------------------------------------------
# OCR / text extraction
# ---------------------------------------------------------------------------

class TestExtractTextOCR:
    """Tests for ``extract_text_ocr``."""

    @patch("vision_tasks.model_manager")
    def test_extract_text_ocr_returns_expected_structure(
        self, mock_mm: MagicMock
    ) -> None:
        """Result must contain 'text', 'has_text', and 'inference_time'."""
        mock_mm.run_inference.return_value = _SAMPLE_OCR_OUTPUT

        from vision_tasks import extract_text_ocr

        result = extract_text_ocr(_create_test_image())

        assert isinstance(result, dict)
        assert "text" in result
        assert "has_text" in result
        assert "inference_time" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["has_text"], bool)


# ---------------------------------------------------------------------------
# Phrase grounding
# ---------------------------------------------------------------------------

class TestGroundObject:
    """Tests for ``ground_object``."""

    @patch("vision_tasks.model_manager")
    def test_ground_object_returns_expected_structure(
        self, mock_mm: MagicMock
    ) -> None:
        """Result must contain 'query', 'objects', 'found', and 'count'."""
        mock_mm.run_inference.return_value = _SAMPLE_GROUNDING_OUTPUT

        from vision_tasks import ground_object

        result = ground_object(_create_test_image(), query="red ball")

        assert isinstance(result, dict)
        assert "query" in result
        assert "objects" in result
        assert "found" in result
        assert "count" in result
        assert result["query"] == "red ball"
        assert isinstance(result["objects"], list)
        assert isinstance(result["found"], bool)
        assert isinstance(result["count"], int)
