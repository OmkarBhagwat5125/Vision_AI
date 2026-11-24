"""Tests for image_utils.py — image loading, validation, resizing, and drawing."""

import io
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from image_utils import (
    draw_bounding_boxes,
    get_image_metadata,
    image_to_bytes,
    load_image,
    resize_image,
    validate_image,
)
from config import MAX_FILE_SIZE_MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_image(
    width: int = 100,
    height: int = 100,
    mode: str = "RGB",
    colour: tuple[int, ...] = (255, 0, 0),
) -> Image.Image:
    """Return a solid-colour PIL Image for testing."""
    return Image.new(mode, (width, height), colour)


def _save_test_image(
    path: Path,
    width: int = 100,
    height: int = 100,
    mode: str = "RGB",
    fmt: str = "PNG",
) -> Path:
    """Save a small test image to *path* and return the path."""
    img = _create_test_image(width, height, mode)
    if mode == "RGBA" and fmt.upper() == "JPEG":
        img = img.convert("RGB")
    img.save(str(path), format=fmt)
    return path


def _make_uploaded_file(
    name: str = "photo.jpg",
    size: int = 1024,
    content: bytes | None = None,
) -> MagicMock:
    """Return a mock that behaves like a Streamlit ``UploadedFile``."""
    mock = MagicMock()
    mock.name = name
    mock.size = size

    if content is None:
        # Create a minimal valid JPEG if no content supplied
        buf = io.BytesIO()
        _create_test_image(10, 10).save(buf, format="JPEG")
        content = buf.getvalue()

    mock.read.return_value = content
    mock.seek = MagicMock()
    mock.getvalue.return_value = content
    # Support len() if checked
    mock.__len__ = MagicMock(return_value=size)
    return mock


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoadImage:
    """Tests for ``load_image``."""

    def test_load_image_from_path(self, tmp_path: Path) -> None:
        """Loading an image from a file path should return a PIL Image."""
        img_path = tmp_path / "test.png"
        _save_test_image(img_path)

        loaded = load_image(str(img_path))

        assert isinstance(loaded, Image.Image)
        assert loaded.size == (100, 100)

    def test_load_image_converts_to_rgb(self, tmp_path: Path) -> None:
        """An RGBA image should be converted to RGB upon loading."""
        img_path = tmp_path / "rgba_test.png"
        _save_test_image(img_path, mode="RGBA")

        loaded = load_image(str(img_path))

        assert loaded.mode == "RGB"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidateImage:
    """Tests for ``validate_image``."""

    def test_validate_image_valid_file(self) -> None:
        """A valid JPEG uploaded file should pass validation."""
        mock_file = _make_uploaded_file(name="photo.jpg", size=1024)

        is_valid, message = validate_image(mock_file)

        assert is_valid is True
        assert isinstance(message, str)

    def test_validate_image_invalid_extension(self) -> None:
        """A file with an unsupported extension should fail validation."""
        mock_file = _make_uploaded_file(name="document.txt", size=512)

        is_valid, message = validate_image(mock_file)

        assert is_valid is False
        assert isinstance(message, str)

    def test_validate_image_oversized(self) -> None:
        """A file exceeding MAX_FILE_SIZE_MB should fail validation."""
        oversized_bytes = int((MAX_FILE_SIZE_MB + 1) * 1024 * 1024)
        mock_file = _make_uploaded_file(name="huge.jpg", size=oversized_bytes)

        is_valid, message = validate_image(mock_file)

        assert is_valid is False
        assert isinstance(message, str)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestGetImageMetadata:
    """Tests for ``get_image_metadata``."""

    def test_get_image_metadata(self) -> None:
        """Returned dict must contain at least 'width', 'height', and 'mode'."""
        img = _create_test_image(320, 240)

        meta = get_image_metadata(img)

        assert isinstance(meta, dict)
        assert "width" in meta
        assert "height" in meta
        assert "mode" in meta
        assert meta["width"] == 320
        assert meta["height"] == 240
        assert meta["mode"] == "RGB"


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------

class TestResizeImage:
    """Tests for ``resize_image``."""

    def test_resize_image_larger(self) -> None:
        """An image larger than max_dim should be downscaled so both dims ≤ max_dim."""
        img = _create_test_image(4000, 3000)

        resized = resize_image(img, max_dim=2048)

        assert resized.width <= 2048
        assert resized.height <= 2048

    def test_resize_image_smaller_no_change(self) -> None:
        """An image already within limits should not be resized."""
        img = _create_test_image(100, 100)

        resized = resize_image(img, max_dim=2048)

        assert resized.size == (100, 100)

    def test_resize_preserves_aspect_ratio(self) -> None:
        """Resizing must preserve the original aspect ratio (within ±1 px rounding)."""
        original_w, original_h = 1000, 500
        img = _create_test_image(original_w, original_h)
        original_ratio = original_w / original_h

        resized = resize_image(img, max_dim=200)

        new_ratio = resized.width / resized.height
        assert math.isclose(original_ratio, new_ratio, rel_tol=0.02), (
            f"Aspect ratio changed: {original_ratio:.4f} → {new_ratio:.4f}"
        )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

class TestDrawBoundingBoxes:
    """Tests for ``draw_bounding_boxes``."""

    def test_draw_bounding_boxes(self) -> None:
        """Drawing bounding boxes on an image should return a valid PIL Image."""
        img = _create_test_image(200, 200)
        detections = [
            {"bbox": [10, 10, 50, 50], "label": "cat"},
            {"bbox": [60, 60, 120, 120], "label": "dog"},
        ]

        result = draw_bounding_boxes(img, detections)

        assert isinstance(result, Image.Image)
        assert result.size[0] > 0 and result.size[1] > 0

    def test_draw_bounding_boxes_empty_list(self) -> None:
        """Drawing with an empty detection list should still return a valid image."""
        img = _create_test_image(200, 200)

        result = draw_bounding_boxes(img, [])

        assert isinstance(result, Image.Image)
        assert result.size == (200, 200)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestImageToBytes:
    """Tests for ``image_to_bytes``."""

    def test_image_to_bytes(self) -> None:
        """Converting an image to bytes should return a non-empty bytes object."""
        img = _create_test_image(50, 50)

        raw = image_to_bytes(img)

        assert isinstance(raw, bytes)
        assert len(raw) > 0
