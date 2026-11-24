"""
Image utility functions for loading, validating, transforming, and
annotating images used by the On-Device Vision AI application.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

import config

logger = logging.getLogger(__name__)


# ======================================================================
# Loading
# ======================================================================


def load_image(source: str | bytes | Any) -> Image.Image:
    """Load an image from a file path, raw bytes, or a Streamlit
    ``UploadedFile``-like object and convert it to RGB.

    Args:
        source: A filesystem path (``str``), raw ``bytes``, or any
            file-like object with a ``.read()`` method (e.g. Streamlit's
            ``UploadedFile``).

    Returns:
        PIL.Image.Image: The loaded image in RGB mode.

    Raises:
        FileNotFoundError: If *source* is a path that does not exist.
        ValueError: If the source type is unsupported or the data cannot
            be decoded as an image.
    """
    try:
        if isinstance(source, str):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {source}")
            img = Image.open(path)

        elif isinstance(source, bytes):
            img = Image.open(io.BytesIO(source))

        elif hasattr(source, "read"):
            # Streamlit UploadedFile or generic file-like object
            data = source.read()
            if isinstance(data, str):
                raise ValueError("Expected binary file-like object, got text")
            img = Image.open(io.BytesIO(data))

        else:
            raise ValueError(
                f"Unsupported image source type: {type(source).__name__}"
            )

        if img.mode != "RGB":
            img = img.convert("RGB")

        logger.debug("Image loaded: %dx%d", img.width, img.height)
        return img

    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        raise ValueError(f"Failed to load image: {exc}") from exc


# ======================================================================
# Validation
# ======================================================================


def validate_image(uploaded_file: Any) -> tuple[bool, str]:
    """Validate an uploaded image file.

    Checks:
    1. File extension is in ``config.SUPPORTED_FORMATS``.
    2. File size does not exceed ``config.MAX_FILE_SIZE_MB``.
    3. The file can be opened as a valid image by PIL.

    Args:
        uploaded_file: A Streamlit ``UploadedFile`` (or similar) with
            ``.name`` and ``.size`` attributes and a ``.read()`` method.

    Returns:
        tuple[bool, str]: ``(True, "")`` on success, or
        ``(False, reason)`` on failure.
    """
    # --- Extension check ---
    name: str = getattr(uploaded_file, "name", "")
    ext = os.path.splitext(name)[1].lower()
    if ext not in config.SUPPORTED_FORMATS:
        return False, (
            f"Unsupported format '{ext}'. "
            f"Accepted: {', '.join(config.SUPPORTED_FORMATS)}"
        )

    # --- Size check ---
    file_size: int = getattr(uploaded_file, "size", 0)
    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_bytes:
        size_mb = file_size / (1024 * 1024)
        return False, (
            f"File too large ({size_mb:.1f} MB). "
            f"Maximum allowed: {config.MAX_FILE_SIZE_MB} MB."
        )

    # --- Decodability check ---
    try:
        data = uploaded_file.read()
        img = Image.open(io.BytesIO(data))
        img.verify()
        # Reset the file pointer so callers can re-read the file.
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
    except Exception as exc:
        return False, f"File cannot be opened as a valid image: {exc}"

    return True, ""


# ======================================================================
# Metadata
# ======================================================================


def get_image_metadata(
    image: Image.Image,
    file_obj: Any | None = None,
) -> dict[str, Any]:
    """Return basic metadata about *image*.

    Args:
        image: An opened PIL image.
        file_obj: Optional uploaded-file object; if provided its ``.size``
            attribute is included as ``file_size``.

    Returns:
        dict with keys ``width``, ``height``, ``format``, ``mode``, and
        optionally ``file_size``.
    """
    meta: dict[str, Any] = {
        "width": image.width,
        "height": image.height,
        "format": image.format or "unknown",
        "mode": image.mode,
    }

    if file_obj is not None:
        size = getattr(file_obj, "size", None)
        if size is not None:
            meta["file_size"] = size

    return meta


# ======================================================================
# Resizing
# ======================================================================


def resize_image(
    image: Image.Image,
    max_dim: int = config.MAX_IMAGE_DIMENSION,
) -> Image.Image:
    """Resize *image* so its longest side is at most *max_dim*.

    Aspect ratio is preserved.  If the image is already within bounds
    it is returned unchanged.

    Args:
        image: The source PIL image.
        max_dim: Maximum allowed width or height.

    Returns:
        PIL.Image.Image: The (possibly resized) image.
    """
    w, h = image.size
    if w <= max_dim and h <= max_dim:
        return image

    scale = max_dim / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = image.resize((new_w, new_h), Image.LANCZOS)
    logger.debug("Resized image from %dx%d → %dx%d", w, h, new_w, new_h)
    return resized


# ======================================================================
# Annotation – bounding boxes
# ======================================================================


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a TrueType font, falling back to the PIL default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def draw_bounding_boxes(
    image: Image.Image,
    detections: list[dict[str, Any]],
    color_map: list[str] | None = None,
) -> Image.Image:
    """Draw labelled bounding boxes onto a **copy** of *image*.

    Args:
        image: Source PIL image.
        detections: List of dicts, each with ``"label"`` (str) and
            ``"bbox"`` ([x1, y1, x2, y2] in pixel coordinates).
        color_map: Optional list of hex colour strings. Defaults to
            ``config.BBOX_COLORS``.

    Returns:
        PIL.Image.Image: Annotated copy.
    """
    colors = color_map or config.BBOX_COLORS
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _get_font(size=max(12, min(image.width // 50, 24)))

    for idx, det in enumerate(detections):
        bbox = det.get("bbox", [])
        label: str = det.get("label", "")
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        colour = colors[idx % len(colors)]

        # Draw rectangle
        line_width = max(2, min(image.width // 300, 4))
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=line_width)

        # Draw label background + text
        if label:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            padding = 4

            label_y = max(0, y1 - text_h - 2 * padding)
            draw.rectangle(
                [x1, label_y, x1 + text_w + 2 * padding, label_y + text_h + 2 * padding],
                fill=colour,
            )
            draw.text(
                (x1 + padding, label_y + padding),
                label,
                fill="#FFFFFF",
                font=font,
            )

    return annotated


def draw_grounding_results(
    image: Image.Image,
    grounding_results: dict[str, Any] | list[dict[str, Any]],
) -> Image.Image:
    """Draw bounding boxes for phrase-grounding results.

    Accepts either the dictionary returned by ``vision_tasks.ground_object``
    (containing an ``"objects"`` key) or a list of detection dictionaries directly.

    Args:
        image: Source PIL image.
        grounding_results: Dict with an ``"objects"`` key or a list of detection dicts.

    Returns:
        PIL.Image.Image: Annotated copy.
    """
    if isinstance(grounding_results, dict):
        objects = grounding_results.get("objects", [])
    elif isinstance(grounding_results, list):
        objects = grounding_results
    else:
        objects = []
    return draw_bounding_boxes(image, objects)


# ======================================================================
# Serialisation & persistence
# ======================================================================


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL image to raw bytes in the given format.

    Args:
        image: The source PIL image.
        fmt: Image format string (e.g. ``"PNG"``, ``"JPEG"``).

    Returns:
        bytes: The encoded image data.
    """
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def save_output_image(
    image: Image.Image,
    filename: str,
    output_dir: str = config.OUTPUT_DIR,
) -> str:
    """Save *image* to *output_dir*/*filename*, creating the directory
    if necessary.

    Args:
        image: The PIL image to save.
        filename: Destination file name (e.g. ``"annotated.png"``).
        output_dir: Directory path. Defaults to ``config.OUTPUT_DIR``.

    Returns:
        str: Absolute path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    image.save(path)
    logger.info("Saved output image to %s", path)
    return os.path.abspath(path)
