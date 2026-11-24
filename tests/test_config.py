"""Tests for config.py — validates all configuration constants and settings."""

import re
import pytest

from config import (
    APP_NAME,
    MODEL_ID,
    SUPPORTED_FORMATS,
    MAX_FILE_SIZE_MB,
    MAX_IMAGE_DIMENSION,
    BBOX_COLORS,
    TASK_PROMPTS,
    OUTPUT_DIR,
)


# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------

def test_app_name_defined() -> None:
    """APP_NAME should be a non-empty string."""
    assert isinstance(APP_NAME, str)
    assert len(APP_NAME) > 0


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

def test_model_id_format() -> None:
    """MODEL_ID should follow the HuggingFace 'owner/model' format."""
    assert isinstance(MODEL_ID, str)
    assert "/" in MODEL_ID, "MODEL_ID must contain a '/' separating owner and model name"
    parts = MODEL_ID.split("/")
    assert len(parts) >= 2
    assert all(len(p) > 0 for p in parts), "Neither owner nor model name may be empty"


# ---------------------------------------------------------------------------
# Supported image formats
# ---------------------------------------------------------------------------

def test_supported_formats_not_empty() -> None:
    """The list of supported image formats must not be empty."""
    assert isinstance(SUPPORTED_FORMATS, (list, tuple, set, frozenset))
    assert len(SUPPORTED_FORMATS) > 0


def test_supported_formats_all_start_with_dot() -> None:
    """Every supported format string should start with a leading dot (e.g. '.png')."""
    for fmt in SUPPORTED_FORMATS:
        assert isinstance(fmt, str)
        assert fmt.startswith("."), f"Format '{fmt}' does not start with '.'"


# ---------------------------------------------------------------------------
# Size / dimension guards
# ---------------------------------------------------------------------------

def test_max_file_size_positive() -> None:
    """MAX_FILE_SIZE_MB must be a positive number."""
    assert isinstance(MAX_FILE_SIZE_MB, (int, float))
    assert MAX_FILE_SIZE_MB > 0


def test_max_image_dimension_positive() -> None:
    """MAX_IMAGE_DIMENSION must be a positive integer."""
    assert isinstance(MAX_IMAGE_DIMENSION, int)
    assert MAX_IMAGE_DIMENSION > 0


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_HEX_COLOUR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")


def test_color_palette_valid_hex() -> None:
    """All colour constants exposed by config must be valid hex colour codes.

    We inspect every module-level name that looks like a colour constant
    (uppercase, ends with _COLOR or _COLOUR, or is in BBOX_COLORS) and
    verify it matches #RGB or #RRGGBB.
    """
    import config

    # Gather explicit colour-like attributes
    colour_attrs: dict[str, str] = {}
    for name in dir(config):
        if name.startswith("_"):
            continue
        value = getattr(config, name)
        if not isinstance(value, str):
            continue
        # Heuristic: attribute name suggests it's a colour
        upper = name.upper()
        if upper.endswith("_COLOR") or upper.endswith("_COLOUR"):
            colour_attrs[name] = value

    # Also validate BBOX_COLORS entries
    for idx, colour in enumerate(BBOX_COLORS):
        colour_attrs[f"BBOX_COLORS[{idx}]"] = colour

    assert len(colour_attrs) > 0, "Expected at least one colour constant to validate"

    for label, value in colour_attrs.items():
        assert _HEX_COLOUR_RE.match(value), (
            f"{label} = '{value}' is not a valid hex colour code"
        )


def test_bbox_colors_sufficient() -> None:
    """BBOX_COLORS should contain at least 10 distinct colours."""
    assert isinstance(BBOX_COLORS, (list, tuple))
    assert len(BBOX_COLORS) >= 10, (
        f"Expected at least 10 bounding-box colours, got {len(BBOX_COLORS)}"
    )


# ---------------------------------------------------------------------------
# Florence-2 task prompts
# ---------------------------------------------------------------------------

def test_task_prompts_defined() -> None:
    """Every Florence-2 task prompt must start with '<' and end with '>'."""
    assert isinstance(TASK_PROMPTS, dict)
    assert len(TASK_PROMPTS) > 0, "TASK_PROMPTS must not be empty"

    for key, prompt in TASK_PROMPTS.items():
        assert isinstance(prompt, str), f"TASK_PROMPTS['{key}'] is not a string"
        assert prompt.startswith("<"), (
            f"TASK_PROMPTS['{key}'] = '{prompt}' does not start with '<'"
        )
        assert prompt.endswith(">"), (
            f"TASK_PROMPTS['{key}'] = '{prompt}' does not end with '>'"
        )


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

def test_output_dir_defined() -> None:
    """OUTPUT_DIR should be a non-empty string."""
    assert isinstance(OUTPUT_DIR, str)
    assert len(OUTPUT_DIR) > 0
