"""
Configuration module for On-Device Vision AI application.

Centralizes all application settings including model parameters,
UI theming, supported formats, and logging configuration.
"""

import os

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_NAME: str = "On-Device Vision AI"
APP_VERSION: str = "1.0.0"
APP_DESCRIPTION: str = "Intelligent Computer Vision. Private. Local. Efficient."

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_ID: str = "florence-community/Florence-2-base"
CACHE_DIR: str = os.path.join(os.path.expanduser("~"), ".cache", "vision_ai_models")

# ---------------------------------------------------------------------------
# Image constraints
# ---------------------------------------------------------------------------
SUPPORTED_FORMATS: list[str] = [".jpg", ".jpeg", ".png", ".webp"]
MAX_FILE_SIZE_MB: int = 20
MAX_IMAGE_DIMENSION: int = 2048

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR: str = "outputs"

# ---------------------------------------------------------------------------
# Florence-2 task prompts
# ---------------------------------------------------------------------------
TASK_OD: str = "<OD>"
TASK_CAPTION: str = "<CAPTION>"
TASK_DETAILED_CAPTION: str = "<DETAILED_CAPTION>"
TASK_MORE_DETAILED_CAPTION: str = "<MORE_DETAILED_CAPTION>"
TASK_OCR: str = "<OCR>"
TASK_OCR_WITH_REGION: str = "<OCR_WITH_REGION>"
TASK_CAPTION_TO_PHRASE_GROUNDING: str = "<CAPTION_TO_PHRASE_GROUNDING>"
TASK_REGION_TO_DESCRIPTION: str = "<REGION_TO_DESCRIPTION>"
TASK_DENSE_REGION_CAPTION: str = "<DENSE_REGION_CAPTION>"

TASK_PROMPTS: dict[str, str] = {
    "od": TASK_OD,
    "caption": TASK_CAPTION,
    "detailed_caption": TASK_DETAILED_CAPTION,
    "more_detailed_caption": TASK_MORE_DETAILED_CAPTION,
    "ocr": TASK_OCR,
    "ocr_with_region": TASK_OCR_WITH_REGION,
    "phrase_grounding": TASK_CAPTION_TO_PHRASE_GROUNDING,
    "region_to_description": TASK_REGION_TO_DESCRIPTION,
    "dense_region_caption": TASK_DENSE_REGION_CAPTION,
}

# ---------------------------------------------------------------------------
# UI colour palette – dark theme
# ---------------------------------------------------------------------------
BG_PRIMARY: str = "#0a0a0a"
BG_SECONDARY: str = "#111111"
BG_CARD: str = "#1a1a1a"
ACCENT_PRIMARY: str = "#6366f1"       # indigo
ACCENT_SECONDARY: str = "#8b5cf6"    # violet
TEXT_PRIMARY: str = "#f5f5f5"
TEXT_SECONDARY: str = "#a1a1aa"
BORDER_COLOR: str = "#27272a"
SUCCESS_COLOR: str = "#22c55e"
WARNING_COLOR: str = "#f59e0b"
ERROR_COLOR: str = "#ef4444"

# ---------------------------------------------------------------------------
# Bounding-box colours (20 distinct colours)
# ---------------------------------------------------------------------------
BBOX_COLORS: list[str] = [
    "#FF0000",  # red
    "#00FF00",  # lime
    "#0000FF",  # blue
    "#FFFF00",  # yellow
    "#FF00FF",  # magenta
    "#00FFFF",  # cyan
    "#FF8000",  # orange
    "#8000FF",  # purple
    "#00FF80",  # spring green
    "#FF0080",  # rose
    "#80FF00",  # chartreuse
    "#0080FF",  # azure
    "#FF4040",  # coral
    "#40FF40",  # light green
    "#4040FF",  # medium blue
    "#FFD700",  # gold
    "#FF69B4",  # hot pink
    "#1E90FF",  # dodger blue
    "#32CD32",  # lime green
    "#FF6347",  # tomato
]

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
