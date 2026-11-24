"""
High-level vision task functions powered by Florence-2.

Each public function accepts a PIL image and an optional ``VisionModelManager``,
runs the appropriate Florence-2 task prompt, and returns a consistently
structured result dictionary.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from PIL import Image

import config
from model import VisionModelManager

logger = logging.getLogger(__name__)

# Module-level model manager instance for convenience and test mocking
model_manager: VisionModelManager | None = None


def _get_manager(mm: VisionModelManager | None = None) -> VisionModelManager:
    """Return the supplied manager or fall back to the module/singleton manager."""
    global model_manager
    if mm is not None:
        return mm
    if model_manager is not None:
        return model_manager
    model_manager = VisionModelManager.get_instance()
    return model_manager


# ======================================================================
# Helper – safely extract a list of bounding-box dicts
# ======================================================================

def _parse_od_results(
    raw: dict[str, Any],
    task_key: str,
) -> list[dict[str, Any]]:
    """Parse Florence-2 object-detection / dense-caption output.

    Florence-2 returns results keyed by the task prompt, with ``bboxes``
    and ``labels`` lists. This helper normalises them into a list of
    ``{"label": str, "bbox": [x1, y1, x2, y2]}`` dicts.
    """
    task_data = raw.get(task_key, {})
    if not isinstance(task_data, dict):
        return []

    bboxes: list[list[float]] = task_data.get("bboxes", [])
    labels: list[str] = task_data.get("labels", [])

    objects: list[dict[str, Any]] = []
    for idx, bbox in enumerate(bboxes):
        label = labels[idx] if idx < len(labels) else f"object_{idx}"
        objects.append({
            "label": label,
            "bbox": [round(c, 2) for c in bbox],
        })
    return objects


# ======================================================================
# Public task functions
# ======================================================================


def detect_objects(
    image: Image.Image,
    model_manager: VisionModelManager | None = None,
) -> dict[str, Any]:
    """Run object detection on *image*.

    Returns:
        dict with keys ``objects``, ``count``, ``inference_time``.
    """
    logger.info("Running object detection …")
    mgr = _get_manager(model_manager)
    try:
        raw = mgr.run_inference(image, config.TASK_OD)
        inference_time: float = raw.pop("inference_time", 0.0)

        objects = _parse_od_results(raw, config.TASK_OD)

        return {
            "objects": objects,
            "count": len(objects),
            "inference_time": round(inference_time, 4),
        }

    except Exception as exc:
        logger.error("Object detection failed: %s", exc)
        return {
            "objects": [],
            "count": 0,
            "inference_time": 0.0,
            "error": str(exc),
        }


def generate_caption(
    image: Image.Image,
    model_manager: VisionModelManager | None = None,
    detail_level: str = "normal",
    detail: str | None = None,
) -> dict[str, Any]:
    """Generate a natural-language caption for *image*.

    Args:
        image: PIL RGB image.
        model_manager: Optional VisionModelManager instance.
        detail_level: One of ``"normal"``, ``"detailed"``, or
            ``"more_detailed"``.
        detail: Alias for ``detail_level``.

    Returns:
        dict with keys ``caption``, ``detail_level``, ``inference_time``.
    """
    if detail is not None:
        detail_level = detail

    task_map: dict[str, str] = {
        "normal": config.TASK_CAPTION,
        "detailed": config.TASK_DETAILED_CAPTION,
        "more_detailed": config.TASK_MORE_DETAILED_CAPTION,
    }
    task_prompt = task_map.get(detail_level, config.TASK_CAPTION)
    effective_level = detail_level if detail_level in task_map else "normal"

    logger.info("Generating %s caption …", effective_level)
    mgr = _get_manager(model_manager)
    try:
        raw = mgr.run_inference(image, task_prompt)
        inference_time = raw.pop("inference_time", 0.0)

        # Florence-2 returns the caption under the task key.
        caption: str = raw.get(task_prompt, "")
        if isinstance(caption, list):
            caption = caption[0] if caption else ""

        return {
            "caption": caption.strip(),
            "detail_level": effective_level,
            "inference_time": round(inference_time, 4),
        }

    except Exception as exc:
        logger.error("Caption generation failed: %s", exc)
        return {
            "caption": "",
            "detail_level": effective_level,
            "inference_time": 0.0,
            "error": str(exc),
        }


def extract_text_ocr(
    image: Image.Image,
    model_manager: VisionModelManager | None = None,
) -> dict[str, Any]:
    """Extract text from *image* using OCR.

    Returns:
        dict with keys ``text``, ``has_text``, ``inference_time``.
    """
    logger.info("Running OCR text extraction …")
    mgr = _get_manager(model_manager)
    try:
        raw = mgr.run_inference(image, config.TASK_OCR)
        inference_time = raw.pop("inference_time", 0.0)

        text: str = raw.get(config.TASK_OCR, "")
        if isinstance(text, list):
            text = " ".join(text)

        text = text.strip()

        return {
            "text": text,
            "has_text": bool(text),
            "inference_time": round(inference_time, 4),
        }

    except Exception as exc:
        logger.error("OCR extraction failed: %s", exc)
        return {
            "text": "",
            "has_text": False,
            "inference_time": 0.0,
            "error": str(exc),
        }


def ground_object(
    image: Image.Image,
    model_manager: VisionModelManager | None = None,
    object_query: str = "",
    query: str = "",
) -> dict[str, Any]:
    """Find and localise target object in *image* (phrase grounding).

    Args:
        image: PIL RGB image.
        model_manager: Optional VisionModelManager instance.
        object_query: A short natural-language description of target object.
        query: Alias for ``object_query``.

    Returns:
        dict with keys ``query``, ``objects``, ``found``, ``count``,
        ``inference_time``.
    """
    effective_query = query if query else object_query
    logger.info("Grounding query '%s' …", effective_query)
    mgr = _get_manager(model_manager)
    try:
        raw = mgr.run_inference(
            image,
            config.TASK_CAPTION_TO_PHRASE_GROUNDING,
            text_input=effective_query,
        )
        inference_time = raw.pop("inference_time", 0.0)

        objects = _parse_od_results(raw, config.TASK_CAPTION_TO_PHRASE_GROUNDING)

        return {
            "query": effective_query,
            "objects": objects,
            "found": len(objects) > 0,
            "count": len(objects),
            "inference_time": round(inference_time, 4),
        }

    except Exception as exc:
        logger.error("Phrase grounding failed for '%s': %s", effective_query, exc)
        return {
            "query": effective_query,
            "objects": [],
            "found": False,
            "count": 0,
            "inference_time": 0.0,
            "error": str(exc),
        }


def get_dense_captions(
    image: Image.Image,
    model_manager: VisionModelManager | None = None,
) -> dict[str, Any]:
    """Generate dense region captions for *image*.

    Returns:
        dict with keys ``regions`` (list of ``{"label", "bbox"}`` dicts),
        ``count``, ``inference_time``.
    """
    logger.info("Generating dense region captions …")
    mgr = _get_manager(model_manager)
    try:
        raw = mgr.run_inference(image, config.TASK_DENSE_REGION_CAPTION)
        inference_time = raw.pop("inference_time", 0.0)

        regions = _parse_od_results(raw, config.TASK_DENSE_REGION_CAPTION)

        return {
            "regions": regions,
            "count": len(regions),
            "inference_time": round(inference_time, 4),
        }

    except Exception as exc:
        logger.error("Dense region captioning failed: %s", exc)
        return {
            "regions": [],
            "count": 0,
            "inference_time": 0.0,
            "error": str(exc),
        }


def analyze_image(
    image: Image.Image,
    model_manager: VisionModelManager | None = None,
) -> dict[str, Any]:
    """Run a combined analysis: object detection **and** captioning.

    Returns:
        dict with keys ``objects``, ``total_objects``, ``categories``,
        ``caption``, ``inference_time``.
    """
    logger.info("Running full image analysis …")
    t0 = time.perf_counter()
    mgr = _get_manager(model_manager)

    try:
        detection = detect_objects(image, mgr)
        caption_result = generate_caption(image, mgr, detail_level="detailed")

        objects: list[dict[str, Any]] = detection.get("objects", [])

        # Count objects by category label
        category_counts: dict[str, int] = {}
        for obj in objects:
            lbl: str = obj.get("label", "unknown")
            category_counts[lbl] = category_counts.get(lbl, 0) + 1

        total_time = time.perf_counter() - t0

        return {
            "objects": objects,
            "total_objects": len(objects),
            "categories": category_counts,
            "caption": caption_result.get("caption", ""),
            "detection_inference_time": detection.get("inference_time", 0.0),
            "caption_inference_time": caption_result.get("inference_time", 0.0),
            "inference_time": round(total_time, 4),
        }

    except Exception as exc:
        total_time = time.perf_counter() - t0
        logger.error("Full image analysis failed: %s", exc)
        return {
            "objects": [],
            "total_objects": 0,
            "categories": {},
            "caption": "",
            "inference_time": round(total_time, 4),
            "error": str(exc),
        }
