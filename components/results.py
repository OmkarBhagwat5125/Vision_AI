"""
Result display components for the Vision AI Streamlit application.

Each function renders a specific type of AI inference result using styled
HTML cards, charts, and Streamlit widgets.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL Image to raw bytes for download buttons."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def _inference_time_html(results: dict) -> str:
    """Return a small HTML snippet showing inference time if present."""
    inference_time = results.get("inference_time") or results.get("time")
    if inference_time is None:
        return ""
    return f"""
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 4px;
        margin-top: 0.5rem;
        font-size: 0.75rem;
        color: #a1a1aa;
    ">
        ⚡ Inference: <strong style="color:#8b5cf6;">{inference_time:.3f}s</strong>
    </div>
    """


# ────────────────────────────────────────────────────────────────
# Object Detection
# ────────────────────────────────────────────────────────────────

def render_detection_results(
    results: Dict[str, Any],
    annotated_image: Optional[Image.Image] = None,
) -> None:
    """Display object-detection results with annotated image and metrics.

    Args:
        results: Dict containing ``objects`` or ``labels`` (list of dicts with ``label``,
            ``bbox``), ``count``, and optional ``inference_time``.
        annotated_image: PIL Image with bounding boxes drawn, or ``None``.
    """
    raw_items = results.get("objects", results.get("labels", []))
    count: int = results.get("count", len(raw_items))
    labels: List[dict] = raw_items

    # ── Annotated image ───────────────────────────────────────
    if annotated_image is not None:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.image(annotated_image, caption="Detection Results", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Objects Detected</div>
                <div class="metric-value">{count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        unique_labels = set(
            lbl.get("label", "unknown") for lbl in labels
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Unique Classes</div>
                <div class="metric-value">{len(unique_labels)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Detection pills ───────────────────────────────────────
    if labels:
        pills_html = '<div style="margin: 1rem 0; display:flex; flex-wrap:wrap; gap:4px;">'
        for lbl in labels:
            name = lbl.get("label", "object")
            conf = lbl.get("confidence")
            conf_str = f'<span class="conf">{conf:.0%}</span>' if conf is not None else ""
            pills_html += f'<span class="detection-label">{name} {conf_str}</span>'
        pills_html += "</div>"
        st.markdown(pills_html, unsafe_allow_html=True)

    # ── Coordinate table ──────────────────────────────────────
    if labels:
        with st.expander("📐 Bounding Box Coordinates"):
            import pandas as pd

            rows = []
            for idx, lbl in enumerate(labels, 1):
                bbox = lbl.get("bbox", [])
                row: dict = {"#": idx, "Label": lbl.get("label", "")}
                if lbl.get("confidence") is not None:
                    row["Confidence"] = f"{lbl['confidence']:.2%}"
                if len(bbox) == 4:
                    row["x1"] = bbox[0]
                    row["y1"] = bbox[1]
                    row["x2"] = bbox[2]
                    row["y2"] = bbox[3]
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Download button ───────────────────────────────────────
    if annotated_image is not None:
        st.download_button(
            label="⬇️ Download Annotated Image",
            data=_image_to_bytes(annotated_image),
            file_name="detection_result.png",
            mime="image/png",
            key="download_detection",
        )

    # ── Inference time ────────────────────────────────────────
    st.markdown(_inference_time_html(results), unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# Image Captioning
# ────────────────────────────────────────────────────────────────

def render_caption_results(results: Dict[str, Any]) -> None:
    """Display image-captioning results.

    Args:
        results: Dict containing ``caption`` (str) and optional
            ``inference_time``.
    """
    caption: str = results.get("caption", "No caption generated.")

    st.markdown(
        f"""
        <div class="result-card">
            <p class="section-title">Generated Caption</p>
            <div class="caption-text">"{caption}"</div>
            {_inference_time_html(results)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Copy-friendly text area
    st.text_area(
        "Copy caption",
        value=caption,
        height=80,
        label_visibility="collapsed",
        key="caption_copy",
    )


# ────────────────────────────────────────────────────────────────
# OCR
# ────────────────────────────────────────────────────────────────

def render_ocr_results(results: Dict[str, Any]) -> None:
    """Display OCR (text extraction) results.

    Args:
        results: Dict containing ``text`` (str) and optional
            ``inference_time``.
    """
    text: str = results.get("text", "").strip()

    if not text:
        st.markdown(
            """
            <div class="result-card" style="text-align:center;">
                <p style="color:#a1a1aa; font-size:1rem; margin:1rem 0;">
                    🔍 No text found in this image.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="result-card">
            <p class="section-title">Extracted Text</p>
            <div class="ocr-text">{text}</div>
            {_inference_time_html(results)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Copy-friendly text area
    st.text_area(
        "Copy extracted text",
        value=text,
        height=160,
        label_visibility="collapsed",
        key="ocr_copy",
    )


# ────────────────────────────────────────────────────────────────
# Object Grounding
# ────────────────────────────────────────────────────────────────

def render_grounding_results(
    results: Dict[str, Any],
    annotated_image: Optional[Image.Image] = None,
) -> None:
    """Display grounding results (phrase-to-box).

    Args:
        results: Dict containing ``objects`` or ``labels`` (list of dicts with ``label``,
            ``bbox``), ``count``, ``found`` (bool), and optional
            ``inference_time``.
        annotated_image: PIL Image with grounding boxes drawn, or ``None``.
    """
    labels: List[dict] = results.get("objects", results.get("labels", []))
    found: bool = results.get("found", len(labels) > 0)
    count: int = results.get("count", len(labels))

    # ── Status ────────────────────────────────────────────────
    if found:
        st.markdown(
            f"""
            <div class="result-card">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.75rem;">
                    <span class="status-badge loaded">✓ Found</span>
                    <span style="color:#a1a1aa; font-size:0.85rem;">
                        {count} object{'s' if count != 1 else ''} located
                    </span>
                </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-card" style="text-align:center;">
                <span class="status-badge not-loaded">✗ Not Found</span>
                <p style="color:#a1a1aa; margin-top:0.5rem; font-size:0.9rem;">
                    The specified object could not be located in the image.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(_inference_time_html(results), unsafe_allow_html=True)
        return

    # ── Annotated image ───────────────────────────────────────
    if annotated_image is not None:
        st.image(annotated_image, caption="Grounding Results", use_container_width=True)

    # ── Location pills ────────────────────────────────────────
    if labels:
        pills_html = '<div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:0.5rem;">'
        for lbl in labels:
            name = lbl.get("label", "object")
            pills_html += f'<span class="detection-label">{name}</span>'
        pills_html += "</div>"
        st.markdown(pills_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Download ──────────────────────────────────────────────
    if annotated_image is not None:
        st.download_button(
            label="⬇️ Download Grounded Image",
            data=_image_to_bytes(annotated_image),
            file_name="grounding_result.png",
            mime="image/png",
            key="download_grounding",
        )

    st.markdown(_inference_time_html(results), unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# Full Analysis Dashboard
# ────────────────────────────────────────────────────────────────

def render_analysis_results(results: Dict[str, Any]) -> None:
    """Render a combined analysis dashboard card.

    Args:
        results: Dict that may contain ``caption``, ``objects``,
            ``total_objects``, ``categories``, and ``inference_time``.
    """
    caption: str = results.get("caption", "")
    detection: dict = results.get("detection", {})
    det_count: int = results.get(
        "total_objects",
        detection.get("count", len(results.get("objects", []))),
    )
    categories: dict = results.get("categories", {})

    # ── Metric columns ────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Objects</div>
                <div class="metric-value">{det_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Categories</div>
                <div class="metric-value">{len(categories)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Caption</div>
                <div class="metric-value" style="font-size:1rem;">{'✓' if caption else '—'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Category bar chart ────────────────────────────────────
    if categories:
        st.markdown(
            '<p class="section-title" style="margin-top:1rem;">Objects by Category</p>',
            unsafe_allow_html=True,
        )
        import pandas as pd

        cat_df = pd.DataFrame(
            {"Category": list(categories.keys()), "Count": list(categories.values())}
        ).sort_values("Count", ascending=False)
        st.bar_chart(cat_df.set_index("Category"), height=260)

    # ── Caption ───────────────────────────────────────────────
    if caption:
        st.markdown(
            f"""
            <div class="result-card" style="margin-top:1rem;">
                <p class="section-title">Caption</p>
                <div class="caption-text">"{caption}"</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(_inference_time_html(results), unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# Performance Metrics
# ────────────────────────────────────────────────────────────────

def render_performance_metrics(perf_monitor: Any) -> None:
    """Render a performance dashboard using data from a performance monitor.

    Args:
        perf_monitor: Object exposing ``get_metrics()`` → dict with keys
            such as ``model_load_time``, ``avg_inference_time``,
            ``total_images_processed``, ``device``, ``memory_usage_mb``.
    """
    metrics: dict = {}
    try:
        metrics = perf_monitor.get_metrics()
    except Exception:
        metrics = {}

    model_load_time = metrics.get("model_load_time")
    avg_inference = metrics.get("avg_inference_time")
    total_images = metrics.get("total_images_processed", 0)
    device = metrics.get("device", "N/A")
    memory_mb = metrics.get("memory_usage_mb")

    st.markdown(
        """
        <div class="header-section" style="padding:1.5rem 0 1rem;">
            <h2 style="
                font-size:1.5rem;
                font-weight:700;
                color:#f5f5f5;
                margin:0;
            ">⚡ Performance Dashboard</h2>
            <p class="subtitle" style="font-size:0.85rem;">Real-time model performance metrics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Top-level metric cards ────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        load_str = f"{model_load_time:.2f}s" if model_load_time is not None else "—"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model Load Time</div>
                <div class="metric-value">{load_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        avg_str = f"{avg_inference:.3f}s" if avg_inference is not None else "—"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Avg Inference</div>
                <div class="metric-value">{avg_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Images Processed</div>
                <div class="metric-value">{total_images}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Device</div>
                <div class="metric-value" style="font-size:1.2rem;">{device}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Memory usage ──────────────────────────────────────────
    if memory_mb is not None:
        st.markdown(
            f"""
            <div class="result-card" style="margin-top:1rem;">
                <p class="section-title">Memory Usage</p>
                <div class="info-row">
                    <span class="info-label">Current</span>
                    <span class="info-value">{memory_mb:.1f} MB</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Inference history (if available) ──────────────────────
    history: list = metrics.get("inference_history", [])
    if history:
        st.markdown(
            '<p class="section-title" style="margin-top:1rem;">Inference Time History</p>',
            unsafe_allow_html=True,
        )
        import pandas as pd

        hist_df = pd.DataFrame(
            {"Run": list(range(1, len(history) + 1)), "Time (s)": history}
        )
        st.line_chart(hist_df.set_index("Run"), height=220)
