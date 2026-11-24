"""
On-Device Vision AI — Intelligent Image Analyzer
=================================================
Main Streamlit application entry point.

This application uses Microsoft Florence-2 for local, private computer vision
inference. No images are sent to external APIs.

Usage:
    streamlit run app.py
"""

import streamlit as st
import logging
import sys
import os

# ---------------------------------------------------------------------------
# Path setup – ensure the project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from config import (
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION,
    OUTPUT_DIR,
)
from model import VisionModelManager
from vision_tasks import (
    detect_objects,
    generate_caption,
    extract_text_ocr,
    ground_object,
    get_dense_captions,
    analyze_image,
)
from image_utils import (
    load_image,
    draw_bounding_boxes,
    draw_grounding_results,
    image_to_bytes,
    get_image_metadata,
)
from performance import PerformanceMonitor
from components.styles import apply_custom_css
from components.sidebar import render_sidebar
from components.upload import render_upload_section
from components.results import (
    render_detection_results,
    render_caption_results,
    render_ocr_results,
    render_grounding_results,
    render_analysis_results,
    render_performance_metrics,
)
from components.dashboard import render_dashboard_header, render_about_page

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Apply custom CSS
# ---------------------------------------------------------------------------
apply_custom_css()

# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _get_model_manager() -> VisionModelManager:
    """Return the cached model manager (singleton via session state)."""
    if "model_manager" not in st.session_state:
        st.session_state.model_manager = VisionModelManager.get_instance()
    return st.session_state.model_manager


def _get_perf_monitor() -> PerformanceMonitor:
    """Return the cached performance monitor."""
    if "perf_monitor" not in st.session_state:
        st.session_state.perf_monitor = PerformanceMonitor()
    return st.session_state.perf_monitor


def _ensure_model_loaded(model_manager: VisionModelManager) -> bool:
    """Load the model if not already loaded. Returns True on success."""
    if model_manager.is_loaded():
        return True

    with st.spinner("🔄 Loading Florence-2 model… (first run may download ~500 MB)"):
        try:
            model_manager.load_model()
            perf = _get_perf_monitor()
            perf.model_load_time = model_manager._load_time
            st.success("✅ Model loaded successfully!")
            return True
        except Exception as exc:
            st.error(f"❌ Failed to load model: {exc}")
            logger.exception("Model loading failed")
            return False


# =====================================================================
# Page renderers
# =====================================================================

def page_dashboard() -> None:
    """Render the main dashboard / landing page."""
    render_dashboard_header()

    model_manager = _get_model_manager()
    model_loaded = _ensure_model_loaded(model_manager)

    st.markdown("---")

    # Quick-start cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """<div class="result-card">
            <h3>🔍 Object Detection</h3>
            <p style="color:#a1a1aa;">Detect and locate objects in your images with bounding boxes.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """<div class="result-card">
            <h3>📝 Image Captioning</h3>
            <p style="color:#a1a1aa;">Generate natural-language descriptions of image content.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """<div class="result-card">
            <h3>📄 OCR</h3>
            <p style="color:#a1a1aa;">Extract text from images – signs, documents, screenshots.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(
            """<div class="result-card">
            <h3>🎯 Object Grounding</h3>
            <p style="color:#a1a1aa;">Locate specific objects by name within an image.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            """<div class="result-card">
            <h3>❓ Visual Q&A</h3>
            <p style="color:#a1a1aa;">Ask natural-language questions about image content.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col6:
        st.markdown(
            """<div class="result-card">
            <h3>⚡ Performance</h3>
            <p style="color:#a1a1aa;">Real-time inference benchmarks and hardware monitoring.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    # Model info footer
    if model_loaded:
        info = model_manager.get_model_info()
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model", info.get("model_id", "N/A").split("/")[-1])
        m2.metric("Device", str(info.get("device", "N/A")).upper())
        m3.metric("Precision", str(info.get("dtype", "N/A")))
        m4.metric("Load Time", f"{info.get('load_time', 0):.1f}s")


def page_object_detection() -> None:
    """Object detection page."""
    st.markdown(
        '<h2 style="color:#f5f5f5;">🔍 Object Detection</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#a1a1aa;">Detect and locate objects in your image using Florence-2.</p>',
        unsafe_allow_html=True,
    )

    model_manager = _get_model_manager()
    if not _ensure_model_loaded(model_manager):
        return

    uploaded_file, image = render_upload_section()
    if image is None:
        return

    if st.button("🔍 Detect Objects", type="primary", use_container_width=True):
        with st.spinner("Running object detection…"):
            results = detect_objects(image, model_manager)
            perf = _get_perf_monitor()
            perf.record_inference(results.get("inference_time", 0))

        if "error" in results:
            st.error(f"Detection failed: {results['error']}")
            return

        annotated = draw_bounding_boxes(image, results.get("objects", []))
        render_detection_results(results, annotated)


def page_image_captioning() -> None:
    """Image captioning page."""
    st.markdown(
        '<h2 style="color:#f5f5f5;">📝 Image Captioning</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#a1a1aa;">Generate a natural-language description of the image.</p>',
        unsafe_allow_html=True,
    )

    model_manager = _get_model_manager()
    if not _ensure_model_loaded(model_manager):
        return

    uploaded_file, image = render_upload_section()
    if image is None:
        return

    detail = st.selectbox(
        "Caption detail level",
        options=["normal", "detailed", "more_detailed"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    if st.button("📝 Generate Caption", type="primary", use_container_width=True):
        with st.spinner("Generating caption…"):
            results = generate_caption(image, model_manager, detail_level=detail)
            perf = _get_perf_monitor()
            perf.record_inference(results.get("inference_time", 0))

        if "error" in results:
            st.error(f"Captioning failed: {results['error']}")
            return

        render_caption_results(results)


def page_ocr() -> None:
    """OCR page."""
    st.markdown(
        '<h2 style="color:#f5f5f5;">📄 Optical Character Recognition</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#a1a1aa;">Extract text from images – signs, documents, screenshots, and more.</p>',
        unsafe_allow_html=True,
    )

    model_manager = _get_model_manager()
    if not _ensure_model_loaded(model_manager):
        return

    uploaded_file, image = render_upload_section()
    if image is None:
        return

    if st.button("📄 Extract Text", type="primary", use_container_width=True):
        with st.spinner("Extracting text…"):
            results = extract_text_ocr(image, model_manager)
            perf = _get_perf_monitor()
            perf.record_inference(results.get("inference_time", 0))

        if "error" in results:
            st.error(f"OCR failed: {results['error']}")
            return

        render_ocr_results(results)


def page_object_grounding() -> None:
    """Object grounding page."""
    st.markdown(
        '<h2 style="color:#f5f5f5;">🎯 Object Grounding</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#a1a1aa;">Enter an object name to locate it within the image.</p>',
        unsafe_allow_html=True,
    )

    model_manager = _get_model_manager()
    if not _ensure_model_loaded(model_manager):
        return

    uploaded_file, image = render_upload_section()
    if image is None:
        return

    query = st.text_input(
        "Object to find",
        placeholder="e.g. person, car, dog, cat…",
    )

    if st.button("🎯 Find Object", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter an object name.")
            return

        with st.spinner(f"Searching for '{query}'…"):
            results = ground_object(image, model_manager, query.strip())
            perf = _get_perf_monitor()
            perf.record_inference(results.get("inference_time", 0))

        if "error" in results:
            st.error(f"Grounding failed: {results['error']}")
            return

        annotated = draw_grounding_results(image, results.get("objects", []))
        render_grounding_results(results, annotated)


def page_visual_qa() -> None:
    """Visual Question Answering page."""
    st.markdown(
        '<h2 style="color:#f5f5f5;">❓ Visual Question Answering</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#a1a1aa;">Ask questions about the content of your image.</p>',
        unsafe_allow_html=True,
    )

    model_manager = _get_model_manager()
    if not _ensure_model_loaded(model_manager):
        return

    uploaded_file, image = render_upload_section()
    if image is None:
        return

    question = st.text_input(
        "Your question",
        placeholder="e.g. What is the person holding? How many cars are there?",
    )

    st.info(
        "ℹ️ Visual Q&A uses Florence-2's detailed captioning capability to describe "
        "the image. For best results, ask descriptive questions. Florence-2-base does "
        "not support open-ended VQA natively — answers are derived from detailed "
        "image analysis.",
    )

    if st.button("❓ Ask Question", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Analyzing image…"):
            # Use detailed caption + dense region captions as a combined analysis
            caption_result = generate_caption(
                image, model_manager, detail_level="more_detailed"
            )
            dense_result = get_dense_captions(image, model_manager)

            perf = _get_perf_monitor()
            total_time = caption_result.get("inference_time", 0) + dense_result.get(
                "inference_time", 0
            )
            perf.record_inference(total_time)

        if "error" in caption_result:
            st.error(f"Analysis failed: {caption_result['error']}")
            return

        # Display Q&A results
        st.markdown(
            f"""<div class="result-card">
            <h4 style="color:#6366f1;">Question</h4>
            <p style="color:#f5f5f5; font-size:1.1em;">{question}</p>
            <h4 style="color:#6366f1; margin-top:1em;">Image Analysis</h4>
            <p style="color:#f5f5f5;">{caption_result.get('caption', 'No analysis available.')}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Dense captions if available
        dense_captions = dense_result.get("captions", [])
        if dense_captions:
            with st.expander("🔎 Detailed Region Descriptions"):
                for i, cap in enumerate(dense_captions, 1):
                    label = cap.get("label", cap) if isinstance(cap, dict) else str(cap)
                    st.markdown(f"**Region {i}:** {label}")

        st.caption(f"⏱️ Total analysis time: {total_time:.2f}s")


def page_performance() -> None:
    """Performance monitoring page."""
    st.markdown(
        '<h2 style="color:#f5f5f5;">⚡ Performance Monitor</h2>',
        unsafe_allow_html=True,
    )

    model_manager = _get_model_manager()
    perf = _get_perf_monitor()

    if model_manager.is_loaded():
        perf.model_load_time = model_manager._load_time

    render_performance_metrics(perf)


def page_about() -> None:
    """About page."""
    render_about_page()


# =====================================================================
# Routing
# =====================================================================

PAGE_MAP = {
    "🏠 Dashboard": page_dashboard,
    "🔍 Object Detection": page_object_detection,
    "📝 Image Captioning": page_image_captioning,
    "📄 OCR": page_ocr,
    "🎯 Object Grounding": page_object_grounding,
    "❓ Visual Q&A": page_visual_qa,
    "⚡ Performance": page_performance,
    "ℹ️ About": page_about,
}


def main() -> None:
    """Application entry point."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_manager = _get_model_manager()
    selected_page = render_sidebar(model_manager)

    page_fn = PAGE_MAP.get(selected_page, page_dashboard)
    page_fn()


if __name__ == "__main__":
    main()
