"""
Sidebar component for the Vision AI Streamlit application.

Renders navigation, model status, and system information in
a styled sidebar panel.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_sidebar(model_manager: Any) -> str:
    """Render the application sidebar with navigation, model status, and info.

    Args:
        model_manager: The ModelManager instance that exposes ``get_model_info()``
            returning a dict with keys such as ``name``, ``device``, ``is_loaded``,
            ``load_time``, etc.

    Returns:
        str: The label of the currently selected navigation page.
    """
    with st.sidebar:
        # ── App branding ──────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align:center; padding: 0.5rem 0 0.25rem;">
                <span style="font-size:2.5rem;">🔮</span>
                <h1 style="
                    margin: 0.1rem 0 0;
                    font-size: 1.6rem;
                    font-weight: 800;
                    background: linear-gradient(135deg, #6366f1, #8b5cf6);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    letter-spacing: -0.02em;
                ">Vision AI</h1>
                <p style="
                    margin: 0;
                    font-size: 0.75rem;
                    color: #a1a1aa;
                    text-transform: uppercase;
                    letter-spacing: 0.12em;
                    font-weight: 500;
                ">On-Device Intelligence</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────────
        st.markdown(
            '<p class="section-title">Navigation</p>',
            unsafe_allow_html=True,
        )

        pages = [
            "🏠 Dashboard",
            "🔍 Object Detection",
            "📝 Image Captioning",
            "📄 OCR",
            "🎯 Object Grounding",
            "❓ Visual Q&A",
            "⚡ Performance",
            "ℹ️ About",
        ]

        selected_page: str = st.radio(
            "Navigate",
            options=pages,
            index=0,
            label_visibility="collapsed",
            key="nav_radio",
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Model status ──────────────────────────────────────────
        st.markdown(
            '<p class="section-title">Model Status</p>',
            unsafe_allow_html=True,
        )

        model_info: dict = {}
        try:
            model_info = model_manager.get_model_info()
        except Exception:
            model_info = {}

        model_name = model_info.get("model_id", model_info.get("name", "Florence-2")).split("/")[-1]
        device = str(model_info.get("device", "cpu")).upper()
        is_loaded: bool = model_info.get("loaded", model_info.get("is_loaded", False))
        load_time = model_info.get("load_time")

        # Status badge
        if is_loaded:
            badge_class = "loaded"
            badge_dot = "●"
            badge_text = "Loaded"
        else:
            badge_class = "not-loaded"
            badge_dot = "●"
            badge_text = "Not Loaded"

        status_html = f"""
        <div style="
            background-color: #1a1a1a;
            border: 1px solid #27272a;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        ">
            <div class="info-row">
                <span class="info-label">Model</span>
                <span class="info-value" style="font-size:0.8rem;">{model_name}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Device</span>
                <span class="info-value">{device}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Status</span>
                <span class="status-badge {badge_class}">{badge_dot} {badge_text}</span>
            </div>
        """
        if is_loaded and load_time is not None:
            status_html += f"""
            <div class="info-row">
                <span class="info-label">Load Time</span>
                <span class="info-value">{load_time:.2f}s</span>
            </div>
            """
        status_html += "</div>"

        st.markdown(status_html, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Privacy badge ─────────────────────────────────────────
        st.markdown(
            """
            <div class="privacy-badge">
                🔒 100% Local Processing
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Version info ──────────────────────────────────────────
        st.markdown(
            """
            <div class="version-text">
                Vision AI v1.0.0<br>
                Built with Florence-2 &amp; Streamlit
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_page
