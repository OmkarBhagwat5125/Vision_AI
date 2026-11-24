"""
Dashboard and About page components for the Vision AI Streamlit application.

Provides the main landing-page header with gradient title and status metrics,
as well as an informational About page.
"""

from __future__ import annotations

import streamlit as st


def render_dashboard_header() -> None:
    """Render the main dashboard hero header with status metrics.

    Displays the application title with a gradient text effect, a subtitle,
    and three metric cards (Model Status, Inference Device, Privacy Status).
    """
    st.markdown(
        """
        <div class="header-section">
            <h1>ON-DEVICE VISION AI</h1>
            <p class="subtitle">
                Intelligent Computer Vision &nbsp;•&nbsp; Private &nbsp;•&nbsp;
                Local &nbsp;•&nbsp; Efficient
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Status metric cards ───────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Model Status</div>
                <div class="metric-value" style="font-size:1.3rem;">
                    <span class="status-badge loaded">● Ready</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # Detect device at render time
        device_label = "CPU"
        try:
            import torch
            if torch.cuda.is_available():
                device_label = "CUDA GPU"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_label = "Apple MPS"
        except ImportError:
            pass

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Inference Device</div>
                <div class="metric-value" style="font-size:1.3rem;">{device_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Privacy Status</div>
                <div class="metric-value" style="font-size:1.3rem;">🔒 100% Local</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Quick-start cards ─────────────────────────────────────
    st.markdown(
        """
        <div style="margin-top: 2rem;">
            <p class="section-title">Quick Start</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    qs1, qs2, qs3 = st.columns(3)
    with qs1:
        st.markdown(
            """
            <div class="result-card" style="text-align:center; min-height:140px;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">🔍</div>
                <strong style="color:#f5f5f5;">Object Detection</strong>
                <p style="color:#a1a1aa; font-size:0.8rem; margin-top:0.4rem;">
                    Detect and label objects in your images with bounding boxes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with qs2:
        st.markdown(
            """
            <div class="result-card" style="text-align:center; min-height:140px;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">📝</div>
                <strong style="color:#f5f5f5;">Image Captioning</strong>
                <p style="color:#a1a1aa; font-size:0.8rem; margin-top:0.4rem;">
                    Generate natural-language descriptions of image contents.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with qs3:
        st.markdown(
            """
            <div class="result-card" style="text-align:center; min-height:140px;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">📄</div>
                <strong style="color:#f5f5f5;">OCR</strong>
                <p style="color:#a1a1aa; font-size:0.8rem; margin-top:0.4rem;">
                    Extract printed and handwritten text from images.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_about_page() -> None:
    """Render the About page with project info, tech stack, and architecture."""

    st.markdown(
        """
        <div class="header-section" style="padding-bottom:0.5rem;">
            <h1 style="font-size:2rem;">About Vision AI</h1>
            <p class="subtitle">On-Device Intelligent Computer Vision</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Project description ───────────────────────────────────
    st.markdown(
        """
        <div class="result-card">
            <p class="section-title">Project Overview</p>
            <p style="color:#e4e4e7; line-height:1.7; font-size:0.95rem;">
                <strong>Vision AI</strong> is a fully local, privacy-first computer
                vision application powered by Microsoft's Florence-2 foundation model.
                All inference runs <em>on your device</em>&nbsp;— no data ever leaves
                your machine.  The app offers object detection, image captioning,
                optical character recognition (OCR), phrase grounding, and visual
                question-answering in a single unified interface.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tech stack ────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            """
            <div class="result-card" style="min-height:280px;">
                <p class="section-title">🛠️ Tech Stack</p>
                <ul style="color:#e4e4e7; line-height:2; font-size:0.9rem; padding-left:1.2rem;">
                    <li><strong>Model:</strong> Florence-2-base (Microsoft)</li>
                    <li><strong>Framework:</strong> PyTorch + HuggingFace Transformers</li>
                    <li><strong>UI:</strong> Streamlit</li>
                    <li><strong>Image processing:</strong> Pillow, OpenCV</li>
                    <li><strong>Language:</strong> Python 3.10+</li>
                    <li><strong>Hardware:</strong> CPU / CUDA / Apple MPS</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            """
            <div class="result-card" style="min-height:280px;">
                <p class="section-title">🏗️ Architecture</p>
                <ul style="color:#e4e4e7; line-height:2; font-size:0.9rem; padding-left:1.2rem;">
                    <li><strong>Modular codebase</strong> — pages, components, utils</li>
                    <li><strong>Lazy model loading</strong> — fast cold start</li>
                    <li><strong>Session-state caching</strong> — model loaded once</li>
                    <li><strong>Performance monitor</strong> — real-time metrics</li>
                    <li><strong>Responsive UI</strong> — desktop &amp; tablet friendly</li>
                    <li><strong>Custom CSS</strong> — polished dark theme</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Privacy explanation ───────────────────────────────────
    st.markdown(
        """
        <div class="result-card" style="border-left: 3px solid #22c55e;">
            <p class="section-title">🔒 Privacy &amp; Security</p>
            <p style="color:#e4e4e7; line-height:1.7; font-size:0.9rem;">
                Vision AI processes every image <strong>entirely on your local
                device</strong>.  No images, prompts, or results are transmitted to
                any remote server.  The model weights are downloaded once from
                Hugging Face and cached locally for subsequent runs.  This
                architecture makes Vision AI suitable for sensitive or confidential
                imagery where cloud-based APIs are not acceptable.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Links / placeholders ──────────────────────────────────
    st.markdown(
        """
        <div class="result-card" style="text-align:center;">
            <p class="section-title">🔗 Links</p>
            <p style="color:#a1a1aa; font-size:0.9rem; line-height:2;">
                GitHub: <a href="#" style="color:#6366f1; text-decoration:none;">github.com/your-username/vision-ai</a><br>
                Portfolio: <a href="#" style="color:#6366f1; text-decoration:none;">your-portfolio-site.com</a><br>
                License: MIT
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Footer ────────────────────────────────────────────────
    st.markdown(
        """
        <div class="version-text" style="margin-top:2rem;">
            Vision AI v1.0.0 &nbsp;•&nbsp; Built with ❤️ using Florence-2 &amp; Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )
