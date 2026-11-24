"""
Image upload component for the Vision AI Streamlit application.

Provides a styled file-uploader with validation, preview, and metadata display.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import streamlit as st
from PIL import Image

from image_utils import validate_image


def render_upload_section() -> Tuple[Any, Optional[Image.Image]]:
    """Render the image upload UI section.

    Displays a styled file uploader that accepts common image formats,
    validates the uploaded file, shows a preview with metadata, and
    returns the raw uploaded file object alongside the decoded PIL Image.

    Returns:
        tuple: ``(uploaded_file, image)`` where *uploaded_file* is the
            Streamlit ``UploadedFile`` (or ``None``), and *image* is a
            ``PIL.Image.Image`` if the file is valid, otherwise ``None``.
    """
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <h3 style="
                margin: 0;
                font-size: 1.1rem;
                font-weight: 700;
                color: #f5f5f5;
            ">📤 Upload Image</h3>
            <p style="
                margin: 0.25rem 0 0;
                font-size: 0.8rem;
                color: #a1a1aa;
            ">Supported formats: JPG, JPEG, PNG, WEBP &nbsp;•&nbsp; Max 10 MB</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key="image_uploader",
    )

    if uploaded_file is None:
        # Show empty-state hint
        st.markdown(
            """
            <div class="upload-area">
                <div class="upload-icon">🖼️</div>
                <div class="upload-text">
                    Drag &amp; drop an image here or click <strong>Browse files</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return None, None

    # ── Validate the image ────────────────────────────────────────
    is_valid, error_message = validate_image(uploaded_file)

    if not is_valid:
        st.markdown(
            f"""
            <div style="
                background-color: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                border-radius: 8px;
                padding: 1rem;
                color: #ef4444;
                font-size: 0.9rem;
            ">
                ⚠️ <strong>Invalid Image:</strong> {error_message}
            </div>
            """,
            unsafe_allow_html=True,
        )
        return uploaded_file, None

    # Reset stream position after validation and open the image
    uploaded_file.seek(0)
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Failed to open image: {exc}")
        return uploaded_file, None

    # ── Preview & metadata ────────────────────────────────────────
    width, height = image.size
    file_size_kb = uploaded_file.size / 1024
    file_size_str = (
        f"{file_size_kb:.1f} KB"
        if file_size_kb < 1024
        else f"{file_size_kb / 1024:.2f} MB"
    )
    file_format = uploaded_file.name.rsplit(".", 1)[-1].upper() if "." in uploaded_file.name else "N/A"

    st.markdown(
        """
        <div class="result-card" style="padding: 1rem;">
        """,
        unsafe_allow_html=True,
    )

    col_img, col_meta = st.columns([3, 2])

    with col_img:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col_meta:
        st.markdown(
            f"""
            <div style="padding: 0.5rem 0;">
                <p class="section-title" style="margin-bottom: 0.5rem;">Image Details</p>
                <div class="info-row">
                    <span class="info-label">File</span>
                    <span class="info-value" style="font-size:0.8rem; word-break:break-all;">
                        {uploaded_file.name}
                    </span>
                </div>
                <div class="info-row">
                    <span class="info-label">Dimensions</span>
                    <span class="info-value">{width} × {height}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Format</span>
                    <span class="info-value">{file_format}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Size</span>
                    <span class="info-value">{file_size_str}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    return uploaded_file, image
