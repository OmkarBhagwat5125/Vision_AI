"""
Custom CSS styles for the Vision AI Streamlit application.

Provides a comprehensive dark-theme design system with modern UI components,
smooth transitions, and professional styling for an AI dashboard.
"""

import streamlit as st


def get_custom_css() -> str:
    """Return the complete custom CSS stylesheet for the Vision AI application.

    Returns:
        str: A full CSS stylesheet wrapped in <style> tags, ready for
             injection via st.markdown with unsafe_allow_html=True.
    """
    return """
    <style>
    /* ============================================================
       GLOBAL / PAGE BACKGROUND
       ============================================================ */
    .stApp {
        background-color: #0a0a0a !important;
        color: #f5f5f5 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     Oxygen, Ubuntu, Cantarell, "Fira Sans", "Droid Sans",
                     "Helvetica Neue", Arial, sans-serif !important;
    }

    /* Hide default Streamlit chrome -------------------------------- */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header[data-testid="stHeader"] {
        background-color: #0a0a0a !important;
        backdrop-filter: blur(10px);
    }
    .stDeployButton {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}

    /* ============================================================
       SIDEBAR
       ============================================================ */
    section[data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #27272a !important;
        padding-top: 1rem;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #f5f5f5 !important;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        color: #a1a1aa !important;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        background-color: transparent !important;
        border-radius: 8px;
        padding: 0.55rem 0.75rem !important;
        margin-bottom: 2px;
        transition: background-color 0.2s ease, color 0.2s ease;
        color: #a1a1aa !important;
        cursor: pointer;
    }
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        background-color: #1a1a1a !important;
        color: #f5f5f5 !important;
    }
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[aria-checked="true"] {
        background-color: rgba(99, 102, 241, 0.15) !important;
        color: #6366f1 !important;
        font-weight: 600;
    }

    /* ============================================================
       CARD COMPONENTS
       ============================================================ */
    .result-card {
        background-color: #1a1a1a;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .result-card:hover {
        border-color: #6366f1;
        box-shadow: 0 4px 32px rgba(99, 102, 241, 0.08);
    }

    .metric-card {
        background-color: #1a1a1a;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        border-radius: 12px 12px 0 0;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.12);
    }
    .metric-card .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f5f5f5;
        margin: 0.25rem 0;
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #a1a1aa;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ============================================================
       HEADER SECTION
       ============================================================ */
    .header-section {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .header-section h1 {
        font-size: 2.75rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .header-section .subtitle {
        color: #a1a1aa;
        font-size: 1.05rem;
        font-weight: 400;
        letter-spacing: 0.02em;
    }

    /* ============================================================
       STATUS BADGES
       ============================================================ */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .status-badge.loaded {
        background-color: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .status-badge.loading {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .status-badge.error, .status-badge.not-loaded {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* ============================================================
       UPLOAD AREA
       ============================================================ */
    .upload-area {
        border: 2px dashed #27272a;
        border-radius: 12px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        background-color: #111111;
        transition: border-color 0.25s ease, background-color 0.25s ease;
    }
    .upload-area:hover {
        border-color: #6366f1;
        background-color: rgba(99, 102, 241, 0.04);
    }
    .upload-area .upload-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .upload-area .upload-text {
        color: #a1a1aa;
        font-size: 0.9rem;
    }

    /* Streamlit file uploader overrides */
    div[data-testid="stFileUploader"] {
        background-color: transparent !important;
    }
    div[data-testid="stFileUploader"] section {
        background-color: #111111 !important;
        border: 2px dashed #27272a !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        transition: border-color 0.25s ease;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #6366f1 !important;
    }

    /* ============================================================
       BUTTONS
       ============================================================ */
    .stButton > button {
        background-color: #6366f1 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        transition: background-color 0.2s ease, transform 0.15s ease,
                    box-shadow 0.2s ease !important;
        letter-spacing: 0.02em;
    }
    .stButton > button:hover {
        background-color: #4f46e5 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3) !important;
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    .stDownloadButton > button {
        background-color: transparent !important;
        color: #6366f1 !important;
        border: 1px solid #6366f1 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 600 !important;
        transition: background-color 0.2s ease, color 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        background-color: rgba(99, 102, 241, 0.1) !important;
        color: #818cf8 !important;
    }

    /* ============================================================
       CODE / TEXT OUTPUT
       ============================================================ */
    .stCodeBlock, pre, code {
        background-color: #111111 !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
        font-family: "JetBrains Mono", "Fira Code", "Cascadia Code",
                     "SF Mono", Consolas, "Liberation Mono", Menlo,
                     monospace !important;
        font-size: 0.85rem !important;
    }
    .stTextArea textarea {
        background-color: #111111 !important;
        color: #f5f5f5 !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
        font-family: "JetBrains Mono", "Fira Code", Consolas, monospace !important;
        font-size: 0.85rem !important;
    }

    /* ============================================================
       DETECTION LABEL (pill badge for detected objects)
       ============================================================ */
    .detection-label {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin: 3px;
        white-space: nowrap;
    }
    .detection-label .conf {
        font-weight: 400;
        opacity: 0.75;
        font-size: 0.7rem;
    }

    /* ============================================================
       CAPTION TEXT
       ============================================================ */
    .caption-text {
        background-color: #1a1a1a;
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 1.25rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 400;
        line-height: 1.65;
        color: #f5f5f5;
        margin: 1rem 0;
        font-style: italic;
    }

    /* ============================================================
       OCR TEXT
       ============================================================ */
    .ocr-text {
        background-color: #111111;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 1.25rem;
        font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #e4e4e7;
        white-space: pre-wrap;
        word-wrap: break-word;
        max-height: 400px;
        overflow-y: auto;
    }

    /* ============================================================
       UTILITY / LAYOUT HELPERS
       ============================================================ */
    .divider {
        border: none;
        border-top: 1px solid #27272a;
        margin: 1rem 0;
    }

    .section-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #a1a1aa;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.35rem 0;
        font-size: 0.85rem;
    }
    .info-row .info-label {
        color: #a1a1aa;
        font-weight: 500;
    }
    .info-row .info-value {
        color: #f5f5f5;
        font-weight: 600;
    }

    .privacy-badge {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(34, 197, 94, 0.05));
        border: 1px solid rgba(34, 197, 94, 0.25);
        border-radius: 8px;
        padding: 0.6rem 0.75rem;
        text-align: center;
        font-size: 0.8rem;
        color: #22c55e;
        font-weight: 600;
    }

    .version-text {
        text-align: center;
        font-size: 0.7rem;
        color: #52525b;
        margin-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Expanders */
    details[data-testid="stExpander"] {
        background-color: #1a1a1a !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
    }
    details[data-testid="stExpander"] summary {
        color: #a1a1aa !important;
        font-weight: 500;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #111111;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #a1a1aa;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1a1a !important;
        color: #6366f1 !important;
    }

    /* Metric overrides */
    div[data-testid="stMetric"] {
        background-color: #1a1a1a;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 1rem;
    }
    div[data-testid="stMetric"] label {
        color: #a1a1aa !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f5f5f5 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    ::-webkit-scrollbar-thumb {
        background: #27272a;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #3f3f46;
    }

    /* Image containers */
    .stImage {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """


def apply_custom_css() -> None:
    """Inject the custom CSS into the current Streamlit page.

    Call this once at the top of your main app entry point to apply the
    full dark-theme design system.
    """
    st.markdown(get_custom_css(), unsafe_allow_html=True)
