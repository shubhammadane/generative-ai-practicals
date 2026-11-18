"""Shared visual styling for the local enhancer."""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
    .stApp {
        background: #0e1116;
        color: #e8edf5;
        font-family: "SF Pro Display", "Segoe UI", sans-serif;
    }
    section[data-testid="stSidebar"] {
        background: #14181f;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    h1, h2, h3 { letter-spacing: -0.03em; }
    div[data-testid="stMetric"] {
        background: #171c24;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 8px 12px;
    }
    .block-container { padding-top: 1.4rem; max-width: 1180px; }
</style>
"""


def inject_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
