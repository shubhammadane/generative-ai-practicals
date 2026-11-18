"""Sidebar controls."""

from __future__ import annotations

import streamlit as st

from config.settings import MODEL_CATALOG
from core.device_manager import select_device
from core.model_manager import available_upscalers, is_available


def render_sidebar() -> dict:
    choice = select_device("auto")
    st.sidebar.header("Settings")
    device_options = ["auto", "cpu"]
    if choice.mps_available:
        device_options.insert(1, "mps")
    device = st.sidebar.radio(
        "Execution device",
        device_options,
        format_func=lambda value: {"auto": "Auto", "mps": "Apple Metal (MPS)", "cpu": "CPU"}[value],
    )
    resolved = select_device(device)
    st.sidebar.caption(f"Will use {resolved.label}.")
    if resolved.fallback_reason:
        st.sidebar.caption(resolved.fallback_reason)
    if resolved.uses_mps_fallback_ops:
        st.sidebar.caption("Unsupported Metal operators can run on the CPU. The selected device is still MPS.")

    ready = available_upscalers()
    st.sidebar.subheader("Models on disk")
    for model_id, spec in MODEL_CATALOG.items():
        if spec.get("kind") != "upscaler":
            continue
        state = "Ready" if model_id in ready else "Not downloaded"
        st.sidebar.write(f"{spec['label']}: {state}")

    if not ready:
        st.sidebar.error("No upscaler weights found. Download them with scripts/download_models.py.")
        return {"ready": False}

    model_id = st.sidebar.selectbox(
        "Model",
        ready,
        format_func=lambda key: MODEL_CATALOG[key]["label"],
    )
    spec = MODEL_CATALOG[model_id]
    scales = list(spec["output_scales"])
    scale = st.sidebar.radio("Upscale", scales, index=scales.index(4) if 4 in scales else 0, format_func=lambda value: f"{value}×")
    if scale != spec["native_scale"]:
        st.sidebar.caption(f"{scale}× uses native {spec['native_scale']}× inference, then a Lanczos resize.")
    tile_label = st.sidebar.selectbox("Tile size", ["Auto", "Whole image", "128", "256", "384", "512"])
    tile = None if tile_label == "Auto" else (0 if tile_label == "Whole image" else int(tile_label))
    quality = st.sidebar.slider("JPEG quality", 70, 100, 92)
    st.sidebar.caption("Face enhancement is off in this build until GFPGAN weights are present.")
    return {
        "ready": True,
        "device": device,
        "resolved_label": resolved.label,
        "model_id": model_id,
        "model_version": spec["version"],
        "scale": int(scale),
        "tile": tile,
        "quality": quality,
    }
