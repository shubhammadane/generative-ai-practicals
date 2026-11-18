"""Small interface helpers."""

from __future__ import annotations

import json

import streamlit as st
from PIL import Image, ImageDraw

from config.settings import STATS_PATH
from core.image_processor import file_size_label
from core.inference import EnhancementResult


def load_stats() -> dict:
    if not STATS_PATH.is_file():
        return {"processed_count": 0, "total_seconds": 0.0}
    try:
        return json.loads(STATS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"processed_count": 0, "total_seconds": 0.0}


def save_stats(processed_count: int, total_seconds: float) -> None:
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(
        json.dumps({"processed_count": processed_count, "total_seconds": round(total_seconds, 3)})
    )


def make_demo_image() -> Image.Image:
    """A local test picture so the app can be tried without a download."""
    image = Image.new("RGB", (320, 240), (28, 36, 48))
    draw = ImageDraw.Draw(image)
    for index in range(12):
        shade = 40 + index * 12
        draw.rectangle((index * 26, 0, index * 26 + 20, 240), fill=(shade, 70, 110))
    draw.ellipse((40, 40, 160, 150), fill=(214, 168, 92))
    draw.rectangle((180, 70, 290, 180), outline=(230, 230, 230), width=3)
    draw.line((0, 200, 320, 40), fill=(90, 180, 140), width=4)
    draw.text((24, 190), "OFFLINE SAMPLE", fill=(245, 245, 245))
    return image


def show_result(original: Image.Image, result: EnhancementResult, original_bytes: int, key_prefix: str = "result") -> None:
    left, right = st.columns(2)
    with left:
        st.caption(f"Original  {original.size[0]}×{original.size[1]}  ·  {file_size_label(original_bytes)}")
        st.image(original, use_container_width=True)
    with right:
        st.caption(
            f"Enhanced  {result.output_size[0]}×{result.output_size[1]}  ·  PNG {result.png_size_label}"
        )
        st.image(result.image, use_container_width=True)

    mix = st.slider("Comparison mix", 0, 100, 50, help="0 is the original. 100 is the enhanced image.")
    preview_size = (min(original.width, 900), int(min(original.width, 900) * original.height / original.width))
    before = original.resize(preview_size, Image.Resampling.LANCZOS)
    after = result.image.resize(preview_size, Image.Resampling.LANCZOS)
    blended = Image.blend(before, after, mix / 100)
    st.image(blended, caption=f"{mix}% enhanced", use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time", f"{result.elapsed_seconds:.2f} s")
    c2.metric("Device", result.device_label)
    c3.metric("Scale", f"{result.requested_scale}×")
    c4.metric("Tile", "whole image" if result.tile_size == 0 else str(result.tile_size))
    st.caption(result.strategy)
    if result.fallback_reason:
        st.warning(result.fallback_reason)
    if result.saved_names:
        st.caption("Saved in the output folder: " + ", ".join(result.saved_names))

    d1, d2 = st.columns(2)
    stem = result.saved_names[0].rsplit(".", 1)[0] if result.saved_names else "enhanced"
    d1.download_button(
        "Download PNG",
        result.png_bytes,
        file_name=f"{stem}.png",
        mime="image/png",
        use_container_width=True,
        key=f"{key_prefix}_png",
    )
    d2.download_button(
        "Download JPEG",
        result.jpeg_bytes,
        file_name=f"{stem}.jpg",
        mime="image/jpeg",
        use_container_width=True,
        key=f"{key_prefix}_jpg",
    )
