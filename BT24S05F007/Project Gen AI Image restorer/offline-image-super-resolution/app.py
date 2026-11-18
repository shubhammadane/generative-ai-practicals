"""OFFLINE AI IMAGE ENHANCER — local Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from config.settings import MODEL_CATALOG
from core.image_processor import file_size_label
from core.inference import InferenceError, enhance_image
from core.validation import ImageValidationError, validate_image_bytes
from ui.components import load_stats, make_demo_image, save_stats, show_result
from ui.sidebar import render_sidebar
from ui.styles import inject_styles


def _bump_stats(elapsed: float) -> None:
    stats = load_stats()
    count = int(stats.get("processed_count", 0)) + 1
    total = float(stats.get("total_seconds", 0.0)) + elapsed
    save_stats(count, total)
    st.session_state["processed_count"] = count
    st.session_state["total_seconds"] = total


def _dashboard(settings: dict) -> None:
    stats = load_stats()
    st.title("OFFLINE AI IMAGE ENHANCER")
    st.caption("Local super-resolution on this Mac. No cloud inference and no API calls while you enhance an image.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inference", "Local")
    c2.metric("Device", settings.get("resolved_label", "—"))
    c3.metric("Images processed", int(stats.get("processed_count", 0)))
    c4.metric("Total time", f"{float(stats.get('total_seconds', 0.0)):.1f} s")
    if settings.get("ready"):
        spec = MODEL_CATALOG[settings["model_id"]]
        st.caption(f"Selected model: {spec['label']} · {spec['version']} · weights loaded from disk")


def _process(image, filename: str, raw_size: int, settings: dict, progress, status):
    def callback(done: int, total: int) -> None:
        progress.progress(done / total if total else 0)
        status.write(f"Tile {done} of {total}" if total > 1 else "Running inference")

    status.write("Loading weights and running inference")
    result = enhance_image(
        image,
        model_id=settings["model_id"],
        scale=settings["scale"],
        device_preference=settings["device"],
        tile=settings["tile"],
        filename=filename,
        jpeg_quality=settings["quality"],
        input_bytes=raw_size,
        progress=callback,
    )
    progress.progress(1.0)
    status.write(f"Finished on {result.device_label} in {result.elapsed_seconds:.2f} s")
    _bump_stats(result.elapsed_seconds)
    return result


def main() -> None:
    st.set_page_config(page_title="OFFLINE AI IMAGE ENHANCER", page_icon="◆", layout="wide")
    inject_styles()
    settings = render_sidebar()
    _dashboard(settings)
    if not settings.get("ready"):
        st.error("Download at least one model into models/ before enhancing. The general x4v3 file is the small one.")
        return

    single, batch = st.tabs(["Enhance", "Batch"])

    with single:
        uploaded = st.file_uploader(
            "Drop a JPG, PNG, or WEBP image",
            type=["jpg", "jpeg", "png", "webp"],
            key="single_upload",
        )
        use_demo = st.button("Use built-in sample")
        image = None
        filename = "sample"
        raw_size = 0
        if uploaded is not None:
            raw = uploaded.getvalue()
            raw_size = len(raw)
            try:
                image = validate_image_bytes(uploaded.name, raw)
                filename = uploaded.name
            except ImageValidationError as exc:
                st.error(str(exc))
        elif use_demo or st.session_state.get("use_demo"):
            if use_demo:
                st.session_state["use_demo"] = True
            image = make_demo_image()
            filename = "builtin-sample.png"
            raw_size = len(image.tobytes())

        if image is not None:
            meta1, meta2, meta3 = st.columns(3)
            meta1.metric("Resolution", f"{image.size[0]}×{image.size[1]}")
            meta2.metric("Format", Path(filename).suffix.lower().lstrip(".") or "png")
            meta3.metric("File size", file_size_label(raw_size))
            st.image(image, width=420)
            if st.button("Start enhancement", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                try:
                    st.session_state["last_result"] = _process(image, filename, raw_size, settings, progress, status)
                    st.session_state["last_original"] = image
                    st.session_state["last_raw_size"] = raw_size
                except (InferenceError, ImageValidationError, Exception) as exc:  # noqa: BLE001
                    st.error(str(exc))
        if st.session_state.get("last_result") is not None and st.session_state.get("last_original") is not None:
            st.subheader("Before and after")
            show_result(
                st.session_state["last_original"],
                st.session_state["last_result"],
                int(st.session_state.get("last_raw_size") or 0),
            )
            st.caption(
                "PSNR and SSIM are not shown here. They need a high-resolution reference, "
                "and an ordinary upload does not have one."
            )

    with batch:
        files = st.file_uploader(
            "Add several images. They are processed one at a time.",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="batch_upload",
        )
        if files and st.button("Process batch"):
            results = []
            bar = st.progress(0)
            for index, item in enumerate(files, start=1):
                st.write(f"{index}/{len(files)} · {item.name}")
                try:
                    raw = item.getvalue()
                    image = validate_image_bytes(item.name, raw)
                    result = _process(image, item.name, len(raw), settings, st.progress(0), st.empty())
                    results.append((item.name, image, raw, result, None))
                except Exception as exc:  # noqa: BLE001
                    results.append((item.name, None, 0, None, str(exc)))
                bar.progress(index / len(files))
            st.session_state["batch_results"] = results
        for index, (name, image, raw_size, result, error) in enumerate(st.session_state.get("batch_results", [])):
            st.markdown(f"**{name}**")
            if error:
                st.error(error)
            elif result is not None and image is not None:
                show_result(image, result, raw_size, key_prefix=f"batch_{index}")


if __name__ == "__main__":
    main()
