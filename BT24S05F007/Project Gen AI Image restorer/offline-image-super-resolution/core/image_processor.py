"""Load, convert, and save images. Color stays in RGB at this boundary."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from config.settings import JPEG_QUALITY_DEFAULT, OUTPUT_DIR
from core.validation import ImageValidationError, validate_image_bytes, validate_image_path


def load_image_bytes(filename: str, data: bytes) -> Image.Image:
    return validate_image_bytes(filename, data)


def load_image_path(path: Path) -> Image.Image:
    return validate_image_path(path)


def image_to_numpy(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageValidationError("Expected a 3-channel RGB image.")
    return array


def numpy_to_image(array: np.ndarray) -> Image.Image:
    if array.dtype != np.uint8:
        array = np.clip(np.round(array), 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageValidationError(f"Cannot build an RGB image from shape {array.shape}.")
    return Image.fromarray(array, mode="RGB")


def rgb_to_bgr_tensor_array(image: Image.Image) -> np.ndarray:
    """Float32 BGR array in [0, 1], shaped (1, 3, H, W), matching Real-ESRGAN."""
    rgb = image_to_numpy(image).astype(np.float32) / 255.0
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    chw = np.transpose(bgr, (2, 0, 1))
    return np.expand_dims(chw, 0)


def bgr_chw_to_image(array: np.ndarray) -> Image.Image:
    """Convert a float BGR CHW or NCHW array in [0, 1] back to an RGB image."""
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3 or array.shape[0] != 3:
        raise ImageValidationError(f"Model output has an unexpected shape: {array.shape}.")
    bgr = np.transpose(array, (1, 2, 0))
    rgb = np.clip(np.round(bgr[:, :, ::-1] * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def resize_lanczos(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    if width < 1 or height < 1:
        raise ImageValidationError(f"Cannot resize to {width}×{height}.")
    return image.resize((width, height), Image.Resampling.LANCZOS)


def downscale_by(image: Image.Image, factor: int) -> Image.Image:
    if factor < 2:
        raise ImageValidationError("Downscale factor must be at least 2.")
    width, height = image.size
    if width % factor or height % factor:
        raise ImageValidationError(
            f"Image {width}×{height} is not divisible by {factor}, so the low-resolution "
            "pair would not map back to the reference."
        )
    return resize_lanczos(image, (width // factor, height // factor))


def encode_image(image: Image.Image, image_format: str, jpeg_quality: int = JPEG_QUALITY_DEFAULT) -> bytes:
    image_format = image_format.lower()
    if image_format not in {"png", "jpeg", "jpg"}:
        raise ImageValidationError("Output format must be PNG or JPEG.")
    buffer = io.BytesIO()
    rgb = image.convert("RGB")
    if image_format == "png":
        rgb.save(buffer, format="PNG", optimize=True)
    else:
        quality = int(jpeg_quality)
        if quality < 70 or quality > 100:
            raise ImageValidationError("JPEG quality must be between 70 and 100.")
        rgb.save(buffer, format="JPEG", quality=quality, optimize=True, subsampling=1)
    return buffer.getvalue()


def save_image(
    image: Image.Image,
    filename: str,
    image_format: str,
    jpeg_quality: int = JPEG_QUALITY_DEFAULT,
    directory: Path | None = None,
) -> Path:
    target_dir = Path(directory or OUTPUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_filename(filename)
    extension = ".png" if image_format.lower() == "png" else ".jpg"
    path = target_dir / f"{safe}{extension}"
    if path.exists():
        stem = path.stem
        index = 2
        while path.exists():
            path = target_dir / f"{stem}_{index}{extension}"
            index += 1
    data = encode_image(image, image_format, jpeg_quality=jpeg_quality)
    try:
        path.write_bytes(data)
    except OSError as exc:
        raise ImageValidationError(f"Could not save the image: {exc}") from exc
    return path


def _safe_filename(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = []
    for char in stem:
        if char.isalnum() or char in {".", "_", "-"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    safe = "".join(cleaned).strip("._")[:80]
    return safe or "enhanced"


def file_size_label(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"
