"""Validate uploads before they reach the model."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from config.settings import MAX_UPLOAD_BYTES, SUPPORTED_EXTENSIONS
from core.device_manager import max_input_edge

_MAGIC = {
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".webp": (b"RIFF",),
}


class ImageValidationError(ValueError):
    """The file cannot be processed as a supported image."""


def _extension(filename: str) -> str:
    return Path(filename or "").suffix.lower()


def _magic_matches(extension: str, data: bytes) -> bool:
    signatures = _MAGIC.get(extension, ())
    if not signatures:
        return False
    if extension == ".webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return any(data.startswith(signature) for signature in signatures)


def validate_image_bytes(filename: str, data: bytes) -> Image.Image:
    """Return an RGB image or raise ImageValidationError with a useful reason."""
    if not filename or not str(filename).strip():
        raise ImageValidationError("The upload has no filename.")
    if data is None or len(data) == 0:
        raise ImageValidationError("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        limit = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise ImageValidationError(
            f"The file is larger than {limit} MB. Resize or compress it before uploading."
        )

    extension = _extension(filename)
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ImageValidationError(
            f"Unsupported format '{extension or 'unknown'}'. Supported formats: {allowed}."
        )
    if not _magic_matches(extension, data):
        raise ImageValidationError(
            f"The file contents do not match a {extension} image. The file may be renamed or damaged."
        )

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
            image = _to_rgb(opened)
    except UnidentifiedImageError as exc:
        raise ImageValidationError("The image is corrupted and could not be decoded.") from exc
    except ImageValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface decoder failures as validation errors
        raise ImageValidationError(f"The image could not be decoded: {exc}") from exc

    edge = max_input_edge()
    width, height = image.size
    if width < 16 or height < 16:
        raise ImageValidationError(
            f"The image is {width}×{height}. Both sides must be at least 16 pixels."
        )
    if max(width, height) > edge:
        raise ImageValidationError(
            f"The image is {width}×{height}. The longest side must be {edge} pixels or less "
            "so a 4× result fits in this Mac's memory. Resize the image and try again."
        )
    if width * height > edge * edge:
        raise ImageValidationError(
            f"The image has {width * height:,} pixels, which is above the limit of {edge * edge:,}. "
            "Resize the image and try again."
        )
    return image


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        return background
    if image.mode == "LA":
        rgba = image.convert("RGBA")
        return _to_rgb(rgba)
    if image.mode == "P" and "transparency" in image.info:
        return _to_rgb(image.convert("RGBA"))
    try:
        return image.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ImageValidationError(f"The image mode '{image.mode}' could not be converted to RGB.") from exc


def validate_image_path(path: Path) -> Image.Image:
    file_path = Path(path)
    if not file_path.is_file():
        raise ImageValidationError(f"Image file was not found: {file_path.name}")
    return validate_image_bytes(file_path.name, file_path.read_bytes())
