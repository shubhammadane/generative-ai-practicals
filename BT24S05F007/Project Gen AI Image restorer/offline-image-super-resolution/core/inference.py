"""Tile-aware Real-ESRGAN inference on MPS or CPU."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config.settings import JPEG_QUALITY_DEFAULT, MODEL_CATALOG, OUTPUT_DIR
from core.device_manager import (
    DeviceChoice,
    memory_snapshot,
    release_cached_memory,
    select_device,
    synchronize,
)
from core.image_processor import (
    bgr_chw_to_image,
    encode_image,
    file_size_label,
    rgb_to_bgr_tensor_array,
    save_image,
)
from core.model_manager import get_model_manager


class InferenceError(RuntimeError):
    """Inference could not finish."""


@dataclass
class EnhancementResult:
    image: Image.Image
    png_bytes: bytes
    jpeg_bytes: bytes
    elapsed_seconds: float
    device_name: str
    device_label: str
    fallback_reason: str | None
    model_id: str
    model_label: str
    model_version: str
    requested_scale: int
    native_scale: int
    strategy: str
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    input_bytes: int
    tile_size: int
    saved_names: list[str] = field(default_factory=list)

    @property
    def png_size_label(self) -> str:
        return file_size_label(len(self.png_bytes))

    @property
    def jpeg_size_label(self) -> str:
        return file_size_label(len(self.jpeg_bytes))


def suggest_tile(width: int, height: int, device_name: str) -> int:
    pixels = width * height
    if pixels <= 320 * 320:
        return 0
    if device_name == "cpu":
        return 192 if pixels > 512 * 512 else 256
    if pixels <= 640 * 640:
        return 384
    return 256


def _scale_plan(model_id: str, requested_scale: int) -> tuple[int, str]:
    spec = MODEL_CATALOG[model_id]
    native = int(spec["native_scale"])
    allowed = set(spec["output_scales"])
    if requested_scale not in allowed:
        raise InferenceError(
            f"{spec['label']} supports {sorted(allowed)}×. {requested_scale}× is not available."
        )
    if requested_scale == native:
        return native, f"Native {native}× inference."
    return native, (
        f"This network is natively {native}×. The app runs that inference, then resizes "
        f"the result to {requested_scale}× with Lanczos. This is not a native {requested_scale}× model."
    )


def _forward(model, tensor: torch.Tensor, native_scale: int, tile: int, progress) -> torch.Tensor:
    _, _, height, width = tensor.shape
    use_tile = tile if tile and (height > tile or width > tile) else 0
    with torch.inference_mode():
        if use_tile <= 0:
            if progress:
                progress(0, 1)
            output = model(tensor)
            if progress:
                progress(1, 1)
            return output

        output = tensor.new_zeros(1, 3, height * native_scale, width * native_scale)
        tile_pad = 10
        tiles_x = math.ceil(width / use_tile)
        tiles_y = math.ceil(height / use_tile)
        total = tiles_x * tiles_y
        done = 0
        if progress:
            progress(0, total)
        for y in range(tiles_y):
            for x in range(tiles_x):
                in_x0 = x * use_tile
                in_y0 = y * use_tile
                in_x1 = min(in_x0 + use_tile, width)
                in_y1 = min(in_y0 + use_tile, height)
                pad_x0 = max(in_x0 - tile_pad, 0)
                pad_y0 = max(in_y0 - tile_pad, 0)
                pad_x1 = min(in_x1 + tile_pad, width)
                pad_y1 = min(in_y1 + tile_pad, height)
                tile_in = tensor[:, :, pad_y0:pad_y1, pad_x0:pad_x1]
                tile_out = model(tile_in)
                out_x0 = in_x0 * native_scale
                out_y0 = in_y0 * native_scale
                out_x1 = in_x1 * native_scale
                out_y1 = in_y1 * native_scale
                crop_x0 = (in_x0 - pad_x0) * native_scale
                crop_y0 = (in_y0 - pad_y0) * native_scale
                crop_x1 = crop_x0 + (in_x1 - in_x0) * native_scale
                crop_y1 = crop_y0 + (in_y1 - in_y0) * native_scale
                output[:, :, out_y0:out_y1, out_x0:out_x1] = tile_out[:, :, crop_y0:crop_y1, crop_x0:crop_x1]
                done += 1
                if progress:
                    progress(done, total)
        return output


def _run_once(model, image: Image.Image, device: torch.device, native_scale: int, tile: int, progress):
    array = rgb_to_bgr_tensor_array(image)
    tensor = torch.from_numpy(array).to(device)
    _, _, height, width = tensor.shape
    # x2 RRDB uses pixel-unshuffle and needs even sides.
    pad_h = (2 - height % 2) % 2 if native_scale == 2 else 0
    pad_w = (2 - width % 2) % 2 if native_scale == 2 else 0
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    synchronize(device)
    started = time.perf_counter()
    output = _forward(model, tensor, native_scale, tile, progress)
    synchronize(device)
    elapsed = time.perf_counter() - started
    if pad_h or pad_w:
        output = output[:, :, : height * native_scale, : width * native_scale]
    result = bgr_chw_to_image(output.float().clamp(0, 1).cpu().numpy())
    del tensor, output
    release_cached_memory(device)
    return result, elapsed


def enhance_image(
    image: Image.Image,
    model_id: str,
    scale: int = 4,
    device_preference: str = "auto",
    tile: int | None = None,
    filename: str = "image",
    jpeg_quality: int = JPEG_QUALITY_DEFAULT,
    input_bytes: int = 0,
    progress=None,
    save: bool = True,
) -> EnhancementResult:
    spec = MODEL_CATALOG[model_id]
    native_scale, strategy = _scale_plan(model_id, scale)
    choice: DeviceChoice = select_device(device_preference)
    tile_size = suggest_tile(image.width, image.height, choice.name) if tile is None else int(tile)
    manager = get_model_manager()

    attempts = [(choice, tile_size)]
    if tile_size != 128:
        attempts.append((choice, 128))
    if choice.name != "cpu":
        attempts.append((select_device("cpu"), 128))

    last_error: Exception | None = None
    used_choice = choice
    used_tile = tile_size
    enhanced = None
    elapsed = 0.0
    for attempt_choice, attempt_tile in attempts:
        try:
            model = manager.get(model_id, attempt_choice.device)
            enhanced, elapsed = _run_once(
                model, image, attempt_choice.device, native_scale, attempt_tile, progress
            )
            used_choice = attempt_choice
            used_tile = attempt_tile
            last_error = None
            break
        except RuntimeError as exc:
            last_error = exc
            message = str(exc).lower()
            release_cached_memory(attempt_choice.device)
            if "memory" not in message and "mps" not in message:
                raise InferenceError(f"Inference failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            release_cached_memory(attempt_choice.device)
            if attempt_choice.name == "cpu" and attempt_tile == 128:
                break
    if enhanced is None:
        raise InferenceError(
            "Inference ran out of memory or the device rejected the model. "
            f"Last error: {last_error}"
        )

    if scale != native_scale:
        enhanced = enhanced.resize(
            (image.width * scale, image.height * scale),
            Image.Resampling.LANCZOS,
        )
    expected = (image.width * scale, image.height * scale)
    if enhanced.size != expected:
        raise InferenceError(
            f"Output size {enhanced.size[0]}×{enhanced.size[1]} did not match {expected[0]}×{expected[1]}."
        )

    fallback = used_choice.fallback_reason
    if used_choice.name != choice.name:
        fallback = f"Fell back to CPU after the first device failed. {last_error or ''}".strip()
    elif used_tile != tile_size and used_tile:
        strategy += f" Tile size was reduced to {used_tile} to stay within memory."

    png_bytes = encode_image(enhanced, "png")
    jpeg_bytes = encode_image(enhanced, "jpeg", jpeg_quality=jpeg_quality)
    saved_names: list[str] = []
    if save:
        stem = f"{Path(filename).stem}_{model_id}_{scale}x"
        png_path = save_image(enhanced, stem, "png", directory=OUTPUT_DIR)
        jpg_path = save_image(enhanced, stem, "jpeg", jpeg_quality=jpeg_quality, directory=OUTPUT_DIR)
        saved_names = [png_path.name, jpg_path.name]

    return EnhancementResult(
        image=enhanced,
        png_bytes=png_bytes,
        jpeg_bytes=jpeg_bytes,
        elapsed_seconds=elapsed,
        device_name=used_choice.name,
        device_label=used_choice.label,
        fallback_reason=fallback,
        model_id=model_id,
        model_label=spec["label"],
        model_version=spec["version"],
        requested_scale=scale,
        native_scale=native_scale,
        strategy=strategy,
        input_size=image.size,
        output_size=enhanced.size,
        input_bytes=input_bytes,
        tile_size=used_tile,
        saved_names=saved_names,
    )


def warmup(model_id: str, device: torch.device) -> None:
    """Run one small forward so GPU timing does not include first-launch setup."""
    model = get_model_manager().get(model_id, device)
    native = int(MODEL_CATALOG[model_id]["native_scale"])
    dummy = torch.zeros(1, 3, 32, 32, device=device)
    with torch.inference_mode():
        model(dummy)
    synchronize(device)
    release_cached_memory(device)
    del dummy
    _ = native
