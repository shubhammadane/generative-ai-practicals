"""Reference-based image quality metrics.

PSNR and SSIM are only meaningful against a ground-truth image of the same
dimensions. They are not estimated for ordinary uploads.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from core.image_processor import image_to_numpy


class MetricError(ValueError):
    """The pair of images cannot be compared."""


def _as_uint8(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return image_to_numpy(image)
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(np.round(array), 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    return array


def _require_same_shape(reference: np.ndarray, estimate: np.ndarray) -> None:
    if reference.shape != estimate.shape:
        raise MetricError(
            "PSNR and SSIM need the estimate and the reference to have the same dimensions. "
            f"Reference is {reference.shape[1]}×{reference.shape[0]}, "
            f"estimate is {estimate.shape[1]}×{estimate.shape[0]}."
        )
    if min(reference.shape[0], reference.shape[1]) < 7:
        raise MetricError("SSIM needs both sides of the image to be at least 7 pixels.")


def compute_psnr(reference: Image.Image | np.ndarray, estimate: Image.Image | np.ndarray) -> float:
    ref = _as_uint8(reference)
    est = _as_uint8(estimate)
    _require_same_shape(ref, est)
    value = peak_signal_noise_ratio(ref, est, data_range=255)
    return float(value)


def compute_ssim(reference: Image.Image | np.ndarray, estimate: Image.Image | np.ndarray) -> float:
    ref = _as_uint8(reference)
    est = _as_uint8(estimate)
    _require_same_shape(ref, est)
    value = structural_similarity(ref, est, data_range=255, channel_axis=-1)
    return float(value)


def compare_to_reference(
    reference: Image.Image | np.ndarray,
    estimate: Image.Image | np.ndarray,
) -> dict[str, float]:
    return {
        "psnr": compute_psnr(reference, estimate),
        "ssim": compute_ssim(reference, estimate),
    }


def format_psnr(value: float) -> str:
    if value == float("inf"):
        return "infinite (identical images)"
    return f"{value:.3f} dB"


def format_ssim(value: float) -> str:
    return f"{value:.4f}"
