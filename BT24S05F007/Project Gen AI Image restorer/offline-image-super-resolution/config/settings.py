"""Paths, model catalog, and runtime limits.

Model URLs live only in scripts/download_models.py. The application reads
weights from disk and never downloads them.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
SAMPLES_DIR = ROOT / "samples"
HIGH_RES_DIR = SAMPLES_DIR / "high_resolution"
LOW_RES_DIR = SAMPLES_DIR / "low_resolution"
REPORTS_DIR = ROOT / "reports"
STATS_PATH = OUTPUT_DIR / "processing_stats.json"
SAMPLE_MANIFEST = SAMPLES_DIR / "manifest.json"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
TILE_CHOICES = (128, 192, 256, 384, 512)
JPEG_QUALITY_DEFAULT = 92

# Checksums are of the official release files. Unknown checksums are filled in
# after the first verified download from the official URL and then pinned.
MODEL_CATALOG: dict[str, dict] = {
    "realesrgan_x4plus": {
        "label": "Real-ESRGAN x4plus",
        "version": "Real-ESRGAN v0.1.0",
        "filename": "RealESRGAN_x4plus.pth",
        "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        "min_bytes": 60_000_000,
        "arch": "rrdb",
        "native_scale": 4,
        "num_block": 23,
        "num_feat": 64,
        "num_grow_ch": 32,
        "output_scales": (2, 4),
        "supports_denoise": False,
        "kind": "upscaler",
        "license": "BSD-3-Clause",
        "source": "xinntao/Real-ESRGAN release v0.1.0",
        "description": "General photographs. 23 RRDB blocks, native 4×.",
    },
    "realesrgan_x2plus": {
        "label": "Real-ESRGAN x2plus",
        "version": "Real-ESRGAN v0.2.1",
        "filename": "RealESRGAN_x2plus.pth",
        "sha256": None,
        "min_bytes": 60_000_000,
        "arch": "rrdb",
        "native_scale": 2,
        "num_block": 23,
        "num_feat": 64,
        "num_grow_ch": 32,
        "output_scales": (2,),
        "supports_denoise": False,
        "kind": "upscaler",
        "license": "BSD-3-Clause",
        "source": "xinntao/Real-ESRGAN release v0.2.1",
        "description": "General photographs. Native 2× RRDB network.",
    },
    "realesrgan_x4plus_anime": {
        "label": "Real-ESRGAN x4plus anime",
        "version": "Real-ESRGAN v0.2.2.4",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "sha256": "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
        "min_bytes": 15_000_000,
        "arch": "rrdb",
        "native_scale": 4,
        "num_block": 6,
        "num_feat": 64,
        "num_grow_ch": 32,
        "output_scales": (2, 4),
        "supports_denoise": False,
        "kind": "upscaler",
        "license": "BSD-3-Clause",
        "source": "xinntao/Real-ESRGAN release v0.2.2.4",
        "description": "Anime and illustration. 6 RRDB blocks, native 4×.",
    },
    "realesrgan_general_x4v3": {
        "label": "Real-ESRGAN general x4v3",
        "version": "Real-ESRGAN v0.2.5.0",
        "filename": "realesr-general-x4v3.pth",
        "sha256": "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        "min_bytes": 3_000_000,
        "arch": "srvgg",
        "native_scale": 4,
        "num_conv": 32,
        "num_feat": 64,
        "output_scales": (2, 4),
        "supports_denoise": True,
        "denoise_filename": "realesr-general-wdn-x4v3.pth",
        "denoise_sha256": None,
        "denoise_min_bytes": 3_000_000,
        "kind": "upscaler",
        "license": "BSD-3-Clause",
        "source": "xinntao/Real-ESRGAN release v0.2.5.0",
        "description": "Compact general restoration network. Optional denoise mix with the wdn weights.",
    },
    "gfpgan_v14": {
        "label": "GFPGAN v1.4",
        "version": "GFPGAN v1.4",
        "filename": "GFPGANv1.4.pth",
        "sha256": "e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad",
        "min_bytes": 300_000_000,
        "arch": "gfpgan",
        "kind": "face",
        "license": "Apache-2.0",
        "source": "TencentARC/GFPGAN release v1.3.0",
        "description": "Optional face restoration applied after super-resolution.",
    },
}

FACE_DETECTOR = {
    "filename": "face_detection_yunet_2023mar.onnx",
    "sha256": None,
    "min_bytes": 100_000,
    "license": "Apache-2.0",
    "source": "opencv/opencv_zoo YuNet 2023mar",
    "description": "Face and landmark detector used to align faces before GFPGAN.",
}


def model_path(model_id: str) -> Path:
    spec = MODEL_CATALOG[model_id]
    return MODELS_DIR / spec["filename"]


def upscaler_ids() -> list[str]:
    return [key for key, spec in MODEL_CATALOG.items() if spec.get("kind") == "upscaler"]
