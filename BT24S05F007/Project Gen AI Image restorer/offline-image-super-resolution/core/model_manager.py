"""Load official Real-ESRGAN weights from local disk."""

from __future__ import annotations

from pathlib import Path

import torch

from config.settings import MODEL_CATALOG, MODELS_DIR
from core.architectures import RRDBNet, SRVGGNetCompact


class ModelError(RuntimeError):
    """A local weight file is missing or does not match the network."""


def is_available(model_id: str) -> bool:
    spec = MODEL_CATALOG[model_id]
    path = MODELS_DIR / spec["filename"]
    return path.is_file() and path.stat().st_size >= spec["min_bytes"]


def available_upscalers() -> list[str]:
    return [
        key
        for key, spec in MODEL_CATALOG.items()
        if spec.get("kind") == "upscaler" and is_available(key)
    ]


def _state_dict(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise ModelError(f"{path.name} is not a PyTorch state dict.")
    cleaned = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        cleaned[key[7:] if key.startswith("module.") else key] = value
    if not cleaned:
        raise ModelError(f"{path.name} does not contain model tensors.")
    return cleaned


def build_model(model_id: str) -> torch.nn.Module:
    spec = MODEL_CATALOG[model_id]
    if spec["arch"] == "rrdb":
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=spec["native_scale"],
            num_feat=spec["num_feat"],
            num_block=spec["num_block"],
            num_grow_ch=spec["num_grow_ch"],
        )
    if spec["arch"] == "srvgg":
        return SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=spec["num_feat"],
            num_conv=spec["num_conv"],
            upscale=spec["native_scale"],
            act_type="prelu",
        )
    raise ModelError(f"{spec['label']} is not an upscaler.")


class ModelManager:
    def __init__(self) -> None:
        self._cache: dict[tuple, torch.nn.Module] = {}

    def get(self, model_id: str, device: torch.device) -> torch.nn.Module:
        if not is_available(model_id):
            spec = MODEL_CATALOG[model_id]
            raise ModelError(
                f"{spec['label']} weights are not on disk. "
                f"Expected {spec['filename']} in the models folder. "
                "Run: .venv/bin/python scripts/download_models.py"
            )
        key = (model_id, device.type)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        model = build_model(model_id)
        state = _state_dict(MODELS_DIR / MODEL_CATALOG[model_id]["filename"])
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ModelError(
                f"Weight mismatch for {MODEL_CATALOG[model_id]['label']}. "
                f"Missing {len(missing)} keys, unexpected {len(unexpected)} keys."
            )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.to(device)
        self._cache[key] = model
        return model

    def unload(self) -> None:
        self._cache.clear()


_MANAGER = ModelManager()


def get_model_manager() -> ModelManager:
    return _MANAGER
