"""Select Apple Metal (MPS) or the CPU, and say which one is actually used."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Must be set before the first MPS operation. Unsupported Metal operators then
# run on the CPU instead of aborting. The UI reports this explicitly.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from config.settings import ROOT  # noqa: F401  (keeps project import path honest)


@dataclass(frozen=True)
class DeviceChoice:
    device: torch.device
    name: str
    label: str
    mps_built: bool
    mps_available: bool
    fallback_reason: str | None
    unified_memory_bytes: int | None

    @property
    def uses_mps_fallback_ops(self) -> bool:
        return self.name == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def unified_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def mps_is_usable() -> tuple[bool, str | None]:
    """Return whether a tensor can actually be computed on MPS."""
    if not torch.backends.mps.is_built():
        return False, "This PyTorch build does not include MPS."
    if not torch.backends.mps.is_available():
        return False, "MPS is built but not available on this Mac."
    try:
        sample = torch.ones(4, 4, device="mps")
        result = (sample @ sample).cpu()
        if result.shape != (4, 4):
            return False, "MPS probe returned an unexpected tensor."
        torch.mps.synchronize()
        return True, None
    except Exception as exc:  # noqa: BLE001 - probe must catch backend failures
        return False, f"MPS probe failed: {exc}"


def select_device(preference: str = "auto") -> DeviceChoice:
    """Resolve a preference to a real device.

    ``preference`` is ``auto``, ``mps``, or ``cpu``. CUDA is never selected.
    """
    preference = (preference or "auto").lower()
    if preference not in {"auto", "mps", "cpu"}:
        raise ValueError("Device preference must be auto, mps, or cpu.")

    memory = unified_memory_bytes()
    built = bool(torch.backends.mps.is_built())
    usable, reason = mps_is_usable()

    if preference == "cpu":
        return DeviceChoice(
            device=torch.device("cpu"),
            name="cpu",
            label="CPU",
            mps_built=built,
            mps_available=usable,
            fallback_reason=None,
            unified_memory_bytes=memory,
        )

    if usable:
        return DeviceChoice(
            device=torch.device("mps"),
            name="mps",
            label="Apple Metal (MPS)",
            mps_built=built,
            mps_available=True,
            fallback_reason=None,
            unified_memory_bytes=memory,
        )

    fallback = reason or "MPS is not available."
    if preference == "mps":
        fallback = f"MPS was requested. {fallback} Using the CPU."
    else:
        fallback = f"Automatic selection used the CPU. {fallback}"
    return DeviceChoice(
        device=torch.device("cpu"),
        name="cpu",
        label="CPU",
        mps_built=built,
        mps_available=False,
        fallback_reason=fallback,
        unified_memory_bytes=memory,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def release_cached_memory(device: torch.device | None = None) -> None:
    if device is None or device.type == "mps":
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def memory_snapshot(device: torch.device) -> dict[str, int | None]:
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes. Linux reports kilobytes.
    if os.uname().sysname != "Darwin":
        rss *= 1024
    allocated = None
    driver = None
    if device.type == "mps":
        allocated = int(torch.mps.current_allocated_memory())
        driver = int(torch.mps.driver_allocated_memory())
    return {
        "peak_rss_bytes": int(rss),
        "mps_allocated_bytes": allocated,
        "mps_driver_bytes": driver,
    }


def max_input_edge(memory_bytes: int | None = None) -> int:
    """Cap the input edge so a 4× float output tensor fits in unified memory."""
    memory = memory_bytes if memory_bytes is not None else unified_memory_bytes()
    gib = (memory or 8 * 1024**3) / float(1024**3)
    if gib <= 8:
        return 1280
    if gib <= 18:
        return 2048
    return 3072
