"""Download official weights once. Safe to run again; existing valid files are skipped."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import MODELS_DIR, MODEL_CATALOG  # noqa: E402

DOWNLOADS = {
    "realesrgan_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "realesrgan_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "realesrgan_x4plus_anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "realesrgan_general_x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "realesrgan_general_wdn": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
}

WDN_NAME = "realesr-general-wdn-x4v3.pth"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "offline-image-enhancer"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    temporary.replace(dest)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for model_id, url in DOWNLOADS.items():
        if model_id == "realesrgan_general_wdn":
            filename = WDN_NAME
            expected = None
            minimum = 3_000_000
        else:
            spec = MODEL_CATALOG[model_id]
            filename = spec["filename"]
            expected = spec.get("sha256")
            minimum = spec["min_bytes"]
        dest = MODELS_DIR / filename
        if dest.is_file() and dest.stat().st_size >= minimum:
            digest = _sha256(dest)
            if expected and digest != expected:
                print(f"checksum mismatch, redownloading {filename}")
            else:
                print(f"ok {filename}  sha256={digest}")
                continue
        print(f"downloading {filename}")
        try:
            _download(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"failed {filename}: {exc}")
            continue
        digest = _sha256(dest)
        print(f"saved {filename}  bytes={dest.stat().st_size}  sha256={digest}")
        if expected and digest != expected:
            print(f"warning: {filename} checksum does not match the pinned value")
        if dest.stat().st_size < minimum:
            print(f"warning: {filename} looks incomplete")


if __name__ == "__main__":
    main()
