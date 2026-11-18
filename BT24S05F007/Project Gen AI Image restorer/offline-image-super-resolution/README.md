# Offline AI Image Enhancer

Local image super-resolution for Apple Silicon. The app loads Real-ESRGAN weights from `models/` and runs them with PyTorch on Metal (MPS) or the CPU. Enhancement does not call a cloud API.

## Run

```bash
cd "/Users/chaitanykakde/Project Gen AI Image restorer/offline-image-super-resolution"
./run.sh
```

Open http://127.0.0.1:8501

Click **Use built-in sample**, then **Start enhancement**. Or drop in a JPG, PNG, or WEBP. Compare the two images, then download PNG or JPEG. Results are also written to `output/`.

## What is included

- Real-ESRGAN general x4v3 is the small model already verified on disk (native 4×). 2× is that 4× result resized with Lanczos, and the screen says so.
- Real-ESRGAN x4plus is the primary model when `models/RealESRGAN_x4plus.pth` is present.
- Device choice is Auto, Apple Metal (MPS), or CPU. The screen shows the device that actually ran.
- Large images are split into tiles so a 16 GB Mac is less likely to run out of memory.
- PSNR and SSIM are not invented for normal uploads. They need a reference image of the same size.

## First-time setup

Python 3.13 is the environment used here (Homebrew). PyTorch’s macOS build includes MPS and does not need CUDA.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
./run.sh
```

`download_models.py` skips files that are already large enough and prints a SHA-256. The general x4v3 checksum pinned in the project is `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292`.

Weights come from the official Real-ESRGAN GitHub releases (BSD-3-Clause): https://github.com/xinntao/Real-ESRGAN

## If something fails

- **No model ready:** the general weight file must be about 4.7 MB. A 0-byte file is a failed download. Run the download script again.
- **MPS error:** choose CPU in the sidebar. The app also retries on CPU if Metal runs out of memory.
- **Image too large:** the longest side must be 2048 pixels or less on this 16 GB Mac.
- **Port in use:** `streamlit run app.py --server.port 8502`
