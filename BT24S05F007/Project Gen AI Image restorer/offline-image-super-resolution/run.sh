#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Create the environment first:"
  echo "  python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python - <<'PY'
import importlib, sys
missing = []
for name in ("torch", "PIL", "streamlit", "numpy"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    print("Missing dependencies:")
    print("\n".join(missing))
    sys.exit(1)
import torch
print(f"PyTorch {torch.__version__}  MPS={torch.backends.mps.is_available()}")
from core.model_manager import available_upscalers
ready = available_upscalers()
print("Ready models:", ", ".join(ready) if ready else "none")
if not ready:
    print("Run: .venv/bin/python scripts/download_models.py")
    sys.exit(1)
PY

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
