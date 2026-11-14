#!/usr/bin/env bash
# Run ON the Colab VM after every `colab new` (VM disk is wiped on stop; Drive persists).
# Usage (inside `colab console`):  bash /content/fusion-detect/colab/setup_vm.sh
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/nehanik2/fusion-detect.git}"
cd /content
[ -d fusion-detect ] || git clone -q "$REPO_URL" fusion-detect
cd fusion-detect && git pull -q
pip install -q -r requirements-colab.txt
export DATA_DIR=/content/drive/MyDrive/fusion-detect/data
mkdir -p "$DATA_DIR" 2>/dev/null || echo "!! Drive not mounted: run 'colab drivemount -s fd' first"
grep -q DATA_DIR ~/.bashrc || echo "export DATA_DIR=$DATA_DIR" >> ~/.bashrc
python - <<'PY'
import torch; print("GPU:", torch.cuda.get_device_name(0), "| cap", torch.cuda.get_device_capability(0),
                    "| VRAM GB", round(torch.cuda.get_device_properties(0).total_memory/1e9,1))
PY
df -h /content | tail -1; free -g | head -2
echo "[setup] ready. DATA_DIR=$DATA_DIR"
