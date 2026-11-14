#!/usr/bin/env bash
# Download datasets, detector repos, and weights.
# Data goes to $DATA_DIR (default ./data). On Colab set DATA_DIR=/content/drive/MyDrive/fusion-detect/data
# Usage: bash scripts/download_data.sh [coco|genimage|raise|repos|weights|all]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
RAW="$DATA_DIR/raw"
EXT="$ROOT/external"
MODELS="${MODELS_DIR:-$ROOT/models}"
mkdir -p "$RAW" "$EXT" "$MODELS"

coco() {
  mkdir -p "$RAW/coco" && cd "$RAW/coco"
  [ -d val2017 ] || { wget -q -c http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip && rm val2017.zip; }
  [ -d annotations ] || { wget -q -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
      && unzip -q annotations_trainval2017.zip && rm annotations_trainval2017.zip; }
  echo "[coco] done -> $RAW/coco"
}

genimage() {
  # HF parquet mirrors: ~0.6GB/shard instead of the ~500GB official release.
  python "$ROOT/scripts/fetch_genimage_hf.py" --subsets ${GENIMAGE_SUBSETS:-ADM glide VQDM} \
    --shards "${GENIMAGE_SHARDS:-1}" --max "${GENIMAGE_MAX:-1500}"
}

raise() {
  # Optional (TIFFs are ~30MB each). Download RAISE_1k.csv from https://loki.disi.unitn.it/RAISE/download.html
  # into $RAW/raise/. Fetches RAISE_MAX (default 300) TIFFs.
  mkdir -p "$RAW/raise/tiff"
  CSV=$(ls "$RAW"/raise/RAISE_*.csv 2>/dev/null | head -1 || true)
  [ -n "$CSV" ] || { echo "[raise] no CSV in $RAW/raise — skipping (optional)"; return 0; }
  python3 - "$CSV" "$RAW/raise/tiff" "${RAISE_MAX:-300}" <<'PY'
import csv, sys, os, urllib.request
csv_path, out, mx = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8-sig")))[:mx]
col = next(c for c in rows[0] if c.strip().upper() == "TIFF")
for i, r in enumerate(rows):
    url = r[col].strip(); dst = os.path.join(out, os.path.basename(url))
    if url and not os.path.exists(dst):
        try: urllib.request.urlretrieve(url, dst)
        except Exception as e: print("fail", url, e)
    if i % 50 == 0: print(f"[raise] {i}/{len(rows)}", flush=True)
PY
}

repos() {
  cd "$EXT"
  clone() { [ -d "$2" ] || git clone -q --depth 1 "$1" "$2"; }
  clone https://github.com/WisconsinAIVision/UniversalFakeDetect.git  ojha
  clone https://github.com/chuangchuangtan/NPR-DeepfakeDetection.git   npr
  clone https://github.com/shilinyan99/AIDE.git                        aide     # README: Chameleon + checkpoint links
  clone https://github.com/YuxinWenRick/tree-ring-watermark.git        treering
  clone https://github.com/facebookresearch/stable_signature.git       stablesig
  echo "[repos] done -> $EXT"
}

weights() {
  cd "$MODELS"
  [ -f dec_48b_whit.torchscript.pt ] || wget -q -c https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt
  echo "[weights] Stable Signature decoder -> $MODELS"
  echo "[weights] Ojha: external/ojha/pretrained_weights/fc_weights.pth (ships with repo)"
  echo "[weights] NPR / AIDE: follow repo README checkpoint links -> $MODELS/<name>/"
}

case "${1:-all}" in
  coco) coco ;; genimage) genimage ;; raise) raise ;; repos) repos ;; weights) weights ;;
  all) coco; genimage; repos; weights; raise ;;
  *) echo "unknown target: $1"; exit 1 ;;
esac
