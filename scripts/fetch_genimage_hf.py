"""Fetch a few GenImage parquet shards from Hugging Face mirrors and unpack to PNG/JPEG files.

Avoids the ~500GB official download. Each shard is ~0.5-0.6GB.
Mirrors (community, bitmind/GenImage_<subset>): verify the repo exists before use.

Usage:
  python scripts/fetch_genimage_hf.py --subsets ADM glide VQDM --shards 1 --max 1500
Output: $DATA_DIR/raw/genimage/<subset>/ai/*.png  (layout matches build_manifest.py)
"""
import argparse, io, os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image


def image_column(table):
    for name in table.column_names:
        col = table.column(name)
        if str(col.type).startswith("struct") and "bytes" in str(col.type):
            return name
    raise ValueError(f"no image column in {table.column_names}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["ADM", "glide", "VQDM"])
    ap.add_argument("--org", default="bitmind")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--max", type=int, default=1500, help="max images per subset")
    ap.add_argument("--out", default=os.path.join(os.environ.get("DATA_DIR", "data"), "raw/genimage"))
    a = ap.parse_args()

    for sub in a.subsets:
        repo = f"{a.org}/GenImage_{sub}"
        try:
            files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
        except Exception as e:
            print(f"[genimage] skip {repo}: {e}"); continue
        out = Path(a.out) / sub / "ai"; out.mkdir(parents=True, exist_ok=True)
        n = len(list(out.glob("*")))
        for f in files[: a.shards]:
            if n >= a.max:
                break
            local = hf_hub_download(repo, f, repo_type="dataset")
            t = pq.read_table(local)
            col = image_column(t)
            if "label" in t.column_names:   # mirrors may encode labels differently: report, don't filter
                from collections import Counter
                print(f"[genimage] {repo}/{f} label counts: {Counter(t.column('label').to_pylist())}")
            for i, cell in enumerate(t.column(col).to_pylist()):
                if n >= a.max:
                    break
                raw = cell["bytes"]
                fmt = (Image.open(io.BytesIO(raw)).format or "PNG").lower()
                (out / f"{sub}_{n:06d}.{ 'jpg' if fmt == 'jpeg' else fmt }").write_bytes(raw)  # original bytes, no re-encode
                n += 1
        print(f"[genimage] {sub}: {n} images -> {out}")


if __name__ == "__main__":
    main()
