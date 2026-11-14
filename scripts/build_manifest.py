"""Build data/manifest.csv from configs/data.yaml.

Columns: id, path, label (0=real,1=ai), source, generator, split, width, height, format, sha256
Splits: held-out generators -> test only; others -> train/cal/test by ratio (stratified by generator).

Usage: python scripts/build_manifest.py --config configs/data.yaml
"""
import argparse, csv, hashlib, random
from pathlib import Path

import yaml
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".JPEG", ".PNG"}


def list_images(d: Path):
    return sorted(p for p in d.rglob("*") if p.suffix in EXTS) if d.exists() else []


def sha256(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(cfg, root: Path):
    rows = []  # (path, label, source, generator)
    for name, c in cfg["real"].items():
        for p in list_images(root / c["dir"])[: c["max_images"]]:
            rows.append((p, 0, name, "real"))
    for name, c in cfg["ai"].items():
        base = root / c["dir"]
        if name == "genimage":
            for sub in c["subsets"]:
                # GenImage layout: <subset>/.../{train,val}/ai/*  (nature/ = ImageNet reals, skipped)
                imgs = [p for p in list_images(base / sub) if "/ai/" in p.as_posix()]
                for p in imgs[: c["max_images_per_subset"]]:
                    rows.append((p, 1, "genimage", sub))
        else:
            for p in list_images(base)[: c["max_images"]]:
                rows.append((p, 1, name, name))
    return rows


def assign_splits(rows, cfg, rng):
    held = set(cfg["heldout_generators"])
    r = cfg["splits"]
    by_gen = {}
    for row in rows:
        by_gen.setdefault(row[3], []).append(row)
    out = []
    for gen, items in sorted(by_gen.items()):
        rng.shuffle(items)
        if gen in held:
            out += [(*it, "test") for it in items]
            continue
        n = len(items); a = int(n * r["train"]); b = a + int(n * r["cal"])
        out += [(*it, "train") for it in items[:a]]
        out += [(*it, "cal") for it in items[a:b]]
        out += [(*it, "test") for it in items[b:]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-hash", action="store_true", help="skip sha256 (faster)")
    ap.add_argument("--data-root", default=None, help="override config data_root (default: $DATA_DIR or config)")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    import os
    root = Path(a.data_root or os.environ.get("DATA_DIR") or cfg["data_root"])
    rng = random.Random(cfg["seed"])
    rows = assign_splits(collect(cfg, root), cfg, rng)

    a.out = a.out or str(root / "manifest.csv")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    seen, dupes = set(), 0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "path", "label", "source", "generator", "split", "width", "height", "format", "sha256"])
        for i, (p, label, src, gen, split) in enumerate(rows):
            try:
                with Image.open(p) as im:
                    wdt, hgt, fmt = im.width, im.height, im.format
            except Exception as e:
                print("skip unreadable", p, e); continue
            h = "" if a.no_hash else sha256(p)
            if h and h in seen:
                dupes += 1; continue
            seen.add(h)
            w.writerow([f"img{i:07d}", p.as_posix(), label, src, gen, split, wdt, hgt, fmt, h])

    # summary
    from collections import Counter
    c = Counter((r[3], r[4]) for r in rows)
    print(f"[manifest] {len(rows) - dupes} rows -> {a.out} (dupes dropped: {dupes})")
    for (gen, split), n in sorted(c.items()):
        print(f"  {gen:28s} {split:5s} {n}")


if __name__ == "__main__":
    main()
