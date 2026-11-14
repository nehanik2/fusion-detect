"""Generate FLUX.1-schnell images from COCO val2017 captions (content-matched to real COCO images).

T4-friendly: fp16 (T4 has no bf16), NF4 4-bit transformer + T5, CPU offload. Resumable.

Usage (Colab T4):
  python scripts/generate_flux.py --n 1000 --size 640 --quant nf4 --out $DATA_DIR/raw/flux_schnell
Requires: diffusers>=0.31, transformers, accelerate, sentencepiece, bitsandbytes
"""
import argparse, json, os, random, time
from pathlib import Path

import torch
from diffusers import FluxPipeline

MODEL = "black-forest-labs/FLUX.1-schnell"


def load_captions(path: Path, n: int, seed: int):
    ann = json.loads(path.read_text())
    by_img = {}
    for a in ann["annotations"]:
        by_img.setdefault(a["image_id"], a["caption"].strip())   # first caption per image
    items = sorted(by_img.items())
    random.Random(seed).shuffle(items)
    return items[:n]


def build_pipe(quant: str, offload: bool):
    dtype = torch.float16 if torch.cuda.get_device_capability()[0] < 8 else torch.bfloat16  # T4 = 7.5 -> fp16
    kw = {"torch_dtype": dtype}
    if quant == "nf4":
        from diffusers import BitsAndBytesConfig as DBnb, FluxTransformer2DModel
        from transformers import BitsAndBytesConfig as TBnb, T5EncoderModel
        kw["transformer"] = FluxTransformer2DModel.from_pretrained(
            MODEL, subfolder="transformer", torch_dtype=dtype,
            quantization_config=DBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype))
        kw["text_encoder_2"] = T5EncoderModel.from_pretrained(
            MODEL, subfolder="text_encoder_2", torch_dtype=dtype,
            quantization_config=TBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype))
    pipe = FluxPipeline.from_pretrained(MODEL, **kw)
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.vae.enable_tiling()
    return pipe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--out", default=os.path.join(os.environ.get("DATA_DIR", "data"), "raw/flux_schnell"))
    p.add_argument("--captions", default=os.path.join(os.environ.get("DATA_DIR", "data"),
                                                      "raw/coco/annotations/captions_val2017.json"))
    p.add_argument("--size", type=int, default=640, help="multiple of 16; 640 ~ COCO scale, fast on T4")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--quant", choices=["none", "nf4"], default="nf4")
    p.add_argument("--no-offload", action="store_true")
    a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = out / "meta.jsonl"
    done = {json.loads(l)["file"] for l in meta.read_text().splitlines() if l.strip()} if meta.exists() else set()
    todo = [(i, iid, cap) for i, (iid, cap) in enumerate(load_captions(Path(a.captions), a.n, a.seed))
            if f"flux_{iid:012d}.png" not in done]
    print(f"[flux] {len(done)} done, {len(todo)} to go")
    if not todo:
        return

    pipe = build_pipe(a.quant, offload=not a.no_offload)
    t0 = time.time()
    with meta.open("a") as f:
        for k, (i, img_id, cap) in enumerate(todo):
            name = f"flux_{img_id:012d}.png"
            g = torch.Generator("cpu").manual_seed(a.seed + i)
            img = pipe(cap, height=a.size, width=a.size, num_inference_steps=4,
                       guidance_scale=0.0, max_sequence_length=256, generator=g).images[0]
            img.save(out / name)   # PNG, no metadata -> clean baseline (C2PA added later in pipeline)
            f.write(json.dumps({"file": name, "coco_id": img_id, "prompt": cap, "seed": a.seed + i,
                                "size": a.size, "quant": a.quant}) + "\n")
            f.flush()
            if k % 25 == 0:
                rate = (time.time() - t0) / (k + 1)
                print(f"[flux] {k + 1}/{len(todo)}  {rate:.1f}s/img  eta {rate * (len(todo) - k - 1) / 60:.0f} min",
                      flush=True)


if __name__ == "__main__":
    main()
