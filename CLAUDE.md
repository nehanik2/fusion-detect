# fusion-detect — Calibrated fusion of provenance, watermark, and passive AI-image detection

## Goal
Personal research project → workshop paper + arXiv. Solo author (Neha).

## Research questions
- RQ1: Does 3-layer fusion (C2PA + watermark + passive classifiers) beat best single detector and classifier-only ensembles?
- RQ2: How does each layer degrade under laundering (JPEG, resize, screenshot, re-encode, metadata strip, crop, regeneration)?
- RQ3: Does stacking with abstention handle missing signals better than imputation?
- RQ4: Does per-detector calibration + stacking improve TPR@1%FPR and TPR@0.1%FPR?

## Hard rules
- NEVER train, tune, or calibrate on the `test` split. Calibration + fusion weights use `cal` split only.
- Held-out generators (configs/data.yaml: `heldout_generators`) never appear in train/cal.
- Fix seeds everywhere (`SEED=1337`). Log every run (config + git hash) to `runs/`.
- Every detector returns `DetectorResult` (see src/fusion_detect/types.py). Abstain = `score=None`.
- Metrics code (TPR@FPR, AUC, ECE) must have unit tests before use.
- Cache detector scores to `scores/*.parquet`; never recompute silently.
- Do not write paper prose. Code, tables, figures only.

## Layout
- scripts/      data download, generation, manifest
- configs/      yaml configs
- src/fusion_detect/  detectors/, launder/, fusion/, eval/
- data/         raw + processed (gitignored)
- scores/       cached detector outputs (gitignored)
- external/     cloned third-party detector repos (gitignored)

## Environments
- Main: Python 3.10+, `env/main.yml`
- Tree-Ring needs isolated env (diffusers==0.11.1). Run via subprocess, exchange files only.

## Compute: Colab T4 via Google Colab CLI
- Session name: `fd`. Start: `colab new -s fd --gpu T4`. Stop when idle: `colab stop -s fd` (idle VMs burn compute units).
- VM disk is wiped on stop. Persist data/scores to Drive: DATA_DIR=/content/drive/MyDrive/fusion-detect/data
- Run remote commands with `colab/rsh.sh "<cmd>"`; long jobs with `BG=1 colab/rsh.sh "<cmd>" <name>`, poll `/content/logs/<name>.log`.
- Never use `colab repl`/`colab console` from Claude Code (needs a TTY).
- T4 = 16GB VRAM, compute cap 7.5: use fp16 (no bf16), NF4 for FLUX, batch small, CPU offload on.
- Code changes: edit locally -> git push -> `colab/rsh.sh "git pull"`. Never edit code on the VM.
