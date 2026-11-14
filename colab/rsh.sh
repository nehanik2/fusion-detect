#!/usr/bin/env bash
# Run a shell command on a Colab CLI session (from your laptop or Claude Code).
#   colab/rsh.sh "nvidia-smi"                                   # foreground (TIMEOUT secs, default 600)
#   BG=1 colab/rsh.sh "python scripts/generate_flux.py --n 1000" flux   # background; log -> /content/logs/flux.log
#   colab/rsh.sh "tail -n 20 /content/logs/flux.log"
# Env: SESSION (default fd), TIMEOUT, REPO_DIR (VM, default /content/fusion-detect), DATA_DIR (VM, default Drive path)
set -euo pipefail
SESSION="${SESSION:-fd}"; TIMEOUT="${TIMEOUT:-600}"; NAME="${2:-job}"; BG="${BG:-0}"
REPO_DIR="${REPO_DIR:-/content/fusion-detect}"; DATA="${DATA_DIR:-/content/drive/MyDrive/fusion-detect/data}"
B64=$(printf '%s' "$1" | base64 | tr -d '\n')   # base64 avoids all quoting issues

colab exec -s "$SESSION" --timeout "$TIMEOUT" - <<PY
import base64, os, subprocess, sys
cmd = base64.b64decode("$B64").decode()
env = {**os.environ, "DATA_DIR": "$DATA", "PYTHONUNBUFFERED": "1"}
cwd = "$REPO_DIR" if os.path.isdir("$REPO_DIR") else "/content"
if "$BG" == "1":
    os.makedirs("/content/logs", exist_ok=True)
    log = open("/content/logs/$NAME.log", "a")
    p = subprocess.Popen(["/bin/bash", "-lc", cmd], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    print("started $NAME pid", p.pid, "-> /content/logs/$NAME.log")
else:
    r = subprocess.run(["/bin/bash", "-lc", cmd], cwd=cwd, env=env, text=True, capture_output=True)
    print(r.stdout[-20000:]); print(r.stderr[-5000:], file=sys.stderr); print("exit", r.returncode)
PY
