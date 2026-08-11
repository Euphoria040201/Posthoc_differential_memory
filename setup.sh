#!/bin/bash
# Set up the delta-mem investigation bundle on a fresh box (e.g. RTX 4080 / Ada sm_89).
set -e
cd "$(dirname "$0")"
PY=${PY:-python3.10}
command -v $PY >/dev/null || PY=python3
echo "[setup] python = $($PY --version 2>&1)"

if [ ! -d .venv ]; then
  echo "[setup] creating .venv"
  $PY -m venv .venv
fi
. .venv/bin/activate
pip install --upgrade pip
# torch 2.6.0+cu124 is in requirements.txt and works on Ada (sm_89). If the index can't find the
# +cu124 wheel, install torch first from the cu124 index, then the rest:
#   pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

echo "[setup] done. To run:"
echo "  source .venv/bin/activate && export PYTHONPATH=.:scripts"
echo "  python investigation/l33_allseeds.py"
echo
echo "[note] Qwen/Qwen3-4B-Instruct-2507 + HotpotQA/Qasper must be in ~/.cache/huggingface"
echo "       (scripts use local_files_only=True). Download once or copy the cache over."
