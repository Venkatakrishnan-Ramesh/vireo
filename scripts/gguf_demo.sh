#!/usr/bin/env sh
# Download a small real model and run it with its KV cache streamed from disk
# under a RAM budget. Proves vireo on a genuine GGUF model, CPU-only.
set -e
M="models/qwen2.5-0.5b-instruct-q8_0.gguf"
URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"
mkdir -p models
[ -f "$M" ] || { echo "downloading $M (~645 MB)"; curl -L -o "$M" "$URL"; }
make -s vireo-gguf
echo "== KV streamed from disk (window=64, cap=200 blocks forces eviction) =="
VIREO_WINDOW=64 VIREO_CAP_BLOCKS=200 VIREO_STRATEGY=all \
  ./vireo-gguf "$M" -p "${1:-Tell me a short story about a robot who learns to paint.}" -n "${2:-160}"
