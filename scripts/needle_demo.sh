#!/usr/bin/env sh
# Flagship demo: a real model over a long document, KV cache streamed from disk.
# Shows (1) resident RAM far below the full KV cache, (2) the 'all' strategy
# retrieves a long-range needle byte-identically while 'recent' forgets it.
set -e
M="${MODEL:-models/qwen2.5-0.5b-instruct-q8_0.gguf}"
[ -f "$M" ] || { echo "run 'make gguf-demo' once to fetch the model"; exit 1; }
make -s vireo-gguf
D="$(mktemp)"; python3 scripts/make_needle.py "${1:-600}" > "$D"
N=$(./vireo-gguf "$M" -f "$D" -n 1 2>&1 | grep -oE 'prompt=[0-9]+' | grep -oE '[0-9]+')
STEPS=$((N + 16))
echo "context = $N tokens"
echo
echo "== all  (exact, KV streamed from disk under a cap) =="
VIREO_WINDOW=256 VIREO_CAP_BLOCKS=4096 VIREO_STRATEGY=all \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" ./vireo-gguf "$M" -f "$D" -n "$STEPS"
echo
echo "== recent (window only — expected to forget the early needle) =="
VIREO_WINDOW=256 VIREO_STRATEGY=recent \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" ./vireo-gguf "$M" -f "$D" -n "$STEPS"
rm -f "$D"
