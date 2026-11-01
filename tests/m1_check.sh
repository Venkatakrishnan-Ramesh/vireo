#!/usr/bin/env sh
# M1 integration check: a real llama2.c model with its KV cache routed through
# vireo's out-of-core store, under a RAM budget that forces disk eviction, must
# generate byte-identical text to the unmodified baseline (greedy decode).
set -e
MODEL="${MODEL:-baseline/stories15M.bin}"
Z="${Z:-baseline/tokenizer.bin}"
URL="https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin"
[ -f "$MODEL" ] || { echo "fetching $MODEL"; curl -sL -o "$MODEL" "$URL"; }
make -s vireo-baseline vireo-run
P="Once upon a time"
B="$(mktemp)"; O="$(mktemp)"
./vireo-baseline "$MODEL" -z "$Z" -t 0 -n 200 -i "$P" 2>/dev/null > "$B"
VIREO_STORE=ooc VIREO_STRATEGY=all VIREO_CAP_BLOCKS=140 \
  ./vireo-run "$MODEL" -z "$Z" -t 0 -n 200 -i "$P" 2>/dev/null > "$O"
if [ -s "$B" ] && diff -q "$B" "$O" >/dev/null; then
  echo "M1 PASS: vireo ooc/all == llama2.c baseline (byte-identical, under RAM budget + eviction)"
  rm -f "$B" "$O"
else
  echo "M1 FAIL: output diverged"; diff "$B" "$O" | head; rm -f "$B" "$O"; exit 1
fi
