#!/usr/bin/env sh
# MoE expert streaming demo: run a real Mixture-of-Experts model with its expert
# weights streamed from disk, bounded by a RAM cap far below the full set.
set -e
M="${MODEL:-models/olmoe-1b-7b-q4_k_m.gguf}"
URL="https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf"
mkdir -p models
[ -f "$M" ] || { echo "downloading OLMoE-1B-7B (~4 GB)"; curl -L -o "$M" "$URL"; }
make -s vireo-gguf
echo "== uncapped (all active experts resident) =="
VIREO_STATS=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" ./vireo-gguf "$M" -p "${1:-The capital of France is}" -n "${2:-40}"
echo
echo "== experts capped at 256 MB (streamed from disk) — same output =="
VIREO_EXPERT_CAP_MB=256 VIREO_STATS=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" ./vireo-gguf "$M" -p "${1:-The capital of France is}" -n "${2:-40}"
