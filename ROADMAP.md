# Roadmap

vireo grows along one line: a correct out-of-core KV cache, then a fast one,
then a real model on top, then a product. Each milestone keeps the exact-match
oracle green.

## M0 — baseline (done)
- Vendored `llama2.c` as a dense, correct reference model.

## M1 — real model through the KVStore seam (open)
- Route a llama2.c run's per-layer, per-head KV through `KVStore`.
- Multi-head support in the store (currently single logical head).
- New oracle: streamed model output == dense model output on a real prompt.

## M2 — out-of-core, exact (done, synthetic)
- Block spill/page, LRU, leases, pinned sinks/window. Bit-exact vs dense.
- Remaining: `mmap` + `madvise` backend to compare against `pread`.

## M3 — sparse selection (partial)
- `topk` by mean-key exists. Needed: better scorers (max-over-keys, SimHash,
  learned summaries), and a quality harness (perplexity vs. blocks loaded).

## M4 — scale and product (open)
- Async prefetch of next-token blocks.
- AVX2/NEON dot products and dequant.
- Block compression on spill (Zstd).
- OpenAI-compatible serving endpoint.
- The demo: 1,000,000-token context on 8 GB RAM, with the GIF.

## Non-goals
- A dependency tree. libc + libm only.
- Beating a GPU. The point is running long context where a GPU cannot fit it.
