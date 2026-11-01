# Roadmap

vireo grows along one line: a correct out-of-core KV cache, then a fast one,
then a real model on top, then a product. Each milestone keeps the exact-match
oracle green.

## M0 — baseline (done)
- Vendored `llama2.c` as a dense, correct reference model.

## M1 — real model through the KVStore seam (done)
- llama2.c's per-head KV is routed through `KVStore` (`src/run_vireo.c`), one
  store-layer per `(layer, kv-head)`, GQA-aware, post-RoPE.
- Proven: `make m1-test` — `ooc`/`all` under a RAM budget that forces eviction
  generates **byte-identical** text to the unmodified baseline.
- Open next: true multi-head batching, and the real-model quality harness for
  the lossy strategies (feeds M3).

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
