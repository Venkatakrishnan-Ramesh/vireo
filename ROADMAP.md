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

## GGUF loader — real models (done: Q8_0, Q5_0, Q4_K, Q6_K)
- `src/gguf.c` (mmap reader), `src/gguf_compute.c` (Q8_0/F16/F32 dequant +
  quantized matmul), `src/gguf_run.c` (Qwen2 forward: NEOX RoPE, QKV bias, GQA,
  SwiGLU, byte-level BPE tokenizer), KV streamed through the pager.
- `make gguf-demo` runs Qwen2.5-0.5B-Instruct Q8_0 CPU-only; under a RAM budget
  that forces eviction, `all` is byte-identical to the unbounded run.
- Q8_0 matmul is AVX2 + OpenMP (~37 tok/s for 0.5B, memory-bandwidth bound).
- Q4_K_M files (Q5_0/Q4_K/Q6_K mix) run. Open next: fused Q4_K/Q5_0 dot (the
  generic dequant-then-dot path is ~2x slower than the fused Q8_0 path), more
  architectures (Llama/Mistral tensor names), NEON for ARM.

## MoE expert streaming (done)
- `src/expert_store.{h,c}`: the KVStore twin for expert weights — lease-LRU
  pager, `pread` from the GGUF, bounded by `VIREO_EXPERT_CAP_MB`. Unit-tested
  (`make expert-test`).
- MoE forward (router + top-k) in gguf_run.c, generic over architecture.
- Verified: OLMoE-1B-7B (64 experts/top-8) coherent; experts capped at 256 MB
  (vs 2.7 GB) stream from disk with identical output (`make moe-demo`).
- **Qwen3-30B-A3B runs** (qwen3moe, 128 experts/top-8, 48 layers): coherent
  on a 14 GB laptop with experts capped at 4 GB, rest streamed from disk.
- Open next: shared-expert
  models (Qwen2-MoE/DeepSeek), prefetch of likely-next experts.

## M4 — scale and product (open)
- Async prefetch of next-token blocks.
- AVX2 Q8_0 matmul done; NEON (ARM) and AVX2 for the KV attention dot still open.
- Block compression on spill (Zstd).
- OpenAI-compatible serving endpoint.
- The demo: 1,000,000-token context on 8 GB RAM, with the GIF.

## Non-goals
- A dependency tree. libc + libm only.
- Beating a GPU. The point is running long context where a GPU cannot fit it.
