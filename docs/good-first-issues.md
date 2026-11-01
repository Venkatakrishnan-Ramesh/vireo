# Good first issues

Each is scoped to an afternoon and self-contained. Open a matching GitHub issue
with the `good first issue` label, then send a PR. Put `make bench` numbers in
the PR where relevant.

## Selection strategies (easiest — pure plugins)
- [ ] `select_maxkey`: score a block by the max `q·k` over its keys, not the mean. Usually better recall than `topk`.
- [ ] `select_simhash`: 1-bit SimHash of keys per block; score by Hamming distance to the query hash. Very cheap.
- [ ] `select_h2o`: keep blocks with the highest accumulated attention mass (Heavy-Hitter Oracle).
- [ ] `select_random`: random B blocks. A deliberate lower bound for the quality harness.

## Pager
- [ ] `mmap` + `madvise(WILLNEED/DONTNEED)` backend; benchmark vs the current `pread` path.
- [ ] Zstd-compress blocks on spill, decompress on page-in; report the RAM/throughput tradeoff.
- [ ] Async prefetch: load the next token's predicted blocks while the current token finishes.

## Performance
- [ ] AVX2 `kv_dot`; gate behind a runtime CPU check.
- [ ] ARM NEON `kv_dot` for Apple Silicon / Raspberry Pi.
- [ ] int8 KV storage with per-block scale; measure quality vs. the fp32 oracle.

## Harness & docs
- [ ] Quality harness: perplexity (or output divergence) vs. blocks-loaded, as a plot.
- [ ] Record the README demo GIF.
- [ ] Port the bench to emit CSV for easy plotting.

## M1 (bigger, mentored)
- [ ] Route llama2.c's per-head KV through `KVStore` and add the real-model oracle.
