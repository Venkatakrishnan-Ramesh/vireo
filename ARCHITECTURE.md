# Architecture

vireo is one abstraction with two implementations and a set of plugins. Read
this page and you can contribute in five minutes.

## The seam: `KVStore` (`src/kvstore.h`)

Everything a transformer does with its KV cache reduces to three calls:

```c
kv_append(store, layer, pos, k, v);   // write one token's key/value
kv_attend(store, layer, q, out);      // out = softmax(q·K/√d)·V over chosen tokens
kv_stats (store, &stats);             // hits, misses, resident bytes, ...
```

Both implementations accumulate attention through the same numerically-stable
online-softmax helpers in `kvstore.h`. Visiting the same tokens in the same
global order therefore gives identical floating-point output — that is what
makes the exact-match oracle possible.

## Two implementations

- **`kvstore_dense.c`** — all KV in RAM, contiguous per layer. Simple, obviously
  correct. The reference every optimization is checked against.
- **`kvstore_ooc.c`** — the out-of-core pager (the heart):
  - KV grouped into **blocks** of `G` tokens.
  - A bounded resident working set; cold blocks **spill to disk** (`pwrite`) and
    page back on demand (`pread`).
  - **Pinned** resident forever: attention sinks (prefix), the recent window,
    and the block currently being written.
  - **LRU eviction** of the globally coldest unpinned, unleased block.
  - **Leases**: a block loaded for an `attend()` is refcount-pinned so eviction
    cannot free it mid-read. This refcount is the one invariant you must not
    break.
  - A **per-block key summary** (running mean key) kept resident so selection
    strategies can score blocks cheaply.

## The plugins: selection strategies (`src/select/`)

A strategy decides *which* blocks a query loads. This is where speed vs. quality
is traded, and where most research lives.

- `select_all` — load every block. Exact, matches dense.
- `select_recent` — sinks + window only. Cheapest, lossy at long range.
- `select_topk` — score blocks by `q · mean_key`, load the top few. Approximate.

Add a strategy by implementing one `SelectFn` and registering it. That is the
intended contribution path.

## Correctness invariant

`tests/test_kvstore_oracle.c`: out-of-core + `all` == dense, bit for bit, while
`resident_bytes <= capacity_bytes`. CI fails if that ever breaks. Optimize
freely underneath it; do not cross it.

## Milestones

| | milestone | state |
|-|-----------|-------|
| M0 | llama2.c dense baseline (oracle for a real model) | vendored |
| M1 | route a real model's KV through `KVStore` | **done** (`make m1-test`) |
| M2 | out-of-core, exact-match preserved | **done (synthetic)** |
| M3 | sparse selection, quality measured vs recall | partial (`topk`) |
| M4 | prefetch, SIMD, serving, 1M-token demo | open |
