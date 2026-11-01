<h1 align="center">vireo</h1>
<p align="center"><b>Million-token context on the RAM you already have.</b></p>
<p align="center">A transformer's KV cache, paged to disk. Pure C, zero dependencies, CPU-first.</p>

<p align="center">
  <img src="docs/demo.gif" width="720" alt="vireo: KV cache RAM stays flat as context grows to 1M tokens">
</p>

---

A transformer that remembers a million tokens normally needs a huge KV cache in
RAM. But attention for any one token lands on only a few places: the first few
tokens (sinks), the recent window, and a handful of older tokens that actually
matter. **vireo keeps that hot set resident and streams the rest from your SSD**,
paging blocks in only when a query reaches for them. The context grows; the RAM
does not.

```
$ vireo bench 200000 topk
vireo bench  strategy=topk  context=200000 tokens
  resident RAM :    3.9 MB   (dense would need  195.3 MB  ->  50x less)
  cache hits   : 68.9%
  throughput   : 216874 tokens/sec
```

## Why it is correct, not just small

vireo ships a dense in-RAM reference and an out-of-core store behind the same
interface. With the exact (`all`) selection strategy, the out-of-core store is
**bit-for-bit identical** to dense — proven on every CI run by
`tests/test_kvstore_oracle.c`. The lossy strategies that make it fast are
measured *against* that oracle, so "50× less RAM" comes with a provable
baseline, not a vibe.

## Quickstart

```bash
git clone https://github.com/USER/vireo
cd vireo
make test      # builds + runs the oracle test (exact-match + RAM budget)
make bench     # streams a 200k-token context, prints the headline numbers
```

Needs a C compiler and `make`. Nothing else.

### Run a real model

vireo routes the KV cache of Andrej Karpathy's `llama2.c` through its
out-of-core store, so you can watch a real transformer generate with its KV
streamed from disk:

```bash
make m1-test     # downloads a tiny model, proves ooc == baseline byte-for-byte
                 # under a RAM budget that forces eviction

# or drive it yourself (env vars pick the store + strategy):
VIREO_STORE=ooc VIREO_STRATEGY=all VIREO_CAP_BLOCKS=140 VIREO_STATS=1 \
  ./vireo-run baseline/stories15M.bin -z baseline/tokenizer.bin -t 0 -n 256 -i "Once upon a time"
```

With `VIREO_STRATEGY=all` the output is identical to stock llama2.c while the
KV cache spills to disk; `topk` / `recent` trade exactness for less I/O.

### Run a real model from Hugging Face (GGUF)

vireo also loads real GGUF models (Qwen2 family; Q8_0, Q5_0, Q4_K, Q6_K — so
Q4_K_M files work) with weights mmap'd from
disk and the KV cache streamed through its pager — CPU-only, no GPU:

```bash
make gguf-demo   # downloads Qwen2.5-0.5B-Instruct (Q8_0, ~645 MB) and runs it
                 # with the KV cache streamed from disk under a RAM budget
```

```
$ ./vireo-gguf models/qwen2.5-0.5b-instruct-q8_0.gguf -p "The capital of France is" -n 40
The capital of France is Paris. It is the largest city in Europe ...
[vireo-gguf] 24 layers, 14/2 heads | KV hits=100% resident=1.5MB
```

Under a RAM budget that forces eviction (`VIREO_CAP_BLOCKS`), the `all` strategy
is **byte-identical** to the unbounded run while the KV cache spills to and pages
from disk — verified on the real model.

The Q8_0 matmul uses AVX2 (int8×float dot) and OpenMP across output rows:
**~37 tok/s** on a 6-core laptop CPU for the 0.5B model (up from ~5 tok/s
scalar). It is memory-bandwidth bound — the LM head streams ~138 MB of weights
per token — so a few threads saturate it; tune with `OMP_NUM_THREADS`.

## Benchmark — resident RAM vs. context length

Measured by `scripts/make_demo.py` (every point is a real `./vireo` run; kv_dim=128,
64-token blocks, 64-block budget, `topk`). Dense = the whole KV cache held in RAM.

| context | dense cache | vireo resident | less RAM | cache hits |
|--------:|------------:|---------------:|---------:|-----------:|
| 10k     | 9.8 MB      | 3.9 MB         | 2.5×     | 87%        |
| 100k    | 97.7 MB     | 3.9 MB         | 25×      | 70%        |
| 1,000k  | 976.6 MB    | **3.9 MB**     | **248×** | 68%        |

Resident RAM is flat from ~5k tokens to a million; dense grows linearly. With the
exact `all` strategy on a real model (`make m1-test`), the streamed output is
**byte-identical** to stock llama2.c. Regenerate the chart with `make demo`.

## Run a Mixture-of-Experts model bigger than RAM

An MoE model has many experts but fires only a few per token. vireo streams the
**expert weights** from disk through the same lease-LRU pager (`ExpertStore`),
keeping only the active experts resident — so a model far larger than RAM runs
with a small fraction of it in memory.

```bash
make moe-demo    # OLMoE-1B-7B (64 experts, top-8): runs with experts streamed
```

```
$ VIREO_EXPERT_CAP_MB=256 ./vireo-gguf models/olmoe-1b-7b-q4_k_m.gguf -p "The capital of France is"
The capital of France is Paris. The currency of France is Euro. ...
[vireo-gguf] experts: 70/1024 resident (256 MB) ...   # vs 2.7 GB full — same output
```

`VIREO_EXPERT_CAP_MB` bounds resident expert RAM; the rest stream from disk.
This is how a 30B-class MoE fits on a laptop.

## How it works

See **[ARCHITECTURE.md](ARCHITECTURE.md)**. The whole project is one idea — an
out-of-core, lease-paged KV store with a pluggable block-selection strategy.
Adding a selection strategy is the easiest way to contribute: see
**[CONTRIBUTING.md](CONTRIBUTING.md)** and the open
**[good first issues](docs/good-first-issues.md)**.

## Status

Early. The pager and the correctness harness work today. Real-model inference,
prefetch, and SIMD are on the **[ROADMAP](ROADMAP.md)**. Stars and PRs welcome.

## License

MIT. Bundles a vendored copy of Andrej Karpathy's `llama2.c` (MIT) under
`baseline/` as the inference skeleton.
