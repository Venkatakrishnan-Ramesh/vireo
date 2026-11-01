# Contributing to vireo

Thanks for looking. vireo is small on purpose — you can hold the whole thing in
your head. Here is how to land a change fast.

## Setup

```bash
make test     # oracle test: exact-match + RAM budget. Must pass.
make bench    # headline numbers.
```

A C11 compiler and `make`. No other dependencies, ever — that is a project rule.

## The one rule you cannot break

The out-of-core store with the `all` strategy must stay **bit-identical** to the
dense store (`tests/test_kvstore_oracle.c`). Optimize anything you like beneath
that line. If your change makes the oracle diverge, it is a bug, not a tradeoff.

## Easiest first contribution: a selection strategy

Selection strategies decide which KV blocks a query loads. They are self-
contained plugins in `src/select/`. To add one:

1. Copy `src/select/select_topk.c` to `src/select/select_yours.c`.
2. Implement your `SelectFn` — you get the query and the per-block key summaries.
3. Export a `SelectStrategy` (or a factory, like `vireo_select_topk`).
4. Add it to `CORE` in the `Makefile` and wire a name in `src/main.c`.
5. Run `make bench` with your strategy and put the numbers in the PR.

Ideas live in [docs/good-first-issues.md](docs/good-first-issues.md).

## Style

- C11, no dependencies beyond libc + libm.
- `clang-format` with the bundled `.clang-format` (`clang-format -i src/**/*.c`).
- Small functions, explicit error handling, no clever macros.
- Every performance claim comes with a number from `make bench` or a test.

## PRs

- One idea per PR. Keep diffs reviewable.
- If you touch the pager, say how you know the oracle still passes.
- Be kind. This is a learning-friendly repo.
