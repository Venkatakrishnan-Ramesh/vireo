#!/usr/bin/env sh
# One-command entry: build vireo, run the oracle test, print the headline bench.
set -e
echo "==> building + testing the out-of-core KV pager"
make test
echo
echo "==> headline bench (200k-token context)"
make bench
echo
echo "vireo is working. See ARCHITECTURE.md for how, ROADMAP.md for what's next."
