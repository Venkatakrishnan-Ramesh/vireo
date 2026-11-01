CC      ?= cc
CFLAGS  ?= -O3 -march=native -funroll-loops -fopenmp -std=c11 -Wall -Wextra -Isrc
LDLIBS  ?= -lm

CORE := src/kvstore_dense.c src/kvstore_ooc.c \
        src/select/select_all.c src/select/select_recent.c src/select/select_topk.c

.PHONY: all test baseline clean
all: test

test: tests/test_kvstore_oracle.c $(CORE)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o build_test
	./build_test

# llama2.c baseline (vendored, MIT). Downloads a tiny model via run.sh.
baseline: baseline/run.c
	$(CC) -O3 -march=native baseline/run.c -lm -o vireo-baseline

clean:
	rm -f build_test vireo-baseline

vireo: src/main.c $(CORE)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o vireo

.PHONY: bench
bench: vireo
	./vireo bench 200000 topk

# M1: llama2.c with its KV cache routed through vireo's KVStore
vireo-run: src/run_vireo.c $(CORE)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o vireo-run

.PHONY: m1-test
m1-test: vireo-baseline vireo-run
	./tests/m1_check.sh

.PHONY: demo
demo: vireo
	python3 scripts/make_demo.py

gguf-dump: tools/gguf_dump.c src/gguf.c
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o gguf-dump

# vireo-gguf: run a real GGUF model with KV streamed through vireo
GGUF_SRC := src/gguf.c src/gguf_compute.c
vireo-gguf: src/gguf_run.c $(GGUF_SRC) $(CORE)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o vireo-gguf

.PHONY: gguf-demo
gguf-demo: vireo-gguf
	./scripts/gguf_demo.sh

expert-test: tests/test_expert_store.c src/expert_store.c
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o build_expert_test && ./build_expert_test
