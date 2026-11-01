#include <stdio.h>
#include "gguf.h"
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 2; }
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g) return 1;
    gguf_dump(g);
    gguf_close(g);
    return 0;
}
