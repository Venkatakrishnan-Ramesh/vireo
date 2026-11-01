#include "select.h"

static int select_all(const SelectCtx *ctx, int *out_ids, int max_out) {
    int n = ctx->n_blocks < max_out ? ctx->n_blocks : max_out;
    for (int i = 0; i < n; i++) out_ids[i] = i;
    return n;
}
const SelectStrategy VIREO_SELECT_ALL = { "all", select_all, 0 };
