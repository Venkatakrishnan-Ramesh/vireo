#include <math.h>
#include <stdlib.h>
#include "select.h"
#include "../kvstore.h"

/* Score each candidate block by dot(q, mean_key) and keep the highest few.
 * Cheap proxy for "does this block contain keys the query attends to".
 * This is intentionally simple — better scorers (max over keys, SimHash,
 * learned summaries) are open issues. */
static int g_budget = 8;

typedef struct { int id; float score; } Scored;
static int by_score_desc(const void *a, const void *b) {
    float d = ((const Scored *)b)->score - ((const Scored *)a)->score;
    return (d > 0) - (d < 0);
}

static int select_topk(const SelectCtx *ctx, int *out_ids, int max_out) {
    if (!ctx->block_ksum || ctx->n_blocks <= 0) return 0;
    int lo = ctx->sink_blocks;
    int hi = ctx->n_blocks - ctx->window_blocks; /* exclusive */
    if (hi <= lo) return 0;
    int ncand = hi - lo;
    Scored *s = (Scored *)malloc((size_t)ncand * sizeof(Scored));
    if (!s) return 0;
    for (int i = 0; i < ncand; i++) {
        int id = lo + i;
        const float *ksum = ctx->block_ksum + (size_t)id * ctx->kv_dim;
        s[i].id = id;
        s[i].score = kv_dot(ctx->q, ksum, ctx->kv_dim);
    }
    qsort(s, (size_t)ncand, sizeof(Scored), by_score_desc);
    int budget = g_budget < ncand ? g_budget : ncand;
    if (budget > max_out) budget = max_out;
    for (int i = 0; i < budget; i++) out_ids[i] = s[i].id;
    free(s);
    return budget;
}

const SelectStrategy *vireo_select_topk(int budget) {
    static SelectStrategy strat = { "topk", select_topk, 8 };
    if (budget > 0) { g_budget = budget; strat.budget = budget; }
    return &strat;
}
