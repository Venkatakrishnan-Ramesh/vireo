/* Selection strategies decide WHICH blocks the out-of-core store loads for a
 * query. This is the heart of the research: load every block and you are exact
 * but slow; load only the blocks a query attends to and you are fast and
 * approximate. Each strategy is a plugin. Add yours here and wire it in main.c
 * — that is deliberately the easiest way to contribute (see CONTRIBUTING.md).
 */
#ifndef VIREO_SELECT_H
#define VIREO_SELECT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Read-only view the store hands a strategy for one query. */
typedef struct {
    int layer;
    int kv_dim;
    int n_blocks;          /* total blocks appended so far                 */
    int block_tokens;      /* G                                            */
    int sink_blocks;       /* leading blocks that are always included      */
    int window_blocks;     /* trailing blocks that are always included     */
    const float *q;        /* query vector, length kv_dim                  */
    /* Per-block key summary (mean key), n_blocks * kv_dim, row-major.
     * NULL if the store does not maintain summaries. */
    const float *block_ksum;
} SelectCtx;

/* Fill out_ids with the block ids to load (0-based), return the count.
 * The store always unions sinks + window itself, so a strategy may ignore
 * them; duplicates are de-duped by the store. Never write more than max_out. */
typedef int (*SelectFn)(const SelectCtx *ctx, int *out_ids, int max_out);

typedef struct SelectStrategy {
    const char *name;
    SelectFn select;
    /* budget: max old (non-sink/window) blocks to load. <=0 means unbounded. */
    int budget;
} SelectStrategy;

/* "all": load every block. Exact, matches the dense store bit-for-bit. */
extern const SelectStrategy VIREO_SELECT_ALL;
/* "recent": sinks + window only. Cheapest, lossy for long-range recall. */
extern const SelectStrategy VIREO_SELECT_RECENT;
/* "topk": score blocks by q . ksum and load the top `budget`. Approximate. */
const SelectStrategy *vireo_select_topk(int budget);

#ifdef __cplusplus
}
#endif
#endif /* VIREO_SELECT_H */
