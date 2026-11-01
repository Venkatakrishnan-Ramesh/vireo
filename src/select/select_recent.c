#include "select.h"

/* Return nothing extra; the store adds sinks + window on its own. */
static int select_recent(const SelectCtx *ctx, int *out_ids, int max_out) {
    (void)ctx; (void)out_ids; (void)max_out;
    return 0;
}
const SelectStrategy VIREO_SELECT_RECENT = { "recent", select_recent, 0 };
