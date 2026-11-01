#define _POSIX_C_SOURCE 200809L
/* Out-of-core KV store. Keeps a bounded resident working set; spills cold
 * blocks to disk and pages them back on demand. Sinks, the recent window and
 * the block currently being written are pinned resident. A selection strategy
 * decides which older blocks to load for each query.
 *
 * Invariant (checked by tests/test_kvstore_oracle.c): with VIREO_SELECT_ALL,
 * output is bit-identical to the dense store while resident_bytes stays under
 * capacity_bytes.
 *
 * Leases: a block loaded for an attend() is lease-pinned so eviction cannot
 * free buffers mid-read. attend() releases every block it leased before it
 * returns. The lease refcount is the one invariant you must not break.
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "kvstore.h"
#include "select/select.h"

typedef struct {
    float *k, *v;      /* resident buffers (g*kv_dim each), NULL if spilled */
    int ntok;
    int resident;
    int lease;
    uint64_t last_used;
    long disk_slot;    /* -1 until first spilled */
} Block;

typedef struct {
    Block *blocks;
    int n_blocks, cap_blocks;
    float *ksum;       /* cap_blocks * kv_dim, running mean key per block */
} OocLayer;

typedef struct {
    OocLayer *layers;
    int n_layers, kv_dim, g;
    int sink_blocks, window_blocks;
    const SelectStrategy *strat;
    int fd;
    char path[1024];
    long next_slot;
    uint64_t tick;
    uint64_t block_bytes, half_bytes;
    uint64_t resident_bytes, capacity_bytes;
    KVStats st;
    float *acc;        /* scratch kv_dim */
    int *idbuf;        /* scratch, >= n_blocks */
    unsigned char *seen;
    int scratch_cap;
} OocState;

static int blocks_for(int tokens, int g) { return tokens > 0 ? (tokens + g - 1) / g : 0; }

static int is_pinned(OocState *S, OocLayer *L, int i) {
    if (i < S->sink_blocks) return 1;                      /* attention sinks */
    if (i >= L->n_blocks - S->window_blocks) return 1;     /* recent window   */
    if (i == L->n_blocks - 1) return 1;                    /* current block   */
    return 0;
}

static void spill(OocState *S, Block *b) {
    if (b->disk_slot < 0) b->disk_slot = S->next_slot++;
    off_t off = (off_t)b->disk_slot * (off_t)S->block_bytes;
    pwrite(S->fd, b->k, S->half_bytes, off);
    pwrite(S->fd, b->v, S->half_bytes, off + (off_t)S->half_bytes);
    S->st.bytes_spilled += S->block_bytes;
    free(b->k); free(b->v); b->k = b->v = NULL;
    b->resident = 0;
    S->resident_bytes -= S->block_bytes;
}

static void evict_until_room(OocState *S) {
    if (!S->capacity_bytes) return;
    while (S->resident_bytes + S->block_bytes > S->capacity_bytes) {
        Block *victim = NULL; uint64_t best = (uint64_t)-1;
        for (int li = 0; li < S->n_layers; li++) {
            OocLayer *L = &S->layers[li];
            for (int i = 0; i < L->n_blocks; i++) {
                Block *b = &L->blocks[i];
                if (b->resident && !b->lease && !is_pinned(S, L, i) && b->last_used < best) {
                    best = b->last_used; victim = b;
                }
            }
        }
        if (!victim) break; /* nothing evictable right now; stay safe */
        spill(S, victim);
    }
}

static void ensure_resident(OocState *S, OocLayer *L, int i) {
    Block *b = &L->blocks[i];
    S->st.requests++;
    if (b->resident) { S->st.hits++; b->last_used = ++S->tick; b->lease++; return; }
    S->st.misses++;
    evict_until_room(S);
    b->k = (float *)malloc(S->half_bytes);
    b->v = (float *)malloc(S->half_bytes);
    off_t off = (off_t)b->disk_slot * (off_t)S->block_bytes;
    pread(S->fd, b->k, S->half_bytes, off);
    pread(S->fd, b->v, S->half_bytes, off + (off_t)S->half_bytes);
    b->resident = 1;
    S->resident_bytes += S->block_bytes;
    S->st.bytes_read += S->block_bytes;
    b->last_used = ++S->tick;
    b->lease++;
}
static void release(Block *b) { if (b->lease > 0) b->lease--; }

static void start_block(KVStore *s, OocLayer *L) {
    OocState *S = (OocState *)s->state;
    if (L->n_blocks >= L->cap_blocks) {
        int cap = L->cap_blocks ? L->cap_blocks * 2 : 64;
        L->blocks = (Block *)realloc(L->blocks, (size_t)cap * sizeof(Block));
        L->ksum = (float *)realloc(L->ksum, (size_t)cap * S->kv_dim * sizeof(float));
        L->cap_blocks = cap;
    }
    Block *b = &L->blocks[L->n_blocks];
    memset(b, 0, sizeof(*b));
    b->disk_slot = -1;
    b->k = (float *)malloc(S->half_bytes);
    b->v = (float *)malloc(S->half_bytes);
    b->resident = 1;
    b->last_used = ++S->tick;
    memset(L->ksum + (size_t)L->n_blocks * S->kv_dim, 0, (size_t)S->kv_dim * sizeof(float));
    L->n_blocks++;
    S->resident_bytes += S->block_bytes;
    evict_until_room(S);
}

static void o_append(KVStore *s, int layer, int pos, const float *k, const float *v) {
    (void)pos;
    OocState *S = (OocState *)s->state;
    OocLayer *L = &S->layers[layer];
    int d = S->kv_dim;
    if (L->n_blocks == 0 || L->blocks[L->n_blocks - 1].ntok == S->g) start_block(s, L);
    int bi = L->n_blocks - 1;
    Block *b = &L->blocks[bi];
    memcpy(b->k + (size_t)b->ntok * d, k, (size_t)d * sizeof(float));
    memcpy(b->v + (size_t)b->ntok * d, v, (size_t)d * sizeof(float));
    b->ntok++;
    float *ks = L->ksum + (size_t)bi * d;              /* running mean key */
    float inv = 1.0f / (float)b->ntok;
    for (int j = 0; j < d; j++) ks[j] += (k[j] - ks[j]) * inv;
    b->last_used = ++S->tick;
}

static void ensure_scratch(OocState *S, int n) {
    if (n <= S->scratch_cap) return;
    int cap = S->scratch_cap ? S->scratch_cap : 64;
    while (cap < n) cap *= 2;
    S->idbuf = (int *)realloc(S->idbuf, (size_t)cap * sizeof(int));
    S->seen = (unsigned char *)realloc(S->seen, (size_t)cap);
    S->scratch_cap = cap;
}

static void o_attend(KVStore *s, int layer, const float *q, float *out) {
    OocState *S = (OocState *)s->state;
    OocLayer *L = &S->layers[layer];
    int d = S->kv_dim, nb = L->n_blocks;
    ensure_scratch(S, nb);
    memset(S->seen, 0, (size_t)nb);

    SelectCtx ctx = { layer, d, nb, S->g, S->sink_blocks, S->window_blocks, q, L->ksum };
    int budget = S->strat->budget > 0 ? S->strat->budget : nb;
    if (budget > nb) budget = nb;
    int nsel = S->strat->select(&ctx, S->idbuf, budget);
    for (int i = 0; i < nsel; i++)
        if (S->idbuf[i] >= 0 && S->idbuf[i] < nb) S->seen[S->idbuf[i]] = 1;
    for (int i = 0; i < S->sink_blocks && i < nb; i++) S->seen[i] = 1;
    for (int i = nb - S->window_blocks; i < nb; i++) if (i >= 0) S->seen[i] = 1;

    float scale = 1.0f / sqrtf((float)d);
    KVSoftmax sm; kv_softmax_init(&sm, S->acc, d);
    for (int i = 0; i < nb; i++) {                     /* ascending global order */
        if (!S->seen[i]) continue;
        ensure_resident(S, L, i);
        Block *b = &L->blocks[i];
        for (int t = 0; t < b->ntok; t++)
            kv_softmax_step(&sm, q, b->k + (size_t)t * d, b->v + (size_t)t * d, scale);
        release(b);
    }
    kv_softmax_finish(&sm, out);
}

static void o_stats(const KVStore *s, KVStats *out) {
    OocState *S = (OocState *)s->state;
    *out = S->st;
    out->resident_bytes = S->resident_bytes;
    out->capacity_bytes = S->capacity_bytes;
}

static void o_destroy(KVStore *s) {
    OocState *S = (OocState *)s->state;
    for (int l = 0; l < S->n_layers; l++) {
        OocLayer *L = &S->layers[l];
        for (int i = 0; i < L->n_blocks; i++) { free(L->blocks[i].k); free(L->blocks[i].v); }
        free(L->blocks); free(L->ksum);
    }
    free(S->layers); free(S->acc); free(S->idbuf); free(S->seen);
    if (S->fd >= 0) close(S->fd);
    if (S->path[0]) unlink(S->path);
    free(S); free(s);
}

static const KVStoreOps OOC_OPS = { o_append, o_attend, o_stats, o_destroy };

KVStore *kvstore_ooc_create(KVConfig cfg, const SelectStrategy *strat, const char *workdir) {
    KVStore *s = (KVStore *)calloc(1, sizeof(KVStore));
    OocState *S = (OocState *)calloc(1, sizeof(OocState));
    S->n_layers = cfg.n_layers; S->kv_dim = cfg.kv_dim; S->g = cfg.block_tokens;
    S->sink_blocks = blocks_for(cfg.sink_tokens, cfg.block_tokens);
    S->window_blocks = blocks_for(cfg.window_tokens, cfg.block_tokens);
    S->strat = strat ? strat : &VIREO_SELECT_ALL;
    S->capacity_bytes = cfg.capacity_bytes;
    S->half_bytes = (uint64_t)cfg.block_tokens * cfg.kv_dim * sizeof(float);
    S->block_bytes = S->half_bytes * 2;
    S->layers = (OocLayer *)calloc((size_t)cfg.n_layers, sizeof(OocLayer));
    S->acc = (float *)malloc((size_t)cfg.kv_dim * sizeof(float));
    S->fd = -1;
    const char *dir = (workdir && *workdir) ? workdir : "/tmp";
    snprintf(S->path, sizeof(S->path), "%s/vireo_kv_XXXXXX", dir);
    S->fd = mkstemp(S->path);
    s->ops = &OOC_OPS; s->cfg = cfg; s->state = S;
    return s;
}
