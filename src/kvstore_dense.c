/* Dense in-RAM KV store. The reference implementation: simple, obviously
 * correct, holds all KV resident. It defines the output the out-of-core store
 * must match. */
#include <stdlib.h>
#include "kvstore.h"

typedef struct {
    float *k;      /* cap * kv_dim */
    float *v;      /* cap * kv_dim */
    int n;         /* tokens appended */
    int cap;
} DenseLayer;

typedef struct {
    DenseLayer *layers;  /* n_layers */
    float *acc;          /* scratch, kv_dim */
} DenseState;

static void dl_reserve(DenseLayer *dl, int need, int kv_dim) {
    if (need <= dl->cap) return;
    int cap = dl->cap ? dl->cap * 2 : 256;
    while (cap < need) cap *= 2;
    dl->k = (float *)realloc(dl->k, (size_t)cap * kv_dim * sizeof(float));
    dl->v = (float *)realloc(dl->v, (size_t)cap * kv_dim * sizeof(float));
    dl->cap = cap;
}

static void d_append(KVStore *s, int layer, int pos, const float *k, const float *v) {
    (void)pos;
    DenseState *st = (DenseState *)s->state;
    DenseLayer *dl = &st->layers[layer];
    int d = s->cfg.kv_dim;
    dl_reserve(dl, dl->n + 1, d);
    memcpy(dl->k + (size_t)dl->n * d, k, (size_t)d * sizeof(float));
    memcpy(dl->v + (size_t)dl->n * d, v, (size_t)d * sizeof(float));
    dl->n++;
}

static void d_attend(KVStore *s, int layer, const float *q, float *out) {
    DenseState *st = (DenseState *)s->state;
    DenseLayer *dl = &st->layers[layer];
    int d = s->cfg.kv_dim;
    float scale = 1.0f / sqrtf((float)d);
    KVSoftmax sm;
    kv_softmax_init(&sm, st->acc, d);
    for (int t = 0; t < dl->n; t++)
        kv_softmax_step(&sm, q, dl->k + (size_t)t * d, dl->v + (size_t)t * d, scale);
    kv_softmax_finish(&sm, out);
}

static void d_stats(const KVStore *s, KVStats *out) {
    DenseState *st = (DenseState *)s->state;
    memset(out, 0, sizeof(*out));
    uint64_t bytes = 0;
    for (int l = 0; l < s->cfg.n_layers; l++)
        bytes += (uint64_t)st->layers[l].n * s->cfg.kv_dim * 2 * sizeof(float);
    out->resident_bytes = bytes;
    out->capacity_bytes = 0; /* unbounded */
}

static void d_destroy(KVStore *s) {
    DenseState *st = (DenseState *)s->state;
    for (int l = 0; l < s->cfg.n_layers; l++) { free(st->layers[l].k); free(st->layers[l].v); }
    free(st->layers); free(st->acc); free(st); free(s);
}

static const KVStoreOps DENSE_OPS = { d_append, d_attend, d_stats, d_destroy };

KVStore *kvstore_dense_create(KVConfig cfg) {
    KVStore *s = (KVStore *)calloc(1, sizeof(KVStore));
    DenseState *st = (DenseState *)calloc(1, sizeof(DenseState));
    st->layers = (DenseLayer *)calloc((size_t)cfg.n_layers, sizeof(DenseLayer));
    st->acc = (float *)malloc((size_t)cfg.kv_dim * sizeof(float));
    s->ops = &DENSE_OPS; s->cfg = cfg; s->state = st;
    return s;
}
