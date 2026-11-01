/* vireo — out-of-core KV cache for long-context transformers.
 *
 * KVStore is the one abstraction the whole project hangs off. It stores the
 * key/value vectors a transformer produces per token per layer, and answers
 * attention queries over them. A dense implementation keeps everything in RAM;
 * the out-of-core implementation keeps only a hot working set resident and
 * streams cold KV blocks from disk on demand.
 *
 * Correctness contract: for identical (append order, queries), the out-of-core
 * store with the "all" selection strategy MUST produce bit-identical output to
 * the dense store. tests/test_kvstore_oracle.c checks exactly that, and it is
 * the line you do not cross while optimizing.
 */
#ifndef VIREO_KVSTORE_H
#define VIREO_KVSTORE_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n_layers;
    int kv_dim;              /* length of each key and value vector        */
    int block_tokens;        /* tokens grouped per on-disk block (G)       */
    int sink_tokens;         /* always-resident attention sinks (prefix)   */
    int window_tokens;       /* always-resident recent window              */
    uint64_t capacity_bytes; /* RAM budget for paged blocks (0 = unbounded)*/
} KVConfig;

typedef struct {
    uint64_t requests;
    uint64_t hits;
    uint64_t misses;
    uint64_t prefetched;
    uint64_t bytes_read;
    uint64_t bytes_spilled;
    uint64_t resident_bytes;
    uint64_t capacity_bytes;
} KVStats;

typedef struct KVStore KVStore;

typedef struct {
    void (*append)(KVStore *s, int layer, int pos, const float *k, const float *v);
    void (*attend)(KVStore *s, int layer, const float *q, float *out);
    void (*stats)(const KVStore *s, KVStats *out);
    void (*destroy)(KVStore *s);
} KVStoreOps;

struct KVStore {
    const KVStoreOps *ops;
    KVConfig cfg;
    void *state;
};

static inline void kv_append(KVStore *s, int layer, int pos,
                             const float *k, const float *v) {
    s->ops->append(s, layer, pos, k, v);
}
static inline void kv_attend(KVStore *s, int layer, const float *q, float *out) {
    s->ops->attend(s, layer, q, out);
}
static inline void kv_stats(const KVStore *s, KVStats *out) { s->ops->stats(s, out); }
static inline void kv_destroy(KVStore *s) {
    if (s && s->ops && s->ops->destroy) s->ops->destroy(s);
}

/* --- shared, numerically-stable online softmax attention ------------------
 * Both implementations accumulate through these so that, token-for-token in
 * the same global order, their results are identical. Do not reorder token
 * visitation if you want the oracle test to pass. */
typedef struct { float m; float l; float *acc; int dim; } KVSoftmax;

static inline float kv_dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
static inline void kv_softmax_init(KVSoftmax *sm, float *acc, int dim) {
    sm->m = -INFINITY; sm->l = 0.0f; sm->acc = acc; sm->dim = dim;
    memset(acc, 0, (size_t)dim * sizeof(float));
}
static inline void kv_softmax_step(KVSoftmax *sm, const float *q,
                                   const float *k, const float *v, float scale) {
    float s = kv_dot(q, k, sm->dim) * scale;
    float new_m = s > sm->m ? s : sm->m;
    float c = expf(sm->m - new_m);
    float p = expf(s - new_m);
    sm->l = sm->l * c + p;
    for (int i = 0; i < sm->dim; i++) sm->acc[i] = sm->acc[i] * c + p * v[i];
    sm->m = new_m;
}
static inline void kv_softmax_finish(KVSoftmax *sm, float *out) {
    float inv = sm->l > 0.0f ? 1.0f / sm->l : 0.0f;
    for (int i = 0; i < sm->dim; i++) out[i] = sm->acc[i] * inv;
}

KVStore *kvstore_dense_create(KVConfig cfg);
struct SelectStrategy;
KVStore *kvstore_ooc_create(KVConfig cfg, const struct SelectStrategy *strat,
                            const char *workdir);

#ifdef __cplusplus
}
#endif
#endif /* VIREO_KVSTORE_H */
