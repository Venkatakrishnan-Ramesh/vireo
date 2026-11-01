/* Oracle test: the out-of-core store with the "all" strategy must match the
 * dense store bit-for-bit, while staying under its RAM budget. Also exercises
 * the lossy strategies and reports their divergence for information. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "kvstore.h"
#include "select/select.h"

static uint32_t rng = 0x1234567u;
static float frand(void) { rng = rng * 1664525u + 1013904223u; return (float)(rng >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f; }

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0; for (int i = 0; i < n; i++) { float d = fabsf(a[i] - b[i]); if (d > m) m = d; } return m;
}

int main(void) {
    KVConfig cfg = { .n_layers = 3, .kv_dim = 16, .block_tokens = 8,
                     .sink_tokens = 8, .window_tokens = 16,
                     .capacity_bytes = 0 };
    int N = 500, d = cfg.kv_dim, L = cfg.n_layers;
    /* capacity for ooc: ~5 blocks per layer -> forces eviction */
    uint64_t block_bytes = (uint64_t)cfg.block_tokens * d * 2 * sizeof(float);
    KVConfig cfg_ooc = cfg; cfg_ooc.capacity_bytes = block_bytes * 20;

    KVStore *dense = kvstore_dense_create(cfg);
    KVStore *ooc   = kvstore_ooc_create(cfg_ooc, &VIREO_SELECT_ALL, "/tmp");

    float *k = malloc(d * sizeof(float)), *v = malloc(d * sizeof(float));
    for (int t = 0; t < N; t++)
        for (int l = 0; l < L; l++) {
            for (int j = 0; j < d; j++) { k[j] = frand(); v[j] = frand(); }
            kv_append(dense, l, t, k, v);
            kv_append(ooc,   l, t, k, v);
        }

    float *od = malloc(d * sizeof(float)), *oo = malloc(d * sizeof(float));
    float worst = 0;
    for (int qn = 0; qn < 200; qn++) {
        int l = qn % L;
        float q[64]; for (int j = 0; j < d; j++) q[j] = frand();
        kv_attend(dense, l, q, od);
        kv_attend(ooc,   l, q, oo);
        float diff = max_abs_diff(od, oo, d);
        if (diff > worst) worst = diff;
    }

    KVStats s; kv_stats(ooc, &s);
    printf("oracle: worst |dense - ooc(all)| = %.3e\n", worst);
    printf("ooc: requests=%llu hits=%llu misses=%llu read=%lluKB spilled=%lluKB resident=%lluKB cap=%lluKB\n",
           (unsigned long long)s.requests, (unsigned long long)s.hits, (unsigned long long)s.misses,
           (unsigned long long)(s.bytes_read / 1024), (unsigned long long)(s.bytes_spilled / 1024),
           (unsigned long long)(s.resident_bytes / 1024), (unsigned long long)(s.capacity_bytes / 1024));

    int ok = 1;
    if (worst > 1e-6f) { printf("FAIL: ooc(all) diverges from dense\n"); ok = 0; }
    if (s.resident_bytes > s.capacity_bytes) { printf("FAIL: over RAM budget\n"); ok = 0; }
    if (s.misses == 0) { printf("WARN: no eviction happened (test too small)\n"); }

    /* informational: lossy strategies */
    KVStore *ooc_recent = kvstore_ooc_create(cfg_ooc, &VIREO_SELECT_RECENT, "/tmp");
    KVStore *ooc_topk   = kvstore_ooc_create(cfg_ooc, vireo_select_topk(4), "/tmp");
    rng = 0x1234567u;
    for (int t = 0; t < N; t++) for (int l = 0; l < L; l++) {
        for (int j = 0; j < d; j++) { k[j] = frand(); v[j] = frand(); }
        kv_append(ooc_recent, l, t, k, v); kv_append(ooc_topk, l, t, k, v);
    }
    float wr = 0, wt = 0;
    for (int qn = 0; qn < 100; qn++) {
        int l = qn % L; float q[64]; for (int j = 0; j < d; j++) q[j] = frand();
        float r[64], tk[64];
        kv_attend(dense, l, q, od);
        kv_attend(ooc_recent, l, q, r);  float a = max_abs_diff(od, r, d);  if (a > wr) wr = a;
        kv_attend(ooc_topk, l, q, tk);   float b = max_abs_diff(od, tk, d); if (b > wt) wt = b;
    }
    printf("approx: recent diverges %.3e, topk(4) diverges %.3e (expected > 0; random data has no structure)\n", wr, wt);

    kv_destroy(dense); kv_destroy(ooc); kv_destroy(ooc_recent); kv_destroy(ooc_topk);
    free(k); free(v); free(od); free(oo);
    printf(ok ? "\nPASS\n" : "\nFAIL\n");
    return ok ? 0 : 1;
}
