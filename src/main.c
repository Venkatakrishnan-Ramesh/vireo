/* vireo demo CLI.
 *   vireo bench [tokens] [strategy]   human-readable headline numbers
 *   vireo csv   [tokens] [strategy]   one CSV line: tokens,resident,dense,hits,toks_per_s
 * Streams a long synthetic context through the out-of-core store and measures
 * the pager: resident RAM vs. what a dense cache would need, hit rate,
 * throughput. Real-model integration is vireo-run (ROADMAP M1, done).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "kvstore.h"
#include "select/select.h"

static uint32_t rng = 777;
static float frand(void){ rng = rng*1664525u+1013904223u; return (float)(rng>>8)*(1.0f/16777216.0f)*2-1; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int main(int argc, char **argv) {
    int csv = argc > 1 && !strcmp(argv[1], "csv");
    int bench = argc > 1 && !strcmp(argv[1], "bench");
    if (!csv && !bench) {
        fprintf(stderr, "usage: %s bench|csv [tokens] [strategy all|recent|topk]\n", argv[0]);
        return 2;
    }
    int N = argc > 2 ? atoi(argv[2]) : 100000;
    const char *sname = argc > 3 ? argv[3] : "topk";
    KVConfig cfg = { .n_layers=1, .kv_dim=128, .block_tokens=64,
                     .sink_tokens=64, .window_tokens=2048,
                     .capacity_bytes=(uint64_t)64*128*2*sizeof(float)*64 };
    const SelectStrategy *strat = &VIREO_SELECT_ALL;
    if (!strcmp(sname,"recent")) strat = &VIREO_SELECT_RECENT;
    else if (!strcmp(sname,"topk")) strat = vireo_select_topk(16);

    KVStore *s = kvstore_ooc_create(cfg, strat, "/tmp");
    float *k = malloc(cfg.kv_dim*sizeof(float)), *v = malloc(cfg.kv_dim*sizeof(float));
    float *out = malloc(cfg.kv_dim*sizeof(float)), *q = malloc(cfg.kv_dim*sizeof(float));

    double t0 = now();
    for (int t = 0; t < N; t++) {
        for (int j=0;j<cfg.kv_dim;j++){k[j]=frand();v[j]=frand();}
        kv_append(s, 0, t, k, v);
        if ((t & 255) == 255) { for (int j=0;j<cfg.kv_dim;j++) q[j]=frand(); kv_attend(s,0,q,out); }
    }
    double dt = now() - t0;

    KVStats st; kv_stats(s, &st);
    uint64_t dense_bytes = (uint64_t)N * cfg.kv_dim * 2 * sizeof(float);
    double hits = st.requests ? 100.0*st.hits/st.requests : 0;
    double toks = N/dt;

    if (csv) {
        printf("%d,%llu,%llu,%.2f,%.0f\n", N,
               (unsigned long long)st.resident_bytes, (unsigned long long)dense_bytes, hits, toks);
    } else {
        printf("vireo bench  strategy=%s  context=%d tokens\n", strat->name, N);
        printf("  resident RAM : %6.1f MB   (dense would need %6.1f MB  ->  %.0fx less)\n",
               st.resident_bytes/1048576.0, dense_bytes/1048576.0,
               st.resident_bytes ? (double)dense_bytes/st.resident_bytes : 0);
        printf("  cache hits   : %.1f%%  (%llu/%llu)\n", hits,
               (unsigned long long)st.hits, (unsigned long long)st.requests);
        printf("  disk read    : %.1f MB   spilled %.1f MB\n",
               st.bytes_read/1048576.0, st.bytes_spilled/1048576.0);
        printf("  throughput   : %.0f tokens/sec\n", toks);
    }
    kv_destroy(s); free(k);free(v);free(out);free(q);
    return 0;
}
