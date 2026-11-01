/* ExpertStore unit test: a synthetic file of experts, each with 3 weight
 * slices filled with a deterministic pattern. Verifies the pager returns the
 * correct bytes, honors the RAM budget, evicts cold experts, and never evicts
 * a leased one. No model required. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "expert_store.h"

#define NL 4
#define NE 16          /* 64 experts total */
#define GATE 3072
#define UP   3072
#define DOWN 2048

static unsigned char pat(int layer, int expert, int slice, int i) {
    return (unsigned char)((layer*131 + expert*17 + slice*7 + i) & 0xFF);
}

int main(void) {
    char path[] = "/tmp/vireo_experts_XXXXXX";
    int fd = mkstemp(path);
    /* write experts: for each (l,e): gate, up, down back to back */
    ExpertSlices locs[NL*NE];
    uint64_t off = 0;
    unsigned char buf[GATE > UP ? (GATE>DOWN?GATE:DOWN) : (UP>DOWN?UP:DOWN)];
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) {
        ExpertSlices *L = &locs[l*NE+e];
        for (int i=0;i<GATE;i++) buf[i]=pat(l,e,0,i); pwrite(fd,buf,GATE,off); L->gate_off=off; L->gate_bytes=GATE; off+=GATE;
        for (int i=0;i<UP;i++)   buf[i]=pat(l,e,1,i); pwrite(fd,buf,UP,off);   L->up_off=off;   L->up_bytes=UP;     off+=UP;
        for (int i=0;i<DOWN;i++) buf[i]=pat(l,e,2,i); pwrite(fd,buf,DOWN,off); L->down_off=off; L->down_bytes=DOWN; off+=DOWN;
    }

    uint64_t expert_bytes = GATE+UP+DOWN;
    uint64_t cap = expert_bytes * 10;      /* only 10 of 64 experts fit */
    ExpertStore *s = expert_store_create(fd, cap, NL, NE);
    for (int l=0;l<NL;l++) for (int e=0;e<NE;e++) expert_store_register(s,l,e,&locs[l*NE+e]);

    /* 1. correctness: look up every expert, verify bytes */
    int ok = 1;
    for (int l=0;l<NL;l++) for (int e=0;e<NE;e++) {
        ExpertView v;
        if (expert_store_lookup(s,l,e,&v)) { printf("FAIL lookup %d,%d\n",l,e); ok=0; continue; }
        const unsigned char *g=v.gate,*u=v.up,*d=v.down;
        for (int i=0;i<GATE;i++) if (g[i]!=pat(l,e,0,i)) { printf("FAIL gate %d,%d@%d\n",l,e,i); ok=0; break; }
        for (int i=0;i<UP;i++)   if (u[i]!=pat(l,e,1,i)) { printf("FAIL up %d,%d@%d\n",l,e,i); ok=0; break; }
        for (int i=0;i<DOWN;i++) if (d[i]!=pat(l,e,2,i)) { printf("FAIL down %d,%d@%d\n",l,e,i); ok=0; break; }
        expert_store_release(s,&v);
    }

    /* 2. RAM budget respected despite touching all 64 experts */
    ExpertStats st; expert_store_stats(s,&st);
    if (st.resident_bytes > st.capacity_bytes) { printf("FAIL over budget: %llu > %llu\n",
        (unsigned long long)st.resident_bytes,(unsigned long long)st.capacity_bytes); ok=0; }
    if (st.misses < 64) { printf("WARN: expected eviction/reload (misses=%llu)\n",(unsigned long long)st.misses); }

    /* 3. lease pin: a leased expert must survive a flood of other lookups */
    ExpertView pinned; expert_store_lookup(s,0,0,&pinned);
    const unsigned char *pg = pinned.gate;
    for (int l=0;l<NL;l++) for (int e=1;e<NE;e++){ ExpertView v; if(!expert_store_lookup(s,l,e,&v)) expert_store_release(s,&v); }
    for (int i=0;i<GATE;i++) if (pg[i]!=pat(0,0,0,i)) { printf("FAIL: leased expert was evicted/corrupted @%d\n",i); ok=0; break; }
    expert_store_release(s,&pinned);

    expert_store_stats(s,&st);
    printf("experts: 64 total, cap=%d resident | requests=%llu hits=%llu misses=%llu read=%lluKB resident=%lluKB/%lluKB\n",
        (int)(cap/expert_bytes),(unsigned long long)st.requests,(unsigned long long)st.hits,
        (unsigned long long)st.misses,(unsigned long long)(st.bytes_read/1024),
        (unsigned long long)(st.resident_bytes/1024),(unsigned long long)(st.capacity_bytes/1024));

    expert_store_destroy(s); close(fd); unlink(path);
    printf(ok ? "\nPASS\n" : "\nFAIL\n");
    return ok?0:1;
}
