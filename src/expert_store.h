/* vireo ExpertStore — stream MoE expert weights from disk.
 *
 * An MoE layer has N experts but only fires k per token (e.g. 64 experts,
 * top-8). The experts are the bulk of the weights. ExpertStore keeps a bounded
 * set of experts resident and pages the rest from disk on demand, so a model
 * far larger than RAM runs with only its active experts in memory.
 *
 * This is the twin of KVStore: the same lease-LRU pager, pointed at expert
 * weights instead of KV blocks. Each expert owns three weight slices
 * (gate, up, down for its SwiGLU FFN).
 */
#ifndef VIREO_EXPERT_STORE_H
#define VIREO_EXPERT_STORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t requests, hits, misses, bytes_read;
    uint64_t resident_bytes, capacity_bytes, experts_resident;
} ExpertStats;

/* Where one expert's three weight slices live in the backing file. */
typedef struct {
    uint64_t gate_off, up_off, down_off;      /* absolute byte offsets        */
    uint32_t gate_bytes, up_bytes, down_bytes;
} ExpertSlices;

/* A leased view of one resident expert. Valid until release(). */
typedef struct {
    const void *gate, *up, *down;
    void *slot;        /* opaque lease handle; do not touch */
} ExpertView;

typedef struct ExpertStore ExpertStore;

/* fd: the open GGUF file. capacity_bytes: RAM budget for resident experts
 * (0 = unbounded). */
ExpertStore *expert_store_create(int fd, uint64_t capacity_bytes,
                                 int n_layers, int n_experts);
void expert_store_register(ExpertStore *s, int layer, int expert,
                           const ExpertSlices *loc);
/* Loads the expert if not resident (evicting the coldest unleased one if over
 * budget), leases it, and fills view. Returns 0 on success. */
int  expert_store_lookup(ExpertStore *s, int layer, int expert, ExpertView *view);
void expert_store_release(ExpertStore *s, ExpertView *view);
void expert_store_stats(const ExpertStore *s, ExpertStats *out);
void expert_store_destroy(ExpertStore *s);

#ifdef __cplusplus
}
#endif
#endif
