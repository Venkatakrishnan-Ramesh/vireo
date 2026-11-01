#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "expert_store.h"

typedef struct {
    ExpertSlices loc;
    void *gate, *up, *down;   /* resident buffers, NULL when not resident */
    uint64_t bytes;           /* gate+up+down */
    int resident;
    int lease;
    int registered;
    uint64_t last_used;
} Slot;

struct ExpertStore {
    int fd;
    int n_layers, n_experts, n_slots;
    Slot *slots;              /* n_layers * n_experts */
    uint64_t tick;
    uint64_t resident_bytes, capacity_bytes;
    ExpertStats st;
};

static int idx(ExpertStore *s, int layer, int expert) { return layer * s->n_experts + expert; }

ExpertStore *expert_store_create(int fd, uint64_t capacity_bytes, int n_layers, int n_experts) {
    ExpertStore *s = calloc(1, sizeof(*s));
    s->fd = fd; s->n_layers = n_layers; s->n_experts = n_experts;
    s->n_slots = n_layers * n_experts;
    s->slots = calloc(s->n_slots, sizeof(Slot));
    s->capacity_bytes = capacity_bytes;
    return s;
}

void expert_store_register(ExpertStore *s, int layer, int expert, const ExpertSlices *loc) {
    Slot *sl = &s->slots[idx(s, layer, expert)];
    sl->loc = *loc;
    sl->bytes = (uint64_t)loc->gate_bytes + loc->up_bytes + loc->down_bytes;
    sl->registered = 1;
}

static void evict_one(ExpertStore *s, Slot *v) {
    free(v->gate); free(v->up); free(v->down);
    v->gate = v->up = v->down = NULL;
    v->resident = 0;
    s->resident_bytes -= v->bytes;
    s->st.experts_resident--;
}

static void make_room(ExpertStore *s, uint64_t need) {
    if (!s->capacity_bytes) return;
    while (s->resident_bytes + need > s->capacity_bytes) {
        Slot *victim = NULL; uint64_t best = (uint64_t)-1;
        for (int i = 0; i < s->n_slots; i++) {
            Slot *sl = &s->slots[i];
            if (sl->resident && !sl->lease && sl->last_used < best) { best = sl->last_used; victim = sl; }
        }
        if (!victim) break;   /* all resident experts are leased; stay over budget safely */
        evict_one(s, victim);
    }
}

static void *load_slice(ExpertStore *s, uint64_t off, uint32_t bytes) {
    void *buf = malloc(bytes);
    if (buf) { if (pread(s->fd, buf, bytes, (off_t)off) < 0) { /* leave as-is */ } }
    s->st.bytes_read += bytes;
    return buf;
}

int expert_store_lookup(ExpertStore *s, int layer, int expert, ExpertView *view) {
    Slot *sl = &s->slots[idx(s, layer, expert)];
    s->st.requests++;
    if (!sl->registered) { memset(view, 0, sizeof(*view)); return -1; }
    if (sl->resident) {
        s->st.hits++;
    } else {
        s->st.misses++;
        make_room(s, sl->bytes);
        sl->gate = load_slice(s, sl->loc.gate_off, sl->loc.gate_bytes);
        sl->up   = load_slice(s, sl->loc.up_off,   sl->loc.up_bytes);
        sl->down = load_slice(s, sl->loc.down_off, sl->loc.down_bytes);
        sl->resident = 1;
        s->resident_bytes += sl->bytes;
        s->st.experts_resident++;
    }
    sl->last_used = ++s->tick;
    sl->lease++;
    view->gate = sl->gate; view->up = sl->up; view->down = sl->down; view->slot = sl;
    return 0;
}

void expert_store_release(ExpertStore *s, ExpertView *view) {
    (void)s;
    if (view && view->slot) { Slot *sl = view->slot; if (sl->lease > 0) sl->lease--; }
    if (view) memset(view, 0, sizeof(*view));
}

void expert_store_stats(const ExpertStore *s, ExpertStats *out) {
    *out = s->st;
    out->resident_bytes = s->resident_bytes;
    out->capacity_bytes = s->capacity_bytes;
    out->experts_resident = s->st.experts_resident;
}

void expert_store_destroy(ExpertStore *s) {
    if (!s) return;
    for (int i = 0; i < s->n_slots; i++) { Slot *sl = &s->slots[i]; free(sl->gate); free(sl->up); free(sl->down); }
    free(s->slots); free(s);
}
