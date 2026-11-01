#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "gguf.h"

/* ---- cursor over the mmap (little-endian host assumed) ---- */
typedef struct { const uint8_t *p, *end; } cur;
static int rd(cur *c, void *dst, size_t n) {
    if ((size_t)(c->end - c->p) < n) return -1;
    memcpy(dst, c->p, n); c->p += n; return 0;
}
static int rd_u32(cur *c, uint32_t *v){ return rd(c, v, 4); }
static int rd_u64(cur *c, uint64_t *v){ return rd(c, v, 8); }
static int rd_str(cur *c, gguf_str *s){
    uint64_t n;
    if (rd_u64(c,&n)) return -1;
    if ((size_t)(c->end - c->p) < n) return -1;
    s->ptr=(const char*)c->p; s->len=n; c->p+=n;
    return 0;
}

int gguf_type_block(uint32_t t, int *be, int *bb) {
    switch (t) {
        case GGML_F32:  *be=1;  *bb=4;   return 0;
        case GGML_F16:  *be=1;  *bb=2;   return 0;
        case GGML_Q8_0: *be=32; *bb=34;  return 0;  /* fp16 scale + 32 int8 */
        case GGML_Q4_0: *be=32; *bb=18;  return 0;  /* fp16 scale + 16 bytes */
        case GGML_Q4_K: *be=256;*bb=144; return 0;
        case GGML_Q6_K: *be=256;*bb=210; return 0;
        default: return -1;
    }
}

/* size in bytes of one metadata scalar type (0 for variable/array). */
static int scalar_size(uint32_t t) {
    switch (t) {
        case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
        case GGUF_U16: case GGUF_I16: return 2;
        case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
        case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
        default: return 0;
    }
}

static int read_value(cur *c, gguf_kv *kv, uint32_t type) {
    kv->type = type;
    if (type == GGUF_STRING) return rd_str(c, &kv->str);
    if (type == GGUF_ARRAY) {
        if (rd_u32(c, &kv->arr_type)) return -1;
        if (rd_u64(c, &kv->arr_len)) return -1;
        kv->arr_data = c->p;
        /* skip the array body so parsing can continue */
        if (kv->arr_type == GGUF_STRING) {
            for (uint64_t i = 0; i < kv->arr_len; i++) { gguf_str s; if (rd_str(c,&s)) return -1; }
        } else {
            int sz = scalar_size(kv->arr_type);
            if (!sz) return -1;
            if ((size_t)(c->end - c->p) < sz * kv->arr_len) return -1;
            c->p += (size_t)sz * kv->arr_len;
        }
        return 0;
    }
    int sz = scalar_size(type);
    if (!sz) return -1;
    uint64_t raw = 0;
    if (rd(c, &raw, sz)) return -1;
    if (type == GGUF_F32) { float f; memcpy(&f,&raw,4); kv->val.f = f; }
    else if (type == GGUF_F64) { double d; memcpy(&d,&raw,8); kv->val.f = d; }
    else kv->val.u = raw;
    return 0;
}

gguf_ctx *gguf_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("gguf open"); return NULL; }
    struct stat st; if (fstat(fd, &st)) { close(fd); return NULL; }
    void *map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return NULL; }

    gguf_ctx *g = calloc(1, sizeof(*g));
    g->fd = fd; g->map = map; g->size = st.st_size; g->alignment = 32;
    cur c = { map, (const uint8_t*)map + st.st_size };

    char magic[4]; uint32_t ver = 0;
    if (rd(&c, magic, 4) || memcmp(magic, "GGUF", 4)) { fprintf(stderr,"not a GGUF file\n"); goto fail; }
    if (rd_u32(&c, &ver) || (ver != 2 && ver != 3)) { fprintf(stderr,"GGUF version %u unsupported\n", ver); goto fail; }
    g->version = ver;
    if (rd_u64(&c, &g->n_tensors) || rd_u64(&c, &g->n_kv)) goto fail;

    g->kv = calloc(g->n_kv, sizeof(gguf_kv));
    for (uint64_t i = 0; i < g->n_kv; i++) {
        uint32_t type;
        if (rd_str(&c, &g->kv[i].key) || rd_u32(&c, &type)) goto fail;
        if (read_value(&c, &g->kv[i], type)) { fprintf(stderr,"bad metadata value at %llu\n",(unsigned long long)i); goto fail; }
    }

    /* alignment override */
    const gguf_kv *al = gguf_find(g, "general.alignment");
    if (al && al->type == GGUF_U32) g->alignment = al->val.u;

    g->tensors = calloc(g->n_tensors, sizeof(gguf_tensor));
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        if (rd_str(&c, &t->name) || rd_u32(&c, &t->n_dims)) goto fail;
        if (t->n_dims > 4) { fprintf(stderr,"tensor %llu n_dims=%u\n",(unsigned long long)i,t->n_dims); goto fail; }
        t->n_elems = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) { if (rd_u64(&c, &t->dims[d])) goto fail; t->n_elems *= t->dims[d]; }
        if (rd_u32(&c, &t->ggml_type) || rd_u64(&c, &t->offset)) goto fail;
        int be, bb;
        if (gguf_type_block(t->ggml_type, &be, &bb) == 0)
            t->nbytes = t->n_elems / be * bb;
    }

    /* data section starts at the next alignment boundary after the tensor table */
    uint64_t pos = (uint64_t)(c.p - (const uint8_t*)map);
    uint64_t pad = (g->alignment - (pos % g->alignment)) % g->alignment;
    g->data = (const uint8_t*)map + pos + pad;
    for (uint64_t i = 0; i < g->n_tensors; i++)
        g->tensors[i].data = g->data + g->tensors[i].offset;

    return g;
fail:
    gguf_close(g);
    return NULL;
}

void gguf_close(gguf_ctx *g) {
    if (!g) return;
    if (g->map && g->map != MAP_FAILED) munmap(g->map, g->size);
    if (g->fd >= 0) close(g->fd);
    free(g->kv); free(g->tensors); free(g);
}

static int key_eq(const gguf_str *s, const char *k) {
    size_t n = strlen(k); return s->len == n && memcmp(s->ptr, k, n) == 0;
}
const gguf_kv *gguf_find(const gguf_ctx *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++) if (key_eq(&g->kv[i].key, key)) return &g->kv[i];
    return NULL;
}
int gguf_get_u64(const gguf_ctx *g, const char *key, uint64_t *out) {
    const gguf_kv *k = gguf_find(g, key); if (!k) return -1;
    switch (k->type) {
        case GGUF_U8: case GGUF_U16: case GGUF_U32: case GGUF_U64:
        case GGUF_I8: case GGUF_I16: case GGUF_I32: case GGUF_I64: case GGUF_BOOL:
            *out = k->val.u; return 0;
        default: return -1;
    }
}
int gguf_get_f32(const gguf_ctx *g, const char *key, float *out) {
    const gguf_kv *k = gguf_find(g, key); if (!k) return -1;
    if (k->type == GGUF_F32 || k->type == GGUF_F64) { *out = (float)k->val.f; return 0; }
    return -1;
}
const char *gguf_get_str(const gguf_ctx *g, const char *key, uint64_t *len) {
    const gguf_kv *k = gguf_find(g, key); if (!k || k->type != GGUF_STRING) return NULL;
    if (len) *len = k->str.len; return k->str.ptr;
}
const gguf_tensor *gguf_tensor_by_name(const gguf_ctx *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++) if (key_eq(&g->tensors[i].name, name)) return &g->tensors[i];
    return NULL;
}

static void print_str(gguf_str s){ fwrite(s.ptr,1,s.len>64?64:s.len,stdout); if(s.len>64)printf("..."); }
void gguf_dump(const gguf_ctx *g) {
    printf("GGUF v%u  tensors=%llu  metadata=%llu  alignment=%llu  data@0x%lx\n",
           g->version,(unsigned long long)g->n_tensors,(unsigned long long)g->n_kv,
           (unsigned long long)g->alignment,(unsigned long)(g->data-(const uint8_t*)g->map));
    printf("-- metadata (scalars) --\n");
    for (uint64_t i = 0; i < g->n_kv; i++) {
        gguf_kv *k = &g->kv[i];
        if (k->type == GGUF_ARRAY) { printf("  "); print_str(k->key); printf(" = [array type=%u len=%llu]\n",k->arr_type,(unsigned long long)k->arr_len); continue; }
        printf("  "); print_str(k->key); printf(" = ");
        if (k->type == GGUF_STRING) print_str(k->str);
        else if (k->type == GGUF_F32 || k->type == GGUF_F64) printf("%g", k->val.f);
        else printf("%llu", (unsigned long long)k->val.u);
        printf("\n");
    }
    printf("-- tensors (first 12) --\n");
    const char *tn[16] = {"F32","F16","Q4_0","Q4_1","?","?","Q5_0","Q5_1","Q8_0","Q8_1","Q2_K","Q3_K","Q4_K","Q5_K","Q6_K","Q8_K"};
    for (uint64_t i = 0; i < g->n_tensors && i < 12; i++) {
        gguf_tensor *t = &g->tensors[i];
        printf("  "); print_str(t->name);
        printf("  type=%s dims=[", t->ggml_type<16?tn[t->ggml_type]:"?");
        for (uint32_t d=0; d<t->n_dims; d++) printf("%s%llu", d?",":"", (unsigned long long)t->dims[d]);
        printf("] %.2f MB @+0x%llx\n", t->nbytes/1048576.0, (unsigned long long)t->offset);
    }
}
