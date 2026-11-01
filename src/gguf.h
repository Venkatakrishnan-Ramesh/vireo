/* Minimal GGUF v2/v3 reader: mmap a model file, parse metadata + the tensor
 * table, hand back tensor pointers into the mmap. No copying, no allocation of
 * weight data — tensors stay on disk and are paged by the OS. Dequant lives in
 * gguf_dequant.*; this file only locates and describes tensors. */
#ifndef VIREO_GGUF_H
#define VIREO_GGUF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF metadata value types (wire enum). */
enum { GGUF_U8=0, GGUF_I8=1, GGUF_U16=2, GGUF_I16=3, GGUF_U32=4, GGUF_I32=5,
       GGUF_F32=6, GGUF_BOOL=7, GGUF_STRING=8, GGUF_ARRAY=9, GGUF_U64=10,
       GGUF_I64=11, GGUF_F64=12 };

/* ggml tensor types we care about (wire enum subset). */
enum { GGML_F32=0, GGML_F16=1, GGML_Q4_0=2, GGML_Q4_1=3, GGML_Q5_0=6,
       GGML_Q5_1=7, GGML_Q8_0=8, GGML_Q8_1=9, GGML_Q2_K=10, GGML_Q3_K=11,
       GGML_Q4_K=12, GGML_Q5_K=13, GGML_Q6_K=14, GGML_Q8_K=15 };

typedef struct { const char *ptr; uint64_t len; } gguf_str;

typedef struct {
    gguf_str key;
    uint32_t type;
    union { uint64_t u; int64_t i; double f; } val;  /* scalars           */
    gguf_str str;                                    /* if GGUF_STRING    */
    uint32_t arr_type; uint64_t arr_len;             /* if GGUF_ARRAY     */
    const uint8_t *arr_data;                         /* first array item  */
} gguf_kv;

typedef struct {
    gguf_str name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t ggml_type;
    uint64_t offset;       /* from data section start */
    const void *data;      /* absolute pointer into the mmap */
    uint64_t n_elems;
    uint64_t nbytes;
} gguf_tensor;

typedef struct {
    int fd; void *map; size_t size;
    uint32_t version;
    uint64_t n_tensors, n_kv;
    gguf_kv *kv;
    gguf_tensor *tensors;
    const uint8_t *data;   /* tensor data section start */
    uint64_t alignment;
} gguf_ctx;

gguf_ctx *gguf_open(const char *path);
void gguf_close(gguf_ctx *g);

const gguf_kv *gguf_find(const gguf_ctx *g, const char *key);
/* Convenience scalar getters; return 0 on success. Integer getter accepts any
 * integer metadata type; float getter accepts F32/F64. */
int gguf_get_u64(const gguf_ctx *g, const char *key, uint64_t *out);
int gguf_get_f32(const gguf_ctx *g, const char *key, float *out);
const char *gguf_get_str(const gguf_ctx *g, const char *key, uint64_t *len);

const gguf_tensor *gguf_tensor_by_name(const gguf_ctx *g, const char *name);

/* elements per block and bytes per block for a ggml type (block=1 for F32/F16). */
int gguf_type_block(uint32_t ggml_type, int *block_elems, int *block_bytes);

void gguf_dump(const gguf_ctx *g);

#ifdef __cplusplus
}
#endif
#endif
