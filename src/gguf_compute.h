#ifndef VIREO_GGUF_COMPUTE_H
#define VIREO_GGUF_COMPUTE_H
#include <stdint.h>
#include "gguf.h"
#ifdef __cplusplus
extern "C" {
#endif
float gguf_f16_to_f32(uint16_t h);
void  gguf_dequant_row(uint32_t type, const void *src, float *dst, int n);
void  gguf_matmul(float *y, const float *x, const gguf_tensor *w, int n, int d);
#ifdef __cplusplus
}
#endif
#endif
