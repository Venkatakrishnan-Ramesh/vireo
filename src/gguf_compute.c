/* Dequantization + quantized mat-vec for GGUF weights. Weights stay quantized
 * on disk (mmap); we dequantize on the fly inside the dot product so nothing
 * large is materialized. Supports F32, F16, Q8_0 — enough to run a Q8_0 model.
 */
#include <stdint.h>
#include <string.h>
#include "gguf_compute.h"

float gguf_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;                 /* +/- 0 */
        else { /* subnormal */
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000 | (mant << 13);    /* inf/nan */
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

/* Q8_0 block: fp16 scale d, then 32 int8 quants. x = qs * d. */
typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;

void gguf_dequant_row(uint32_t type, const void *src, float *dst, int n) {
    if (type == GGML_F32) { memcpy(dst, src, (size_t)n * 4); return; }
    if (type == GGML_F16) {
        const uint16_t *h = (const uint16_t *)src;
        for (int i = 0; i < n; i++) dst[i] = gguf_f16_to_f32(h[i]);
        return;
    }
    if (type == GGML_Q8_0) {
        const block_q8_0 *b = (const block_q8_0 *)src;
        int nb = n / 32;
        for (int j = 0; j < nb; j++) {
            float d = gguf_f16_to_f32(b[j].d);
            for (int i = 0; i < 32; i++) dst[j * 32 + i] = b[j].qs[i] * d;
        }
        return;
    }
}

/* y[o] = dot(x[0..n], dequant(row o of w)), for o in 0..d. w rows are
 * contiguous (n elements each). n must be a multiple of 32 for Q8_0. */
void gguf_matmul(float *y, const float *x, const gguf_tensor *w, int n, int d) {
    uint32_t type = w->ggml_type;
    if (type == GGML_Q8_0) {
        int nb = n / 32;
        const block_q8_0 *base = (const block_q8_0 *)w->data;
        for (int o = 0; o < d; o++) {
            const block_q8_0 *row = base + (size_t)o * nb;
            float acc = 0.0f;
            for (int j = 0; j < nb; j++) {
                float dscale = gguf_f16_to_f32(row[j].d);
                const float *xb = x + j * 32;
                float s = 0.0f;
                for (int i = 0; i < 32; i++) s += row[j].qs[i] * xb[i];
                acc += s * dscale;
            }
            y[o] = acc;
        }
        return;
    }
    if (type == GGML_F32) {
        const float *W = (const float *)w->data;
        for (int o = 0; o < d; o++) { const float *r = W + (size_t)o*n; float a=0; for (int i=0;i<n;i++) a+=r[i]*x[i]; y[o]=a; }
        return;
    }
    if (type == GGML_F16) {
        const uint16_t *W = (const uint16_t *)w->data;
        for (int o = 0; o < d; o++) { const uint16_t *r = W + (size_t)o*n; float a=0; for (int i=0;i<n;i++) a+=gguf_f16_to_f32(r[i])*x[i]; y[o]=a; }
        return;
    }
}
