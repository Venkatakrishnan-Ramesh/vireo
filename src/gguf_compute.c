/* Dequantization + quantized mat-vec for GGUF weights. Weights stay quantized
 * on disk (mmap); we dequantize on the fly inside the dot product so nothing
 * large is materialized. AVX2 + FMA fast path for Q8_0 with a scalar fallback;
 * OpenMP across output rows. Supports F32, F16, Q8_0.
 *
 * Determinism: each output row is computed by a single thread in block order,
 * so results are run-to-run identical — vireo-gguf's "all == unbounded" KV
 * invariant is preserved. */
#include <stdint.h>
#include <string.h>
#include "gguf_compute.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

float gguf_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000 | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

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
    }
}

#if defined(__AVX2__)
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    __m128 s  = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, s);
    s  = _mm_add_ss(s, sh);
    return _mm_cvtss_f32(s);
}
/* dot of one Q8_0 block (32 int8 quants) with 32 floats, unscaled. */
static inline float dot_block_q8_0(const int8_t *qs, const float *x) {
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < 4; k++) {
        __m128i qi8  = _mm_loadl_epi64((const __m128i *)(qs + 8 * k)); /* 8 int8 */
        __m256i qi32 = _mm256_cvtepi8_epi32(qi8);
        __m256  qf   = _mm256_cvtepi32_ps(qi32);
        __m256  xf   = _mm256_loadu_ps(x + 8 * k);
        acc = _mm256_fmadd_ps(qf, xf, acc);
    }
    return hsum256(acc);
}
#endif

void gguf_matmul(float *y, const float *x, const gguf_tensor *w, int n, int d) {
    uint32_t type = w->ggml_type;
    if (type == GGML_Q8_0) {
        int nb = n / 32;
        const block_q8_0 *base = (const block_q8_0 *)w->data;
        #pragma omp parallel for schedule(static) if((long)n*(long)d > (1L<<20))
        for (int o = 0; o < d; o++) {
            const block_q8_0 *row = base + (size_t)o * nb;
            float acc = 0.0f;
            for (int j = 0; j < nb; j++) {
                float dscale = gguf_f16_to_f32(row[j].d);
                const float *xb = x + j * 32;
#if defined(__AVX2__)
                acc += dot_block_q8_0(row[j].qs, xb) * dscale;
#else
                float s = 0.0f;
                for (int i = 0; i < 32; i++) s += row[j].qs[i] * xb[i];
                acc += s * dscale;
#endif
            }
            y[o] = acc;
        }
        return;
    }
    if (type == GGML_F32) {
        const float *W = (const float *)w->data;
        #pragma omp parallel for schedule(static) if((long)n*(long)d > (1L<<20))
        for (int o = 0; o < d; o++) { const float *r = W + (size_t)o*n; float a=0; for (int i=0;i<n;i++) a+=r[i]*x[i]; y[o]=a; }
        return;
    }
    if (type == GGML_F16) {
        const uint16_t *W = (const uint16_t *)w->data;
        #pragma omp parallel for schedule(static) if((long)n*(long)d > (1L<<20))
        for (int o = 0; o < d; o++) { const uint16_t *r = W + (size_t)o*n; float a=0; for (int i=0;i<n;i++) a+=gguf_f16_to_f32(r[i])*x[i]; y[o]=a; }
        return;
    }
}
