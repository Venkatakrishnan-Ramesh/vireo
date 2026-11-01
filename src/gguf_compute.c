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

/* ---- K-quants (super-block = 256 elements) ---- */
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K; /* 144 B */
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; } block_q6_K;  /* 210 B */

/* unpack a 6-bit scale and 6-bit min for sub-block j from the packed 12 bytes */
static inline void q4k_scale_min(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}
static void dequant_row_q4_K(const block_q4_K *x, float *y, int k) {
    int nb = k / 256;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        float d = gguf_f16_to_f32(x[i].d), mn = gguf_f16_to_f32(x[i].dmin);
        int is = 0; uint8_t sc, m;
        for (int j = 0; j < 256; j += 64) {
            q4k_scale_min(is + 0, x[i].scales, &sc, &m); float d1 = d * sc, m1 = mn * m;
            q4k_scale_min(is + 1, x[i].scales, &sc, &m); float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4)  - m2;
            q += 32; is += 2;
        }
    }
}
typedef struct { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; } block_q5_0; /* 22 B, 32 elems */
static void dequant_row_q5_0(const block_q5_0 *x, float *y, int k) {
    int nb = k / 32;
    for (int i = 0; i < nb; i++) {
        float d = gguf_f16_to_f32(x[i].d);
        uint32_t qh; memcpy(&qh, x[i].qh, 4);
        for (int j = 0; j < 16; j++) {
            uint8_t xh0 = ((qh >> (j +  0)) << 4) & 0x10;
            uint8_t xh1 = ((qh >> (j + 12))     ) & 0x10;
            int x0 = ((x[i].qs[j] & 0x0F) | xh0) - 16;
            int x1 = ((x[i].qs[j] >>   4) | xh1) - 16;
            y[i*32 + j]      = x0 * d;
            y[i*32 + j + 16] = x1 * d;
        }
    }
}
static void dequant_row_q6_K(const block_q6_K *x, float *y, int k) {
    int nb = k / 256;
    for (int i = 0; i < nb; i++) {
        float d = gguf_f16_to_f32(x[i].d);
        const uint8_t *ql = x[i].ql; const uint8_t *qh = x[i].qh; const int8_t *sc = x[i].scales;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l]      >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l]      = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

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
    if (type == GGML_Q5_0) { dequant_row_q5_0((const block_q5_0 *)src, dst, n); return; }
    if (type == GGML_Q4_K) { dequant_row_q4_K((const block_q4_K *)src, dst, n); return; }
    if (type == GGML_Q6_K) { dequant_row_q6_K((const block_q6_K *)src, dst, n); return; }
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
static inline float dot_f32(const float *a, const float *b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    float s = hsum256(acc);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
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
    {
        int be, bb;
        if (type != GGML_Q8_0 && type != GGML_F32 && type != GGML_F16 &&
            gguf_type_block(type, &be, &bb) == 0) {
        size_t row_bytes = (size_t)(n / be) * bb;
        const uint8_t *data = (const uint8_t *)w->data;
        #pragma omp parallel for schedule(static) if((long)n*(long)d > (1L<<20))
        for (int o = 0; o < d; o++) {
            float tmp[8192];
            gguf_dequant_row(type, data + (size_t)o * row_bytes, tmp, n);
#if defined(__AVX2__)
            y[o] = dot_f32(tmp, x, n);
#else
            float a = 0; for (int i = 0; i < n; i++) a += tmp[i] * x[i]; y[o] = a;
#endif
        }
        return;
        }
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
