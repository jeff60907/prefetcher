#ifndef TRANSPOSE_IMPL
#define TRANSPOSE_IMPL

void naive_transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x++)
        for (int y = 0; y < h; y++)
            *(dst + x * h + y) = *(src + y * w + x);
}

void sse_transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x += 4) {
        for (int y = 0; y < h; y += 4) {
            __m128i I0 = _mm_loadu_si128((__m128i *)(src + (y + 0) * w + x));
            __m128i I1 = _mm_loadu_si128((__m128i *)(src + (y + 1) * w + x));
            __m128i I2 = _mm_loadu_si128((__m128i *)(src + (y + 2) * w + x));
            __m128i I3 = _mm_loadu_si128((__m128i *)(src + (y + 3) * w + x));
            __m128i T0 = _mm_unpacklo_epi32(I0, I1);
            __m128i T1 = _mm_unpacklo_epi32(I2, I3);
            __m128i T2 = _mm_unpackhi_epi32(I0, I1);
            __m128i T3 = _mm_unpackhi_epi32(I2, I3);
            I0 = _mm_unpacklo_epi64(T0, T1);
            I1 = _mm_unpackhi_epi64(T0, T1);
            I2 = _mm_unpacklo_epi64(T2, T3);
            I3 = _mm_unpackhi_epi64(T2, T3);
            _mm_storeu_si128((__m128i *)(dst + ((x + 0) * h) + y), I0);
            _mm_storeu_si128((__m128i *)(dst + ((x + 1) * h) + y), I1);
            _mm_storeu_si128((__m128i *)(dst + ((x + 2) * h) + y), I2);
            _mm_storeu_si128((__m128i *)(dst + ((x + 3) * h) + y), I3);
        }
    }
}

void sse_prefetch_transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x += 4) {
        for (int y = 0; y < h; y += 4) {
#define PFDIST  8
            _mm_prefetch(src+(y + PFDIST + 0) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 1) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 2) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 3) *w + x, _MM_HINT_T1);

            __m128i I0 = _mm_loadu_si128 ((__m128i *)(src + (y + 0) * w + x));
            __m128i I1 = _mm_loadu_si128 ((__m128i *)(src + (y + 1) * w + x));
            __m128i I2 = _mm_loadu_si128 ((__m128i *)(src + (y + 2) * w + x));
            __m128i I3 = _mm_loadu_si128 ((__m128i *)(src + (y + 3) * w + x));
            __m128i T0 = _mm_unpacklo_epi32(I0, I1);
            __m128i T1 = _mm_unpacklo_epi32(I2, I3);
            __m128i T2 = _mm_unpackhi_epi32(I0, I1);
            __m128i T3 = _mm_unpackhi_epi32(I2, I3);
            I0 = _mm_unpacklo_epi64(T0, T1);
            I1 = _mm_unpackhi_epi64(T0, T1);
            I2 = _mm_unpacklo_epi64(T2, T3);
            I3 = _mm_unpackhi_epi64(T2, T3);
            _mm_storeu_si128((__m128i *)(dst + ((x + 0) * h) + y), I0);
            _mm_storeu_si128((__m128i *)(dst + ((x + 1) * h) + y), I1);
            _mm_storeu_si128((__m128i *)(dst + ((x + 2) * h) + y), I2);
            _mm_storeu_si128((__m128i *)(dst + ((x + 3) * h) + y), I3);
        }
    }
}

void avx_transpose(int *src, int *dst, int w, int h)
{
    for ( int x = 0; x < w; x += 8 ) {
        for ( int y = 0; y < h; y += 8 ) {
            __m256i ymm0 = _mm256_loadu_si256((__m256i *) (src + (y + 0) * w + x));
            __m256i ymm1 = _mm256_loadu_si256((__m256i *) (src + (y + 1) * w + x));
            __m256i ymm2 = _mm256_loadu_si256((__m256i *) (src + (y + 2) * w + x));
            __m256i ymm3 = _mm256_loadu_si256((__m256i *) (src + (y + 3) * w + x));
            __m256i ymm4 = _mm256_loadu_si256((__m256i *) (src + (y + 4) * w + x));
            __m256i ymm5 = _mm256_loadu_si256((__m256i *) (src + (y + 5) * w + x));
            __m256i ymm6 = _mm256_loadu_si256((__m256i *) (src + (y + 6) * w + x));
            __m256i ymm7 = _mm256_loadu_si256((__m256i *) (src + (y + 7) * w + x));

            __m256  T0 = _mm256_unpacklo_ps((__m256 )ymm0, (__m256 )ymm1);
            __m256  T1 = _mm256_unpackhi_ps((__m256 )ymm0, (__m256 )ymm1);
            __m256  T2 = _mm256_unpacklo_ps((__m256 )ymm2, (__m256 )ymm3);
            __m256  T3 = _mm256_unpackhi_ps((__m256 )ymm2, (__m256 )ymm3);
            __m256  T4 = _mm256_unpacklo_ps((__m256 )ymm4, (__m256 )ymm5);
            __m256  T5 = _mm256_unpackhi_ps((__m256 )ymm4, (__m256 )ymm5);
            __m256  T6 = _mm256_unpacklo_ps((__m256 )ymm6, (__m256 )ymm7);
            __m256  T7 = _mm256_unpackhi_ps((__m256 )ymm6, (__m256 )ymm7);

            __m256 ymm10 = _mm256_unpacklo_ps(T0, T2);
            __m256 ymm11 = _mm256_unpackhi_ps(T0, T2);
            __m256 ymm12 = _mm256_unpacklo_ps(T1, T3);
            __m256 ymm13 = _mm256_unpackhi_ps(T1, T3);
            __m256 ymm14 = _mm256_unpacklo_ps(T4, T6);
            __m256 ymm15 = _mm256_unpackhi_ps(T4, T6);
            __m256 ymm16 = _mm256_unpacklo_ps(T5, T7);
            __m256 ymm17 = _mm256_unpackhi_ps(T5, T7);

            T0 = _mm256_permute2f128_ps(ymm10, ymm14, 0x20);
            T1 = _mm256_permute2f128_ps(ymm11, ymm15, 0x20);
            T2 = _mm256_permute2f128_ps(ymm12, ymm16, 0x20);
            T3 = _mm256_permute2f128_ps(ymm13, ymm17, 0x20);
            T4 = _mm256_permute2f128_ps(ymm10, ymm14, 0x31);
            T5 = _mm256_permute2f128_ps(ymm11, ymm15, 0x31);
            T6 = _mm256_permute2f128_ps(ymm12, ymm16, 0x31);
            T7 = _mm256_permute2f128_ps(ymm13, ymm17, 0x31);

            _mm256_storeu_ps((float *)(dst + ((x + 0) * h) + y), T0);
            _mm256_storeu_ps((float *)(dst + ((x + 1) * h) + y), T1);
            _mm256_storeu_ps((float *)(dst + ((x + 2) * h) + y), T2);
            _mm256_storeu_ps((float *)(dst + ((x + 3) * h) + y), T3);
            _mm256_storeu_ps((float *)(dst + ((x + 4) * h) + y), T4);
            _mm256_storeu_ps((float *)(dst + ((x + 5) * h) + y), T5);
            _mm256_storeu_ps((float *)(dst + ((x + 6) * h) + y), T6);
            _mm256_storeu_ps((float *)(dst + ((x + 7) * h) + y), T7);
        }
    }
}

void avx_prefetch_transpose(int *src, int *dst, int w, int h)
{
    for ( int x = 0; x < w; x += 8 ) {
        for ( int y = 0; y < h; y += 8 ) {
#define AVXPFDIST  8
            _mm_prefetch(src+(y + AVXPFDIST + 0) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + AVXPFDIST + 1) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + AVXPFDIST + 2) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + AVXPFDIST + 3) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + AVXPFDIST + 4) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + AVXPFDIST + 5) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + AVXPFDIST + 6) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + AVXPFDIST + 7) *w + x, _MM_HINT_T1);

            __m256i ymm0 = _mm256_loadu_si256((__m256i *) (src + (y + 0) * w + x));
            __m256i ymm1 = _mm256_loadu_si256((__m256i *) (src + (y + 1) * w + x));
            __m256i ymm2 = _mm256_loadu_si256((__m256i *) (src + (y + 2) * w + x));
            __m256i ymm3 = _mm256_loadu_si256((__m256i *) (src + (y + 3) * w + x));
            __m256i ymm4 = _mm256_loadu_si256((__m256i *) (src + (y + 4) * w + x));
            __m256i ymm5 = _mm256_loadu_si256((__m256i *) (src + (y + 5) * w + x));
            __m256i ymm6 = _mm256_loadu_si256((__m256i *) (src + (y + 6) * w + x));
            __m256i ymm7 = _mm256_loadu_si256((__m256i *) (src + (y + 7) * w + x));

            __m256  T0 = _mm256_unpacklo_ps((__m256 )ymm0, (__m256 )ymm1);
            __m256  T1 = _mm256_unpackhi_ps((__m256 )ymm0, (__m256 )ymm1);
            __m256  T2 = _mm256_unpacklo_ps((__m256 )ymm2, (__m256 )ymm3);
            __m256  T3 = _mm256_unpackhi_ps((__m256 )ymm2, (__m256 )ymm3);
            __m256  T4 = _mm256_unpacklo_ps((__m256 )ymm4, (__m256 )ymm5);
            __m256  T5 = _mm256_unpackhi_ps((__m256 )ymm4, (__m256 )ymm5);
            __m256  T6 = _mm256_unpacklo_ps((__m256 )ymm6, (__m256 )ymm7);
            __m256  T7 = _mm256_unpackhi_ps((__m256 )ymm6, (__m256 )ymm7);

            __m256 ymm10 = _mm256_unpacklo_ps(T0, T2);
            __m256 ymm11 = _mm256_unpackhi_ps(T0, T2);
            __m256 ymm12 = _mm256_unpacklo_ps(T1, T3);
            __m256 ymm13 = _mm256_unpackhi_ps(T1, T3);
            __m256 ymm14 = _mm256_unpacklo_ps(T4, T6);
            __m256 ymm15 = _mm256_unpackhi_ps(T4, T6);
            __m256 ymm16 = _mm256_unpacklo_ps(T5, T7);
            __m256 ymm17 = _mm256_unpackhi_ps(T5, T7);

            T0 = _mm256_permute2f128_ps(ymm10, ymm14, 0x20);
            T1 = _mm256_permute2f128_ps(ymm11, ymm15, 0x20);
            T2 = _mm256_permute2f128_ps(ymm12, ymm16, 0x20);
            T3 = _mm256_permute2f128_ps(ymm13, ymm17, 0x20);
            T4 = _mm256_permute2f128_ps(ymm10, ymm14, 0x31);
            T5 = _mm256_permute2f128_ps(ymm11, ymm15, 0x31);
            T6 = _mm256_permute2f128_ps(ymm12, ymm16, 0x31);
            T7 = _mm256_permute2f128_ps(ymm13, ymm17, 0x31);

            _mm256_storeu_ps((float *)(dst + ((x + 0) * h) + y), T0);
            _mm256_storeu_ps((float *)(dst + ((x + 1) * h) + y), T1);
            _mm256_storeu_ps((float *)(dst + ((x + 2) * h) + y), T2);
            _mm256_storeu_ps((float *)(dst + ((x + 3) * h) + y), T3);
            _mm256_storeu_ps((float *)(dst + ((x + 4) * h) + y), T4);
            _mm256_storeu_ps((float *)(dst + ((x + 5) * h) + y), T5);
            _mm256_storeu_ps((float *)(dst + ((x + 6) * h) + y), T6);
            _mm256_storeu_ps((float *)(dst + ((x + 7) * h) + y), T7);
        }
    }
}


#endif /* TRANSPOSE_IMPL */
