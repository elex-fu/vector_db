#include "DistanceUtils.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <cstdint>

namespace vectordb {

float euclideanDistanceAVX2(const float* a, const float* b, size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    __m128 sumLow = _mm256_castps256_ps128(sum);
    __m128 sumHigh = _mm256_extractf128_ps(sum, 1);
    sumLow = _mm_add_ps(sumLow, sumHigh);
    sumLow = _mm_hadd_ps(sumLow, sumLow);
    sumLow = _mm_hadd_ps(sumLow, sumLow);
    float total = _mm_cvtss_f32(sumLow);

    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }

    return total;
}

float cosineDistanceAVX2(const float* a, const float* b, size_t dim) {
    __m256 dot = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        dot = _mm256_fmadd_ps(va, vb, dot);
    }

    __m128 dotLow = _mm256_castps256_ps128(dot);
    __m128 dotHigh = _mm256_extractf128_ps(dot, 1);
    dotLow = _mm_add_ps(dotLow, dotHigh);
    dotLow = _mm_hadd_ps(dotLow, dotLow);
    dotLow = _mm_hadd_ps(dotLow, dotLow);
    float dotProduct = _mm_cvtss_f32(dotLow);

    for (; i < dim; i++) {
        dotProduct += a[i] * b[i];
    }

    return 1.0f - dotProduct;
}

void batchEuclideanDistanceAVX2(const float* query, const float* vectors,
                                       size_t n, size_t dim, float* distances) {
    const size_t simdWidth = 8;

    for (size_t i = 0; i < n; i++) {
        const float* vec = vectors + i * dim;
        __m256 sumVec = _mm256_setzero_ps();

        size_t j = 0;
        for (; j + simdWidth <= dim; j += simdWidth) {
            __m256 a = _mm256_loadu_ps(query + j);
            __m256 b = _mm256_loadu_ps(vec + j);
            __m256 diff = _mm256_sub_ps(a, b);
            sumVec = _mm256_fmadd_ps(diff, diff, sumVec);
        }

        float sum = 0.0f;
        float temp[8];
        _mm256_storeu_ps(temp, sumVec);
        for (int k = 0; k < 8; k++) {
            sum += temp[k];
        }

        for (; j < dim; j++) {
            float diff = query[j] - vec[j];
            sum += diff * diff;
        }

        distances[i] = sum;
    }
}

} // namespace vectordb

#endif
