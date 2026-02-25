#include "DistanceUtils.h"
#include <cmath>
#include <cstdint>

namespace vectordb {

float euclideanDistanceScalar(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float cosineDistanceScalar(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }
    return 1.0f - dot;
}

float dotProductScalar(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }
    return -dot;
}

static void batchEuclideanDistanceScalar(const float* query, const float* vectors,
                                         size_t n, size_t dim, float* distances) {
    for (size_t i = 0; i < n; i++) {
        const float* vec = vectors + i * dim;
        distances[i] = euclideanDistanceScalar(query, vec, dim);
    }
}

} // namespace vectordb
