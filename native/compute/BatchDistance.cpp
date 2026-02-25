#include "BatchDistance.h"
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
    #define USE_ACCELERATE 1
#elif defined(__linux__)
    extern "C" {
        #include <cblas.h>
    }
    #define USE_OPENBLAS 1
#else
    #define USE_FALLBACK 1
#endif

namespace vectordb {

void matrixMultiply(const float* A, const float* B, float* C,
                    size_t m, size_t n, size_t k) {
#if defined(USE_ACCELERATE)
    // C = A * B^T, dimensions: [m][k] * [n][k]^T = [m][n]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                1.0f, A, static_cast<int>(k),
                B, static_cast<int>(k),
                0.0f, C, static_cast<int>(n));
#elif defined(USE_OPENBLAS)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                1.0f, A, static_cast<int>(k),
                B, static_cast<int>(k),
                0.0f, C, static_cast<int>(n));
#else
    // Fallback naive implementation
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; l++) {
                sum += A[i * k + l] * B[j * k + l];
            }
            C[i * n + j] = sum;
        }
    }
#endif
}

void computeRowNormsSquared(const float* matrix, size_t rows, size_t cols, float* norms) {
#if defined(USE_ACCELERATE)
    for (size_t i = 0; i < rows; i++) {
        norms[i] = cblas_sdot(static_cast<int>(cols),
                              matrix + i * cols, 1,
                              matrix + i * cols, 1);
    }
#elif defined(USE_OPENBLAS)
    for (size_t i = 0; i < rows; i++) {
        norms[i] = cblas_sdot(static_cast<int>(cols),
                              matrix + i * cols, 1,
                              matrix + i * cols, 1);
    }
#else
    for (size_t i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            float val = matrix[i * cols + j];
            sum += val * val;
        }
        norms[i] = sum;
    }
#endif
}

void batchEuclideanDistance(const float* query, const float* vectors,
                            size_t n, size_t dim, float* distances) {
    // Compute ||query||^2
    float queryNormSq = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        queryNormSq += query[i] * query[i];
    }

    // Compute ||vector_i||^2 for each vector
    std::vector<float> vectorNorms(n);
    computeRowNormsSquared(vectors, n, dim, vectorNorms.data());

    // Compute 2 * query^T * vectors using matrix multiplication
    // query is [1][dim], vectors is [n][dim], result is [n]
    std::vector<float> dotProducts(n);
#if defined(USE_ACCELERATE)
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                static_cast<int>(n), static_cast<int>(dim),
                2.0f, vectors, static_cast<int>(dim),
                query, 1,
                0.0f, dotProducts.data(), 1);
#elif defined(USE_OPENBLAS)
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                static_cast<int>(n), static_cast<int>(dim),
                2.0f, vectors, static_cast<int>(dim),
                query, 1,
                0.0f, dotProducts.data(), 1);
#else
    for (size_t i = 0; i < n; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            sum += vectors[i * dim + j] * query[j];
        }
        dotProducts[i] = 2.0f * sum;
    }
#endif

    // Compute Euclidean distance: ||q - v||^2 = ||q||^2 + ||v||^2 - 2*q^T*v
    for (size_t i = 0; i < n; i++) {
        distances[i] = queryNormSq + vectorNorms[i] - dotProducts[i];
        // Handle numerical errors (can be slightly negative)
        if (distances[i] < 0.0f && distances[i] > -1e-6f) {
            distances[i] = 0.0f;
        }
    }
}

void batchEuclideanDistanceMultiQuery(const float* queries, const float* vectors,
                                      size_t nQueries, size_t nVectors, size_t dim,
                                      float* distances) {
    // Compute ||query_i||^2 for each query
    std::vector<float> queryNorms(nQueries);
    computeRowNormsSquared(queries, nQueries, dim, queryNorms.data());

    // Compute ||vector_j||^2 for each vector
    std::vector<float> vectorNorms(nVectors);
    computeRowNormsSquared(vectors, nVectors, dim, vectorNorms.data());

    // Compute 2 * queries * vectors^T using matrix multiplication
    // queries is [nQueries][dim], vectors is [nVectors][dim]
    // result is [nQueries][nVectors]
    matrixMultiply(queries, vectors, distances, nQueries, nVectors, dim);

    // Scale by 2 and compute final distances
    for (size_t i = 0; i < nQueries; i++) {
        for (size_t j = 0; j < nVectors; j++) {
            float dist = queryNorms[i] + vectorNorms[j] - 2.0f * distances[i * nVectors + j];
            if (dist < 0.0f && dist > -1e-6f) {
                dist = 0.0f;
            }
            distances[i * nVectors + j] = dist;
        }
    }
}

} // namespace vectordb
