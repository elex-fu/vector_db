#include "DistanceUtils.h"
#include <atomic>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
    #include <cpuid.h>
    #include <immintrin.h>
#endif

namespace vectordb {

static std::atomic<ISA> g_isa{ISA::SCALAR};
static std::atomic<bool> g_initialized{false};

static void initSIMD() {
    if (g_initialized.load(std::memory_order_acquire)) return;
    g_isa.store(detectISA(), std::memory_order_release);
    g_initialized.store(true, std::memory_order_release);
}

ISA detectISA() {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int eax, ebx, ecx, edx;

    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        if (ebx & bit_AVX512F) {
            if (ebx & bit_AVX512DQ) {
                return ISA::AVX512;
            }
        }
    }

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        if (ecx & bit_AVX2) {
            return ISA::AVX2;
        }
        if (ecx & bit_SSE4_2) {
            return ISA::SSE4;
        }
    }

    return ISA::SCALAR;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return ISA::NEON;
#else
    return ISA::SCALAR;
#endif
}

const char* getISAName() {
    initSIMD();
    switch (g_isa.load(std::memory_order_acquire)) {
        case ISA::AVX512: return "AVX-512";
        case ISA::AVX2:   return "AVX2";
        case ISA::SSE4:   return "SSE4.2";
        case ISA::NEON:   return "NEON";
        default:          return "Scalar";
    }
}

extern float euclideanDistanceAVX2(const float*, const float*, size_t);
extern float cosineDistanceAVX2(const float*, const float*, size_t);
extern float dotProductAVX2(const float*, const float*, size_t);

static float euclideanDistanceScalar(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

DistanceFunc getEuclideanDistanceFunc() {
    initSIMD();
#if defined(__x86_64__) || defined(_M_X64)
    ISA isa = g_isa.load(std::memory_order_acquire);
    if (isa == ISA::AVX2 || isa == ISA::AVX512 || isa == ISA::SSE4) {
        return euclideanDistanceAVX2;
    }
#endif
    return euclideanDistanceScalar;
}

DistanceFunc getCosineDistanceFunc() {
    initSIMD();
#if defined(__x86_64__) || defined(_M_X64)
    ISA isa = g_isa.load(std::memory_order_acquire);
    if (isa == ISA::AVX2 || isa == ISA::AVX512 || isa == ISA::SSE4) {
        return cosineDistanceAVX2;
    }
#endif
    return [](const float* a, const float* b, size_t dim) {
        float dot = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            dot += a[i] * b[i];
        }
        return 1.0f - dot;
    };
}

extern void batchEuclideanDistanceAVX2(const float*, const float*, size_t, size_t, float*);

static void batchEuclideanDistanceScalar(const float* query, const float* vectors,
                                         size_t n, size_t dim, float* distances) {
    for (size_t i = 0; i < n; i++) {
        const float* vec = vectors + i * dim;
        distances[i] = euclideanDistanceScalar(query, vec, dim);
    }
}

BatchDistanceFunc getBatchEuclideanDistanceFunc() {
    initSIMD();
#if defined(__x86_64__) || defined(_M_X64)
    ISA isa = g_isa.load(std::memory_order_acquire);
    if (isa == ISA::AVX2 || isa == ISA::AVX512 || isa == ISA::SSE4) {
        return batchEuclideanDistanceAVX2;
    }
#endif
    return batchEuclideanDistanceScalar;
}

} // namespace vectordb
