#include "VectorStore.h"
#include <cmath>
#include <algorithm>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #define PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
#else
    #define PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
#endif

#ifdef __linux__
    #include <sys/mman.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

namespace vectordb {

VectorStore::VectorStore(int dimension, int maxElements)
    : dimension_(dimension), maxElements_(maxElements) {
    if (dimension <= 0) {
        throw std::invalid_argument("Dimension must be positive");
    }
    if (maxElements <= 0) {
        throw std::invalid_argument("MaxElements must be positive");
    }

    vectors_.resize(static_cast<size_t>(maxElements) * dimension);
    ids_.resize(maxElements);
    norms_.resize(maxElements);
}

int VectorStore::add(int id, const float* vector) {
    int index = size_.fetch_add(1, std::memory_order_acq_rel);

    if (index >= maxElements_) {
        size_.fetch_sub(1, std::memory_order_relaxed);
        throw std::runtime_error("VectorStore is full");
    }

    float* dest = &vectors_[static_cast<size_t>(index) * dimension_];
    std::copy(vector, vector + dimension_, dest);
    norms_[index] = computeNorm(vector, dimension_);
    ids_[index] = id;

    return index;
}

int VectorStore::addBatch(const int* ids, const float* vectors, int count) {
    if (count <= 0) return size_.load();

    int startIndex = size_.fetch_add(count, std::memory_order_acq_rel);

    if (startIndex + count > maxElements_) {
        size_.fetch_sub(count, std::memory_order_relaxed);
        throw std::runtime_error("VectorStore capacity exceeded");
    }

    for (int i = 0; i < count; i++) {
        int index = startIndex + i;
        const float* vec = vectors + static_cast<size_t>(i) * dimension_;
        float* dest = &vectors_[static_cast<size_t>(index) * dimension_];

        std::copy(vec, vec + dimension_, dest);
        norms_[index] = computeNorm(vec, dimension_);
        ids_[index] = ids[i];
    }

    return startIndex;
}

void VectorStore::clear() {
    size_.store(0, std::memory_order_release);
    std::fill(vectors_.begin(), vectors_.end(), 0.0f);
    std::fill(ids_.begin(), ids_.end(), -1);
    std::fill(norms_.begin(), norms_.end(), 0.0f);
}

void VectorStore::prefetchVector(int index) const {
    const float* vec = getVector(index);
    for (int i = 0; i < dimension_; i += 16) {
        PREFETCH(&vec[i]);
    }
    PREFETCH(&ids_[index]);
    PREFETCH(&norms_[index]);
}

void VectorStore::prefetchVectors(const int* indices, int count) const {
    for (int i = 0; i < count && i < 8; i++) {
        prefetchVector(indices[i]);
    }
}

float VectorStore::computeNorm(const float* vector, int dimension) {
    float sum = 0.0f;
    for (int i = 0; i < dimension; i++) {
        sum += vector[i] * vector[i];
    }
    return sum;  // Return squared norm to avoid sqrt overhead
}

bool VectorStore::enableHugePages() {
#ifdef __linux__
    if (usingHugePages_) return true;

    size_t vectorSize = static_cast<size_t>(maxElements_) * dimension_ * sizeof(float);
    size_t idsSize = static_cast<size_t>(maxElements_) * sizeof(int32_t);
    size_t normsSize = static_cast<size_t>(maxElements_) * sizeof(float);
    hugePageSize_ = ((vectorSize + idsSize + normsSize + (2 * 1024 * 1024) - 1) / (2 * 1024 * 1024)) * (2 * 1024 * 1024);

    hugePageMemory_ = mmap(nullptr, hugePageSize_,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                           -1, 0);

    if (hugePageMemory_ == MAP_FAILED) {
        hugePageMemory_ = nullptr;
        return false;
    }

    usingHugePages_ = true;
    return true;
#else
    return false;
#endif
}

} // namespace vectordb
