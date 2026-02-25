#pragma once
#include <vector>
#include <cstdint>
#include <atomic>
#include <stdexcept>

namespace vectordb {

/**
 * 向量存储类
 * 使用Structure of Arrays (SoA)布局优化缓存性能
 */
class VectorStore {
public:
    VectorStore(int dimension, int maxElements);

    // 添加向量，返回索引
    int add(int id, const float* vector);

    // 批量添加
    int addBatch(const int* ids, const float* vectors, int count);

    // 获取向量
    const float* getVector(int index) const {
        if (index < 0 || index >= size_.load()) {
            return nullptr;
        }
        return vectors_.data() + static_cast<size_t>(index) * dimension_;
    }

    // 获取ID
    int getId(int index) const {
        if (index < 0 || index >= size_.load()) {
            return -1;
        }
        return ids_[index];
    }

    // 获取模长
    float getNorm(int index) const {
        if (index < 0 || index >= size_.load()) {
            return 0.0f;
        }
        return norms_[index];
    }

    // 预取向量到缓存
    void prefetchVector(int index) const;
    void prefetchVectors(const int* indices, int count) const;

    // 清空存储
    void clear();

    // 基本信息
    int size() const { return size_.load(); }
    int dimension() const { return dimension_; }
    int capacity() const { return maxElements_; }

    // HugePages支持
    bool enableHugePages();

private:
    int dimension_;
    int maxElements_;
    std::atomic<int> size_{0};

    // SoA布局存储
    std::vector<float> vectors_;  // [maxElements][dimension]
    std::vector<int32_t> ids_;    // [maxElements]
    std::vector<float> norms_;    // [maxElements] 预计算模长

    // HugePages支持
    void* hugePageMemory_ = nullptr;
    size_t hugePageSize_ = 0;
    bool usingHugePages_ = false;

    // 工具方法
    static float computeNorm(const float* vector, int dimension);
};

} // namespace vectordb
