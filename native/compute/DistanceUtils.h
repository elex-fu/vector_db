#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>

namespace vectordb {

/**
 * 支持的SIMD指令集
 */
enum class ISA {
    SCALAR,
    SSE4,
    AVX2,
    AVX512,
    NEON
};

/**
 * 距离计算函数类型
 */
using DistanceFunc = float(*)(const float*, const float*, size_t);

/**
 * 批量距离计算函数类型
 */
using BatchDistanceFunc = void(*)(const float*, const float*, size_t, size_t, float*);

/**
 * 运行时检测当前CPU支持的指令集
 */
ISA detectISA();

/**
 * 获取当前CPU指令集名称字符串
 */
const char* getISAName();

/**
 * 获取最优欧氏距离计算函数
 */
DistanceFunc getEuclideanDistanceFunc();

/**
 * 获取最优余弦相似度计算函数
 */
DistanceFunc getCosineDistanceFunc();

/**
 * 获取批量欧氏距离计算函数
 */
BatchDistanceFunc getBatchEuclideanDistanceFunc();

/**
 * 预计算向量的模长
 */
inline float computeNorm(const float* vector, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum += vector[i] * vector[i];
    }
    return sum;
}

} // namespace vectordb
