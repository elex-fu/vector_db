#pragma once
#include <cstddef>
#include <cstdint>

namespace vectordb {

/**
 * SIMD 优化的 ADC (Asymmetric Distance Computation) 距离计算
 * 用于 Product Quantization 索引的快速距离计算
 */

/**
 * 基础 ADC 距离计算 (无 SIMD)
 * 计算查询向量与量化编码之间的距离
 * @param distanceTable 预计算的距离表 [pqM][nCentroids]
 * @param codes 量化编码 [pqM]
 * @param pqM 子空间数量
 * @param nCentroids 每个子空间的聚类中心数
 * @return 距离值
 */
float adcDistanceScalar(const float* distanceTable, const uint8_t* codes, int pqM, int nCentroids);

/**
 * AVX2 优化的 ADC 距离计算
 * 使用 AVX2 指令集并行计算 8 个子空间的距离累加
 * @param distanceTable 预计算的距离表 [pqM][nCentroids]
 * @param codes 量化编码 [pqM]
 * @param pqM 子空间数量 (必须是 8 的倍数，如果不是会退化到标量)
 * @param nCentroids 每个子空间的聚类中心数
 * @return 距离值
 */
float adcDistanceAVX2(const float* distanceTable, const uint8_t* codes, int pqM, int nCentroids);

/**
 * 批量 ADC 距离计算 - AVX2 优化
 * 同时计算多个查询的距离，最大化 SIMD 利用率
 * @param distanceTable 预计算的距离表 [pqM][nCentroids]
 * @param codes 多个量化编码 [nCodes][pqM]
 * @param nCodes 编码数量
 * @param pqM 子空间数量
 * @param nCentroids 每个子空间的聚类中心数
 * @param distances 输出距离 [nCodes]
 */
void adcDistanceBatchAVX2(const float* distanceTable, const uint8_t* codes,
                          int nCodes, int pqM, int nCentroids, float* distances);

/**
 * 获取最优 ADC 距离计算函数
 */
using ADCDistanceFunc = float(*)(const float*, const uint8_t*, int, int);
ADCDistanceFunc getADCDistanceFunc();

/**
 * 获取批量 ADC 距离计算函数
 */
using ADCDistanceBatchFunc = void(*)(const float*, const uint8_t*, int, int, int, float*);
ADCDistanceBatchFunc getADCDistanceBatchFunc();

} // namespace vectordb
