#pragma once
#include <cstddef>
#include <cstdint>

namespace vectordb {

/**
 * 批量距离计算接口
 * 使用 BLAS/Accelerate 进行矩阵运算优化
 */

/**
 * 批量欧氏距离计算
 * 计算 query 与 vectors 中每个向量的欧氏距离
 * @param query 查询向量 [dim]
 * @param vectors 向量数组 [n][dim]
 * @param n 向量数量
 * @param dim 维度
 * @param distances 输出距离数组 [n]
 */
void batchEuclideanDistance(const float* query, const float* vectors,
                            size_t n, size_t dim, float* distances);

/**
 * 批量欧氏距离计算（多查询版本）
 * 计算 queries 中每个查询与 vectors 中每个向量的距离
 * @param queries 查询向量数组 [nQueries][dim]
 * @param vectors 向量数组 [nVectors][dim]
 * @param nQueries 查询数量
 * @param nVectors 向量数量
 * @param dim 维度
 * @param distances 输出距离矩阵 [nQueries][nVectors]
 */
void batchEuclideanDistanceMultiQuery(const float* queries, const float* vectors,
                                      size_t nQueries, size_t nVectors, size_t dim,
                                      float* distances);

/**
 * 矩阵乘法 C = A * B^T
 * 用于批量距离计算中的核心运算
 * @param A 矩阵 [m][k]
 * @param B 矩阵 [n][k]
 * @param C 结果矩阵 [m][n]
 * @param m A的行数
 * @param n B的行数
 * @param k 内维
 */
void matrixMultiply(const float* A, const float* B, float* C,
                    size_t m, size_t n, size_t k);

/**
 * 计算 ||x||^2 for each row
 * @param matrix 输入矩阵 [rows][cols]
 * @param rows 行数
 * @param cols 列数
 * @param norms 输出范数平方 [rows]
 */
void computeRowNormsSquared(const float* matrix, size_t rows, size_t cols, float* norms);

} // namespace vectordb
