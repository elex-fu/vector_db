package com.vectordb.config;

import lombok.Builder;
import lombok.Data;

/**
 * 向量压缩配置类
 * 用于配置向量存储的压缩选项
 */
@Data
@Builder
public class CompressionConfig {

    /**
     * 是否启用压缩
     */
    private boolean enabled;

    /**
     * 压缩类型
     */
    private CompressionType type;

    /**
     * PQ (Product Quantization) 子空间数量
     * 仅当使用PQ压缩时有效
     * 默认值: 64 (适用于128维向量，每个子空间2维)
     */
    private int pqSubspaces;

    /**
     * PQ 每个子空间的位数
     * 默认值: 8 (256个聚类中心)
     */
    private int pqBits;

    /**
     * PQ 训练迭代次数
     * 默认值: 25
     */
    private int pqIterations;

    /**
     * 压缩类型枚举
     */
    public enum CompressionType {
        /**
         * 不压缩
         */
        NONE,

        /**
         * Product Quantization 乘积量化
         * 高压缩比，适用于大规模向量存储
         * 压缩比: 4x ~ 64x (取决于配置)
         */
        PQ,

        /**
         * HNSW + PQ 混合索引
         * 结合HNSW的快速搜索和PQ的内存压缩
         * 推荐用于大规模高维向量
         */
        HNSWPQ
    }

    /**
     * 获取默认配置 (不启用压缩)
     */
    public static CompressionConfig defaultConfig() {
        return CompressionConfig.builder()
                .enabled(false)
                .type(CompressionType.NONE)
                .pqSubspaces(64)
                .pqBits(8)
                .pqIterations(25)
                .build();
    }

    /**
     * 获取PQ压缩配置 (启用压缩)
     */
    public static CompressionConfig pqConfig(int pqSubspaces, int pqBits) {
        return CompressionConfig.builder()
                .enabled(true)
                .type(CompressionType.PQ)
                .pqSubspaces(pqSubspaces)
                .pqBits(pqBits)
                .pqIterations(25)
                .build();
    }

    /**
     * 获取HNSW+PQ混合压缩配置 (推荐)
     */
    public static CompressionConfig hnswPqConfig(int pqSubspaces, int pqBits) {
        return CompressionConfig.builder()
                .enabled(true)
                .type(CompressionType.HNSWPQ)
                .pqSubspaces(pqSubspaces)
                .pqBits(pqBits)
                .pqIterations(25)
                .build();
    }

    /**
     * 获取推荐的压缩配置 (高精度版本)
     * 根据向量维度自动计算最佳PQ子空间数量
     * 目标: 每个子空间8维，平衡精度和压缩比
     *
     * @param dimension 向量维度
     * @return 推荐的压缩配置
     */
    public static CompressionConfig recommendedConfig(int dimension) {
        // 优化: 每个子空间8维，减少量化误差
        // 原实现: dimension/2 = 2维/子空间 (量化误差大)
        // 新实现: dimension/8 = 8维/子空间 (量化误差小)
        int targetSubDim = 8;
        int pqSubspaces = dimension / targetSubDim;

        // 确保至少8个子空间
        pqSubspaces = Math.max(8, pqSubspaces);

        // 确保dimension能被pqSubspaces整除
        while (dimension % pqSubspaces != 0 && pqSubspaces > 1) {
            pqSubspaces--;
        }

        return hnswPqConfig(pqSubspaces, 8);
    }

    /**
     * 获取高召回率配置 (低压缩比，高精度)
     * 每个子空间仅4维，量化误差更小
     *
     * @param dimension 向量维度
     * @return 高召回率配置
     */
    public static CompressionConfig highRecallConfig(int dimension) {
        // 每个子空间4维，精度更高
        int targetSubDim = 4;
        int pqSubspaces = dimension / targetSubDim;

        pqSubspaces = Math.max(16, pqSubspaces);

        while (dimension % pqSubspaces != 0 && pqSubspaces > 1) {
            pqSubspaces--;
        }

        return hnswPqConfig(pqSubspaces, 8);
    }

    /**
     * 获取高压缩比配置 (高压缩比，稍低精度)
     * 每个子空间16维，压缩比更高
     *
     * @param dimension 向量维度
     * @return 高压缩比配置
     */
    public static CompressionConfig highCompressionConfig(int dimension) {
        // 每个子空间16维，压缩比更高
        int targetSubDim = 16;
        int pqSubspaces = dimension / targetSubDim;

        pqSubspaces = Math.max(4, pqSubspaces);

        while (dimension % pqSubspaces != 0 && pqSubspaces > 1) {
            pqSubspaces--;
        }

        return hnswPqConfig(pqSubspaces, 8);
    }

    /**
     * 计算预期的压缩比
     *
     * @param dimension 向量维度
     * @return 压缩比 (原始大小 / 压缩后大小)
     */
    public double getCompressionRatio(int dimension) {
        if (!enabled || type == CompressionType.NONE) {
            return 1.0;
        }

        // 原始大小: dimension * sizeof(float) = dimension * 4 bytes
        double originalSize = dimension * 4.0;

        // 压缩后大小: pqSubspaces * sizeof(uint8_t) = pqSubspaces bytes
        double compressedSize = pqSubspaces * 1.0;

        return originalSize / compressedSize;
    }

    /**
     * 计算预期的内存节省百分比
     *
     * @param dimension 向量维度
     * @return 内存节省百分比 (0-100)
     */
    public double getMemorySavings(int dimension) {
        double ratio = getCompressionRatio(dimension);
        return (1.0 - 1.0 / ratio) * 100.0;
    }
}
