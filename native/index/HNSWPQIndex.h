#pragma once
#include "VectorIndex.h"
#include "../core/VectorStore.h"
#include "../compute/DistanceUtils.h"
#include "../compute/BatchDistance.h"
#include <vector>
#include <random>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <unordered_set>

namespace vectordb {

/**
 * HNSW + Product Quantization 混合索引
 * 使用 PQ 压缩向量，HNSW 构建图索引
 * 实现 ADC (Asymmetric Distance Computation) 加速搜索
 */

struct HNSWPQConfig {
    // HNSW 参数
    int M = 32;
    int efConstruction = 64;
    int efSearch = 64;
    int maxLevel = 16;
    double levelMultiplier = 1.0 / std::log(1.0 * 32);
    bool useHeuristicSelection = true;

    // PQ 参数 - 优化为更高精度
    int pqM = 64;          // PQ 子空间数量 (128维/64=2维/子空间，更高精度)
    int pqBits = 8;        // 每个子空间的位数 (256个聚类中心)
    int pqIterations = 25; // KMeans 迭代次数
};

class HNSWPQIndex : public VectorIndex {
public:
    HNSWPQIndex(int dimension, int maxElements);
    HNSWPQIndex(int dimension, int maxElements, const HNSWPQConfig& config);

    void add(int id, const float* vector) override;
    void search(const float* query, int k,
               int* resultIds, float* resultDistances,
               int* resultCount) override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    int size() const override { return size_.load(std::memory_order_acquire); }
    int dimension() const override { return dimension_; }
    int capacity() const override { return maxElements_; }

    // 训练 PQ
    void train(int nSamples, const float* samples);
    bool isTrained() const { return trained_; }

    // 批量操作
    void searchBatch(const float* queries, int nQueries, int k,
                    int* resultIds, float* resultDistances);
    void addBatch(const float* vectors, const int* ids, int n);

    // 内存统计
    size_t getMemoryUsage() const;
    float getCompressionRatio() const;

private:
    int dimension_;
    int maxElements_;
    HNSWPQConfig config_;
    std::atomic<int> size_{0};
    std::atomic<int> entryPoint_{-1};
    std::mt19937 rng_;
    mutable std::shared_mutex mutex_;
    mutable std::vector<std::unique_ptr<std::shared_mutex>> bucketMutexes_;  // 细粒度锁
    bool trained_ = false;
    static constexpr int NUM_BUCKETS = 64;  // 桶数量，用于细粒度锁

    // PQ 数据
    int subDim_ = 0;
    int nCentroids_ = 0;
    std::vector<float> codebooks_;  // [pqM][nCentroids][subDim]
    std::vector<uint8_t> codes_;    // [nVectors][pqM]

    // 内存池优化的邻居存储
    struct NeighborLevel {
        int* data;           // 邻居ID数组（内存池分配）
        uint16_t size;       // 当前邻居数量
        uint16_t capacity;   // 容量
    };

    struct Node {
        int level;
        NeighborLevel* levels;  // 每层邻居数组（内存池分配）
    };
    std::vector<Node> nodes_;

    // 内存池
    std::vector<int> neighborPool_;       // 邻居ID内存池
    size_t neighborPoolUsed_ = 0;         // 已使用大小
    static constexpr size_t NEIGHBOR_POOL_BLOCK_SIZE = 1024;  // 每次扩容大小

    // 原始向量存储 (可选，用于 refine)
    VectorStore vectorStore_;

    // 距离函数
    DistanceFunc distanceFunc_;
    BatchDistanceFunc batchDistFunc_;

    // 线程局部存储
    mutable std::vector<std::unordered_set<int>> threadVisited_;

    // 内部方法
    int getRandomLevel();
    void encode(const float* vector, uint8_t* codes);
    float computeDistancePQ(const float* query, int nodeId);
    float computeExactDistance(int idA, int idB);
    float computeExactDistanceToQuery(const float* query, int nodeId);

    void searchLevel(const float* query, const uint8_t* queryCodes,
                     int entryPoint, int ef, int level,
                     std::vector<std::pair<float, int>>& results);

    std::vector<int> selectNeighbors(const std::vector<std::pair<float, int>>& candidates, int M);
    std::vector<int> selectNeighborsHeuristic(const std::vector<std::pair<float, int>>& candidates,
                                              int M, int level);

    void connectNeighbors(int newId, const std::vector<int>& neighbors, int level);
    void pruneNeighbors(int nodeId, int level);

    // 锁优化辅助函数
    std::shared_mutex& getBucketMutex(int nodeId);
    void lockBuckets(const std::vector<int>& nodeIds);
    void unlockBuckets(const std::vector<int>& nodeIds);

    // 内存池辅助函数
    NeighborLevel* allocateNeighborLevels(int levelCount);
    void addNeighborToLevel(int nodeId, int level, int neighborId);
    void clearNeighborLevel(int nodeId, int level);
    void reserveNeighborPool(size_t totalNeighbors);

    // PQ 训练
    void trainSubspace(int subspaceIdx, int nSamples, const float* samples);
    int findNearestCentroid(int subspaceIdx, const float* subVector);
    float* getCodebookCentroid(int subspaceIdx, int centroidIdx);
};

} // namespace vectordb
