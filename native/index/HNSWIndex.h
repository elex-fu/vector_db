#pragma once
#include "VectorIndex.h"
#include "../core/VectorStore.h"
#include "../compute/DistanceUtils.h"
#include <vector>
#include <random>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <atomic>

namespace vectordb {

struct HNSWConfig {
    int M = 32;
    int efConstruction = 64;
    int efSearch = 64;
    int maxLevel = 16;
    double levelMultiplier = 1.0 / std::log(1.0 * M);
    int efSearchDelta = 32;
    float distanceThreshold = 0.0f;
    bool useEarlyTermination = true;
    int maxExpansionsMultiplier = 4;
    bool useHeuristicSelection = true;
    int heuristicCandidates = 8;
    int pruneOverflowFactor = 2;

    HNSWConfig() = default;

    int getEfSearch(int k, int dataSize = 1000) const {
        int baseEf = k + efSearchDelta;
        if (dataSize > 100) {
            float scaleFactor = 1.0f + 0.2f * std::log10(dataSize / 100.0f + 1.0f);
            baseEf = static_cast<int>(baseEf * scaleFactor);
        }
        int minEfMultiplier = 4;
        if (dataSize > 1000) minEfMultiplier = 5;
        if (dataSize > 5000) minEfMultiplier = 6;
        if (dataSize > 20000) minEfMultiplier = 8;
        int minEf = k * minEfMultiplier;
        int result = std::max(baseEf, minEf);
        int maxEf = 300;
        if (dataSize > 10000) maxEf = 400;
        return std::min(result, maxEf);
    }

    int getHeuristicCandidateCount() const { return M * heuristicCandidates; }
    int getMaxExpansions(int ef) const { return ef * maxExpansionsMultiplier; }
};

class HNSWIndex : public VectorIndex {
public:
    HNSWIndex(int dimension, int maxElements);
    HNSWIndex(int dimension, int maxElements, const HNSWConfig& config);

    void add(int id, const float* vector) override;
    void search(const float* query, int k,
               int* resultIds, float* resultDistances,
               int* resultCount) override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    int size() const override { return size_.load(std::memory_order_acquire); }
    int dimension() const override { return vectorStore_.dimension(); }
    int capacity() const override { return vectorStore_.capacity(); }

    void searchBatch(const float* queries, int nQueries, int k,
                    int* resultIds, float* resultDistances);
    void addBatch(const float* vectors, const int* ids, int n,
                 int* failedIndices = nullptr, int* failedCount = nullptr);
    void setNumThreads(int numThreads);
    int getNumThreads() const;

private:
    VectorStore vectorStore_;
    HNSWConfig config_;
    std::atomic<int> size_{0};
    std::atomic<int> entryPoint_{-1};
    std::mt19937 rng_;
    mutable std::shared_mutex mutex_;

    struct Node {
        int level;
        std::vector<std::vector<int>> neighbors;
    };
    std::vector<Node> nodes_;
    DistanceFunc distanceFunc_;
    int numThreads_ = 4;

    int getRandomLevel();
    void searchLevel(const float* query, int entryPoint, int ef, int level,
                    std::vector<std::pair<float, int>>& results);
    std::vector<int> selectNeighbors(const std::vector<std::pair<float, int>>& candidates, int M);
    std::vector<int> selectNeighborsHeuristic(const float* query,
                                              const std::vector<std::pair<float, int>>& candidates,
                                              int M, int level);
    void connectNeighbors(int newId, const std::vector<int>& neighbors, int level);
    void pruneNeighbors(int nodeId, int level);
    float computeDistance(const float* a, int bIndex);
};

} // namespace vectordb
