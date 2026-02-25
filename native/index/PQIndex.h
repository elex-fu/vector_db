#pragma once
#include "VectorIndex.h"
#include "../core/VectorStore.h"
#include "../compute/DistanceUtils.h"
#include <vector>
#include <cstdint>

namespace vectordb {

struct PQConfig {
    int M = 8;
    int nBits = 8;
    int maxIterations = 25;
};

class PQIndex : public VectorIndex {
public:
    PQIndex(int dimension, int maxElements);
    PQIndex(int dimension, int maxElements, const PQConfig& config);

    void add(int id, const float* vector) override;
    void search(const float* query, int k,
               int* resultIds, float* resultDistances,
               int* resultCount) override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    int size() const override { return size_; }
    int dimension() const override { return vectorStore_.dimension(); }
    int capacity() const override { return vectorStore_.capacity(); }

    void train(int nSamples, const float* samples);
    bool isTrained() const { return trained_; }
    void addBatch(const float* vectors, const int* ids, int n);
    void searchBatch(const float* queries, int nQueries, int k,
                    int* resultIds, float* resultDistances);

private:
    VectorStore vectorStore_;
    int size_ = 0;
    bool trained_ = false;
    PQConfig config_;
    int subDim_ = 0;
    int nCentroids_ = 0;
    std::vector<float> codebooks_;
    std::vector<uint8_t> codes_;
    DistanceFunc distanceFunc_;

    void trainSubspace(int subspaceIdx, int nSamples, const float* samples);
    int findNearestCentroid(int subspaceIdx, const float* subVector);
    void encode(const float* vector, uint8_t* codes);
    float* getCodebookCentroid(int subspaceIdx, int centroidIdx) {
        return codebooks_.data() +
               (static_cast<size_t>(subspaceIdx) * nCentroids_ + centroidIdx) * subDim_;
    }
};

} // namespace vectordb
