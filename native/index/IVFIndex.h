#pragma once
#include "VectorIndex.h"
#include "../core/VectorStore.h"
#include "../compute/DistanceUtils.h"
#include <vector>

namespace vectordb {

struct IVFConfig {
    int nLists = 100;
    int nProbes = 10;
    int maxIterations = 25;
};

class IVFIndex : public VectorIndex {
public:
    IVFIndex(int dimension, int maxElements);
    IVFIndex(int dimension, int maxElements, const IVFConfig& config);

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

private:
    VectorStore vectorStore_;
    IVFConfig config_;
    int size_ = 0;
    bool trained_ = false;
    std::vector<float> centroids_;
    std::vector<std::vector<int>> invertedLists_;
    std::vector<int> idToList_;
    DistanceFunc distanceFunc_;

    int findNearestCentroid(const float* vector);
};

} // namespace vectordb
