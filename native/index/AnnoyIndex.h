#pragma once
#include "VectorIndex.h"
#include "../core/VectorStore.h"
#include <vector>
#include <random>

namespace vectordb {

class AnnoyIndex : public VectorIndex {
public:
    AnnoyIndex(int dimension, int maxElements);
    AnnoyIndex(int dimension, int maxElements, int numTrees);

    void add(int id, const float* vector) override;
    void search(const float* query, int k,
               int* resultIds, float* resultDistances,
               int* resultCount) override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    int size() const override { return size_; }
    int dimension() const override { return dimension_; }
    int capacity() const override { return maxElements_; }

    void build();

private:
    int dimension_;
    int maxElements_;
    int numTrees_;
    int size_ = 0;
    bool built_ = false;

    VectorStore vectorStore_;
    std::mt19937 rng_;

    struct Node {
        int left = -1;
        int right = -1;
        std::vector<float> hyperplane;
        float bias = 0.0f;
        std::vector<int> indices;
    };

    std::vector<std::vector<Node>> trees_;

    void buildTree(int treeIdx, const std::vector<int>& indices);
    void searchTree(int treeIdx, int nodeIdx, const float* query,
                   std::vector<int>& candidates, int maxCandidates);
};

} // namespace vectordb
