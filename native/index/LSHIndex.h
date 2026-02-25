#pragma once
#include "VectorIndex.h"
#include "../core/VectorStore.h"
#include <vector>
#include <random>

namespace vectordb {

class LSHIndex : public VectorIndex {
public:
    LSHIndex(int dimension, int maxElements);
    LSHIndex(int dimension, int maxElements, int numHashTables, int numHashFunctions);

    void add(int id, const float* vector) override;
    void search(const float* query, int k,
               int* resultIds, float* resultDistances,
               int* resultCount) override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    int size() const override { return size_; }
    int dimension() const override { return dimension_; }
    int capacity() const override { return maxElements_; }

private:
    int dimension_;
    int maxElements_;
    int numHashTables_;
    int numHashFunctions_;
    int size_ = 0;

    VectorStore vectorStore_;
    std::vector<std::vector<std::vector<float>>> hashFunctions_;
    std::vector<std::vector<float>> hashBiases_;
    std::vector<std::vector<std::vector<int>>> hashTables_;

    std::mt19937 rng_;

    void generateHashFunctions();
    std::vector<int> computeHash(const float* vector, int tableIdx);
};

} // namespace vectordb
