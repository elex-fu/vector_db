#include "LSHIndex.h"
#include "../compute/DistanceUtils.h"
#include <random>
#include <algorithm>
#include <unordered_map>

namespace vectordb {

LSHIndex::LSHIndex(int dimension, int maxElements)
    : LSHIndex(dimension, maxElements, 10, 20) {}

LSHIndex::LSHIndex(int dimension, int maxElements, int numHashTables, int numHashFunctions)
    : dimension_(dimension), maxElements_(maxElements), numHashTables_(numHashTables), numHashFunctions_(numHashFunctions),
      vectorStore_(dimension, maxElements) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    rng_.seed(seed);

    generateHashFunctions();
    hashTables_.resize(numHashTables);
}

void LSHIndex::generateHashFunctions() {
    hashFunctions_.resize(numHashTables_);
    hashBiases_.resize(numHashTables_);

    std::normal_distribution<float> dist(0.0, 1.0);

    for (int t = 0; t < numHashTables_; t++) {
        hashFunctions_[t].resize(numHashFunctions_);
        hashBiases_[t].resize(numHashFunctions_);

        for (int h = 0; h < numHashFunctions_; h++) {
            hashFunctions_[t][h].resize(dimension_);
            for (int d = 0; d < dimension_; d++) {
                hashFunctions_[t][h][d] = dist(rng_);
            }
            hashBiases_[t][h] = dist(rng_) * 0.5f;
        }
    }
}

std::vector<int> LSHIndex::computeHash(const float* vector, int tableIdx) {
    std::vector<int> hash(numHashFunctions_);

    for (int h = 0; h < numHashFunctions_; h++) {
        float dot = 0.0f;
        for (int d = 0; d < dimension_; d++) {
            dot += vector[d] * hashFunctions_[tableIdx][h][d];
        }
        hash[h] = (dot + hashBiases_[tableIdx][h] > 0) ? 1 : 0;
    }

    return hash;
}

void LSHIndex::add(int id, const float* vector) {
    int index = size_;
    vectorStore_.add(id, vector);

    for (int t = 0; t < numHashTables_; t++) {
        std::vector<int> hash = computeHash(vector, t);

        int bucketIdx = 0;
        for (int h = 0; h < numHashFunctions_; h++) {
            bucketIdx = (bucketIdx << 1) | hash[h];
        }

        if (bucketIdx >= static_cast<int>(hashTables_[t].size())) {
            hashTables_[t].resize(bucketIdx + 1);
        }
        hashTables_[t][bucketIdx].push_back(index);
    }

    size_++;
}

void LSHIndex::search(const float* query, int k,
                     int* resultIds, float* resultDistances,
                     int* resultCount) {

    DistanceFunc distFunc = getEuclideanDistanceFunc();
    std::unordered_map<int, int> candidates;

    for (int t = 0; t < numHashTables_; t++) {
        std::vector<int> hash = computeHash(query, t);

        int bucketIdx = 0;
        for (int h = 0; h < numHashFunctions_; h++) {
            bucketIdx = (bucketIdx << 1) | hash[h];
        }

        if (bucketIdx < static_cast<int>(hashTables_[t].size())) {
            for (int idx : hashTables_[t][bucketIdx]) {
                candidates[idx]++;
            }
        }
    }

    std::vector<std::pair<float, int>> scoredCandidates;
    for (const auto& pair : candidates) {
        const float* vec = vectorStore_.getVector(pair.first);
        float dist = distFunc(query, vec, dimension_);
        scoredCandidates.emplace_back(dist, vectorStore_.getId(pair.first));
    }

    std::partial_sort(scoredCandidates.begin(),
                      scoredCandidates.begin() + std::min(k, static_cast<int>(scoredCandidates.size())),
                      scoredCandidates.end());

    int count = std::min(k, static_cast<int>(scoredCandidates.size()));
    for (int i = 0; i < count; i++) {
        resultDistances[i] = scoredCandidates[i].first;
        resultIds[i] = scoredCandidates[i].second;
    }
    *resultCount = count;
}

void LSHIndex::save(const std::string& /*path*/) {
    // TODO: Implement save
}

void LSHIndex::load(const std::string& /*path*/) {
    // TODO: Implement load
}

} // namespace vectordb