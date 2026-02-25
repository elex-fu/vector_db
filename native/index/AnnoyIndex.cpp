#include "AnnoyIndex.h"
#include "../compute/DistanceUtils.h"
#include <random>
#include <algorithm>
#include <limits>

namespace vectordb {

AnnoyIndex::AnnoyIndex(int dimension, int maxElements)
    : AnnoyIndex(dimension, maxElements, 10) {}

AnnoyIndex::AnnoyIndex(int dimension, int maxElements, int numTrees)
    : dimension_(dimension), maxElements_(maxElements), numTrees_(numTrees),
      vectorStore_(dimension, maxElements) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    rng_.seed(seed);
    trees_.resize(numTrees);
}

void AnnoyIndex::add(int id, const float* vector) {
    vectorStore_.add(id, vector);
    size_++;
}

void AnnoyIndex::build() {
    std::vector<int> indices(size_);
    for (int i = 0; i < size_; i++) {
        indices[i] = i;
    }

    for (int t = 0; t < numTrees_; t++) {
        std::shuffle(indices.begin(), indices.end(), rng_);
        buildTree(t, indices);
    }

    built_ = true;
}

void AnnoyIndex::buildTree(int treeIdx, const std::vector<int>& indices) {
    if (indices.empty()) return;

    std::vector<int> toProcess = indices;
    trees_[treeIdx].clear();

    trees_[treeIdx].push_back(Node{});
    int rootIdx = 0;

    std::normal_distribution<float> dist(0.0, 1.0);

    while (!toProcess.empty()) {
        int nodeIdx = toProcess.back();
        toProcess.pop_back();

        if (static_cast<int>(toProcess.size()) < 10) {
            trees_[treeIdx][nodeIdx].indices = toProcess;
            toProcess.clear();
            continue;
        }

        std::vector<float> hyperplane(dimension_);
        for (int d = 0; d < dimension_; d++) {
            hyperplane[d] = dist(rng_);
        }

        float norm = 0.0f;
        for (float v : hyperplane) {
            norm += v * v;
        }
        norm = std::sqrt(norm);
        for (float& v : hyperplane) {
            v /= norm;
        }

        float bias = 0.0f;
        for (int idx : toProcess) {
            const float* vec = vectorStore_.getVector(idx);
            float dot = 0.0f;
            for (int d = 0; d < dimension_; d++) {
                dot += vec[d] * hyperplane[d];
            }
            bias += dot;
        }
        bias /= toProcess.size();

        std::vector<int> leftIndices, rightIndices;
        for (int idx : toProcess) {
            const float* vec = vectorStore_.getVector(idx);
            float dot = 0.0f;
            for (int d = 0; d < dimension_; d++) {
                dot += vec[d] * hyperplane[d];
            }
            if (dot < bias) {
                leftIndices.push_back(idx);
            } else {
                rightIndices.push_back(idx);
            }
        }

        trees_[treeIdx][nodeIdx].hyperplane = hyperplane;
        trees_[treeIdx][nodeIdx].bias = bias;

        if (!leftIndices.empty()) {
            trees_[treeIdx][nodeIdx].left = trees_[treeIdx].size();
            trees_[treeIdx].push_back(Node{});
            toProcess.insert(toProcess.end(), leftIndices.begin(), leftIndices.end());
        }

        if (!rightIndices.empty()) {
            trees_[treeIdx][nodeIdx].right = trees_[treeIdx].size();
            trees_[treeIdx].push_back(Node{});
            toProcess.insert(toProcess.end(), rightIndices.begin(), rightIndices.end());
        }
    }
}

void AnnoyIndex::search(const float* query, int k,
                       int* resultIds, float* resultDistances,
                       int* resultCount) {
    if (!built_) {
        *resultCount = 0;
        return;
    }

    DistanceFunc distFunc = getEuclideanDistanceFunc();
    const int maxCandidates = k * numTrees_ * 2;
    std::vector<int> candidates;

    for (int t = 0; t < numTrees_; t++) {
        searchTree(t, 0, query, candidates, maxCandidates);
    }

    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    std::vector<std::pair<float, int>> results;
    for (int idx : candidates) {
        const float* vec = vectorStore_.getVector(idx);
        float dist = distFunc(query, vec, dimension_);
        results.emplace_back(dist, vectorStore_.getId(idx));
    }

    std::partial_sort(results.begin(),
                      results.begin() + std::min(k, static_cast<int>(results.size())),
                      results.end());

    int count = std::min(k, static_cast<int>(results.size()));
    for (int i = 0; i < count; i++) {
        resultDistances[i] = results[i].first;
        resultIds[i] = results[i].second;
    }
    *resultCount = count;
}

void AnnoyIndex::searchTree(int treeIdx, int nodeIdx, const float* query,
                           std::vector<int>& candidates, int maxCandidates) {
    if (nodeIdx < 0 || nodeIdx >= static_cast<int>(trees_[treeIdx].size())) return;
    if (static_cast<int>(candidates.size()) >= maxCandidates) return;

    const Node& node = trees_[treeIdx][nodeIdx];

    if (node.left < 0 && node.right < 0) {
        candidates.insert(candidates.end(), node.indices.begin(), node.indices.end());
        return;
    }

    float dot = 0.0f;
    for (int d = 0; d < dimension_; d++) {
        dot += query[d] * node.hyperplane[d];
    }

    if (dot < node.bias) {
        searchTree(treeIdx, node.left, query, candidates, maxCandidates);
        searchTree(treeIdx, node.right, query, candidates, maxCandidates);
    } else {
        searchTree(treeIdx, node.right, query, candidates, maxCandidates);
        searchTree(treeIdx, node.left, query, candidates, maxCandidates);
    }
}

void AnnoyIndex::save(const std::string& /*path*/) {
    // TODO: Implement save
}

void AnnoyIndex::load(const std::string& /*path*/) {
    // TODO: Implement load
}

} // namespace vectordb
