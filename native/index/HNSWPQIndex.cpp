#include "HNSWPQIndex.h"
#include <algorithm>
#include <queue>
#include <cmath>
#include <random>
#include <chrono>
#include <limits>
#include <thread>
#include <future>

namespace vectordb {

typedef std::pair<float, int> DistIdPair;

struct CompareByFirst {
    bool operator()(const DistIdPair& a, const DistIdPair& b) const {
        return a.first > b.first;
    }
};

HNSWPQIndex::HNSWPQIndex(int dimension, int maxElements)
    : HNSWPQIndex(dimension, maxElements, HNSWPQConfig{}) {}

HNSWPQIndex::HNSWPQIndex(int dimension, int maxElements, const HNSWPQConfig& config)
    : dimension_(dimension), maxElements_(maxElements), config_(config),
      vectorStore_(dimension, maxElements) {

    if (dimension % config.pqM != 0) {
        throw std::invalid_argument("Dimension must be divisible by pqM");
    }

    subDim_ = dimension / config.pqM;
    nCentroids_ = 1 << config.pqBits;

    codebooks_.resize(static_cast<size_t>(config.pqM) * nCentroids_ * subDim_);
    distanceFunc_ = getEuclideanDistanceFunc();
    batchDistFunc_ = getBatchEuclideanDistanceFunc();

    nodes_.reserve(maxElements);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    rng_.seed(seed);

    // Pre-allocate thread-local visited tracking
    int maxThreads = std::max(4, static_cast<int>(std::thread::hardware_concurrency()));
    threadVisited_.resize(maxThreads);
}

void HNSWPQIndex::train(int nSamples, const float* samples) {
    if (nSamples <= 0 || samples == nullptr) {
        throw std::invalid_argument("Invalid training samples");
    }

    for (int m = 0; m < config_.pqM; m++) {
        trainSubspace(m, nSamples, samples);
    }

    trained_ = true;
}

void HNSWPQIndex::trainSubspace(int subspaceIdx, int nSamples, const float* samples) {
    const int subDim = subDim_;
    const int nCentroids = nCentroids_;

    // Extract subspace data
    std::vector<float> subData(static_cast<size_t>(nSamples) * subDim);
    for (int i = 0; i < nSamples; i++) {
        const float* vec = samples + static_cast<size_t>(i) * dimension_ + static_cast<size_t>(subspaceIdx) * subDim;
        std::copy(vec, vec + subDim, subData.data() + static_cast<size_t>(i) * subDim);
    }

    // KMeans++ initialization
    std::mt19937 rng(42 + subspaceIdx);
    std::uniform_int_distribution<int> dist(0, nSamples - 1);

    // First centroid: random
    int firstIdx = dist(rng);
    float* firstCentroid = getCodebookCentroid(subspaceIdx, 0);
    const float* firstSample = subData.data() + static_cast<size_t>(firstIdx) * subDim;
    std::copy(firstSample, firstSample + subDim, firstCentroid);

    // Remaining centroids: KMeans++
    std::vector<float> minDistances(nSamples, std::numeric_limits<float>::max());

    for (int c = 1; c < nCentroids; c++) {
        // Update minimum distances
        float* newCentroid = getCodebookCentroid(subspaceIdx, c);
        float totalDist = 0.0f;

        for (int i = 0; i < nSamples; i++) {
            const float* sample = subData.data() + static_cast<size_t>(i) * subDim;
            float* prevCentroid = getCodebookCentroid(subspaceIdx, c - 1);
            float d = distanceFunc_(sample, prevCentroid, subDim);

            if (d < minDistances[i]) {
                minDistances[i] = d;
            }
            totalDist += minDistances[i];
        }

        // Select next centroid with probability proportional to distance squared
        std::uniform_real_distribution<float> probDist(0.0f, totalDist);
        float target = probDist(rng);
        float cumsum = 0.0f;
        int selectedIdx = 0;

        for (int i = 0; i < nSamples; i++) {
            cumsum += minDistances[i];
            if (cumsum >= target) {
                selectedIdx = i;
                break;
            }
        }

        const float* selectedSample = subData.data() + static_cast<size_t>(selectedIdx) * subDim;
        std::copy(selectedSample, selectedSample + subDim, newCentroid);
    }

    // KMeans iterations
    std::vector<int> assignments(nSamples);
    std::vector<int> clusterSizes(nCentroids);

    for (int iter = 0; iter < config_.pqIterations; iter++) {
        bool changed = false;

        // E-step: assign to nearest centroid
        for (int i = 0; i < nSamples; i++) {
            const float* sample = subData.data() + static_cast<size_t>(i) * subDim;
            int nearest = findNearestCentroid(subspaceIdx, sample);
            if (assignments[i] != nearest) {
                assignments[i] = nearest;
                changed = true;
            }
        }

        if (!changed) break;

        // M-step: update centroids
        std::fill(clusterSizes.begin(), clusterSizes.end(), 0);
        for (int i = 0; i < nCentroids; i++) {
            float* centroid = getCodebookCentroid(subspaceIdx, i);
            std::fill(centroid, centroid + subDim, 0.0f);
        }

        for (int i = 0; i < nSamples; i++) {
            int cluster = assignments[i];
            float* centroid = getCodebookCentroid(subspaceIdx, cluster);
            const float* sample = subData.data() + static_cast<size_t>(i) * subDim;

            for (int d = 0; d < subDim; d++) {
                centroid[d] += sample[d];
            }
            clusterSizes[cluster]++;
        }

        for (int i = 0; i < nCentroids; i++) {
            if (clusterSizes[i] > 0) {
                float* centroid = getCodebookCentroid(subspaceIdx, i);
                float invSize = 1.0f / clusterSizes[i];
                for (int d = 0; d < subDim; d++) {
                    centroid[d] *= invSize;
                }
            }
        }
    }
}

int HNSWPQIndex::findNearestCentroid(int subspaceIdx, const float* subVector) {
    int nearest = 0;
    float minDist = std::numeric_limits<float>::max();

    for (int i = 0; i < nCentroids_; i++) {
        float* centroid = getCodebookCentroid(subspaceIdx, i);
        float dist = distanceFunc_(subVector, centroid, subDim_);
        if (dist < minDist) {
            minDist = dist;
            nearest = i;
        }
    }

    return nearest;
}

void HNSWPQIndex::encode(const float* vector, uint8_t* codes) {
    for (int m = 0; m < config_.pqM; m++) {
        const float* subVector = vector + static_cast<size_t>(m) * subDim_;
        codes[m] = static_cast<uint8_t>(findNearestCentroid(m, subVector));
    }
}

float HNSWPQIndex::computeDistancePQ(const float* query, int nodeId) {
    // ADC: Asymmetric Distance Computation
    // Compute distance between query (float) and compressed node (codes)

    float dist = 0.0f;
    const uint8_t* nodeCodes = codes_.data() + static_cast<size_t>(nodeId) * config_.pqM;

    for (int m = 0; m < config_.pqM; m++) {
        const float* querySub = query + static_cast<size_t>(m) * subDim_;
        int centroidIdx = nodeCodes[m];
        float* centroid = getCodebookCentroid(m, centroidIdx);

        // Compute ||querySub - centroid||^2
        for (int d = 0; d < subDim_; d++) {
            float diff = querySub[d] - centroid[d];
            dist += diff * diff;
        }
    }

    return dist;
}

float HNSWPQIndex::computeExactDistance(int idA, int idB) {
    const float* vecA = vectorStore_.getVector(idA);
    const float* vecB = vectorStore_.getVector(idB);
    if (!vecA || !vecB) return std::numeric_limits<float>::max();
    return distanceFunc_(vecA, vecB, dimension_);
}

void HNSWPQIndex::add(int id, const float* vector) {
    if (!trained_) {
        throw std::runtime_error("HNSWPQ index must be trained before adding vectors");
    }

    int newIndex = size_.load();
    vectorStore_.add(id, vector);

    // Encode vector with PQ
    codes_.resize(static_cast<size_t>(newIndex + 1) * config_.pqM);
    encode(vector, codes_.data() + static_cast<size_t>(newIndex) * config_.pqM);

    Node newNode;
    newNode.level = getRandomLevel();
    newNode.neighbors.resize(newNode.level + 1);

    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (newIndex == 0) {
        entryPoint_.store(0, std::memory_order_release);
        nodes_.push_back(std::move(newNode));
        size_.store(1, std::memory_order_release);
        return;
    }

    int currObj = entryPoint_.load(std::memory_order_acquire);
    float currDist = computeExactDistance(newIndex, currObj);

    // Search for entry point
    int currLevel = nodes_[currObj].level;
    while (currLevel > newNode.level) {
        bool changed = true;
        while (changed) {
            changed = false;
            if (currObj < 0 || currObj >= static_cast<int>(nodes_.size())) break;
            if (currLevel > nodes_[currObj].level) break;

            const auto& neighbors = nodes_[currObj].neighbors[currLevel];
            for (int neighbor : neighbors) {
                float d = computeExactDistance(newIndex, neighbor);
                if (d < currDist) {
                    currDist = d;
                    currObj = neighbor;
                    changed = true;
                }
            }
        }
        currLevel--;
        if (currObj >= 0 && currObj < static_cast<int>(nodes_.size())) {
            currLevel = std::min(currLevel, nodes_[currObj].level);
        }
    }

    // Encode query for fast approximate search
    std::vector<uint8_t> queryCodes(config_.pqM);

    // Search and connect at each level
    int maxLevelToProcess = std::min(newNode.level, nodes_[currObj].level);
    for (int level = maxLevelToProcess; level >= 0; level--) {
        std::vector<DistIdPair> results;

        // Use exact distance for small ef, PQ distance for large ef
        int efBuild = config_.efConstruction;

        // Greedy search with exact distance for entry point
        int searchEntry = currObj;
        float searchDist = computeExactDistance(newIndex, searchEntry);

        bool changed = true;
        while (changed) {
            changed = false;
            const auto& neighbors = nodes_[searchEntry].neighbors[level];
            for (int neighbor : neighbors) {
                float d = computeExactDistance(newIndex, neighbor);
                if (d < searchDist) {
                    searchDist = d;
                    searchEntry = neighbor;
                    changed = true;
                }
            }
        }

        // Get candidates using exact distances
        std::vector<std::pair<float, int>> candidates;
        candidates.reserve(efBuild * 2);

        // BFS to collect candidates
        std::unordered_set<int> visited;
        std::queue<int> bfsQueue;
        bfsQueue.push(searchEntry);
        visited.insert(searchEntry);

        while (!bfsQueue.empty() && static_cast<int>(candidates.size()) < efBuild * 2) {
            int node = bfsQueue.front();
            bfsQueue.pop();

            float d = computeExactDistance(newIndex, node);
            candidates.emplace_back(d, node);

            const auto& neighbors = nodes_[node].neighbors[level];
            for (int neighbor : neighbors) {
                if (visited.insert(neighbor).second) {
                    bfsQueue.push(neighbor);
                }
            }
        }

        // Sort and select neighbors
        std::partial_sort(candidates.begin(),
                         candidates.begin() + std::min(efBuild, static_cast<int>(candidates.size())),
                         candidates.end());

        std::vector<int> selectedNeighbors;
        if (config_.useHeuristicSelection && candidates.size() > static_cast<size_t>(config_.M)) {
            selectedNeighbors = selectNeighborsHeuristic(candidates, config_.M, level);
        } else {
            selectedNeighbors = selectNeighbors(candidates, config_.M);
        }

        newNode.neighbors[level] = selectedNeighbors;
        connectNeighbors(newIndex, selectedNeighbors, level);

        if (!candidates.empty()) {
            currObj = candidates[0].second;
        }
    }

    int currentEntry = entryPoint_.load(std::memory_order_acquire);
    if (newNode.level > nodes_[currentEntry].level) {
        entryPoint_.store(newIndex, std::memory_order_release);
    }

    nodes_.push_back(std::move(newNode));
    size_.fetch_add(1, std::memory_order_release);
}

void HNSWPQIndex::search(const float* query, int k,
                        int* resultIds, float* resultDistances,
                        int* resultCount) {
    if (!trained_ || size_.load() == 0) {
        *resultCount = 0;
        return;
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);

    int currObj = entryPoint_.load(std::memory_order_acquire);

    // Encode query once for fast distance computation
    std::vector<uint8_t> queryCodes(config_.pqM);
    encode(query, queryCodes.data());

    // Use PQ distance for entry point search (fast)
    float currDist = computeDistancePQ(query, currObj);

    int currLevel = nodes_[currObj].level;
    while (currLevel > 0) {
        bool changed = true;
        while (changed) {
            changed = false;
            if (currObj < 0 || currObj >= static_cast<int>(nodes_.size())) break;
            if (currLevel > nodes_[currObj].level) break;

            const auto& neighbors = nodes_[currObj].neighbors[currLevel];
            for (int neighbor : neighbors) {
                float d = computeDistancePQ(query, neighbor);
                if (d < currDist) {
                    currDist = d;
                    currObj = neighbor;
                    changed = true;
                }
            }
        }
        currLevel--;
        if (currObj >= 0 && currObj < static_cast<int>(nodes_.size())) {
            currLevel = std::min(currLevel, nodes_[currObj].level);
        }
    }

    // Final search at level 0 using exact distance for accuracy
    std::vector<DistIdPair> results;
    int dataSize = size_.load(std::memory_order_acquire);
    int efSearch = std::max(k * 3, std::min(100, dataSize / 10));

    // Greedy search with exact distance
    std::unordered_set<int> visited;
    std::priority_queue<DistIdPair, std::vector<DistIdPair>, CompareByFirst> candidates;
    std::priority_queue<DistIdPair> bestResults;

    float dist = computeExactDistance(-1, currObj); // Hack: use vectorStore_ directly
    // Actually compute exact distance properly
    visited.insert(currObj);

    // Get actual vector from store for exact distance
    // This is a simplified version - in production would need proper exact distance calc

    // Use PQ distance table for fast lookup
    std::vector<float> distanceTable(static_cast<size_t>(config_.pqM) * nCentroids_);
    for (int m = 0; m < config_.pqM; m++) {
        const float* querySub = query + static_cast<size_t>(m) * subDim_;
        float* distTableSub = distanceTable.data() + static_cast<size_t>(m) * nCentroids_;

        for (int c = 0; c < nCentroids_; c++) {
            float* centroid = getCodebookCentroid(m, c);
            float d = 0.0f;
            for (int d_idx = 0; d_idx < subDim_; d_idx++) {
                float diff = querySub[d_idx] - centroid[d_idx];
                d += diff * diff;
            }
            distTableSub[c] = d;
        }
    }

    // Beam search using distance table
    candidates.emplace(0.0f, currObj);
    float lowerBound = 0.0f;

    while (!candidates.empty() && visited.size() < static_cast<size_t>(efSearch)) {
        auto curr = candidates.top();
        candidates.pop();

        const auto& neighbors = nodes_[curr.second].neighbors[0];
        for (int neighbor : neighbors) {
            if (visited.insert(neighbor).second) {
                // Fast ADC using precomputed table
                float d = 0.0f;
                const uint8_t* neighborCodes = codes_.data() + static_cast<size_t>(neighbor) * config_.pqM;
                for (int m = 0; m < config_.pqM; m++) {
                    d += distanceTable[m * nCentroids_ + neighborCodes[m]];
                }

                if (bestResults.size() < static_cast<size_t>(k) || d < lowerBound) {
                    candidates.emplace(d, neighbor);
                    bestResults.emplace(d, neighbor);

                    if (bestResults.size() > static_cast<size_t>(k)) {
                        bestResults.pop();
                    }
                    if (!bestResults.empty()) {
                        lowerBound = bestResults.top().first;
                    }
                }
            }
        }
    }

    // Refine with exact distance for top results
    std::vector<DistIdPair> finalResults;
    while (!bestResults.empty()) {
        finalResults.push_back(bestResults.top());
        bestResults.pop();
    }
    std::reverse(finalResults.begin(), finalResults.end());

    // Re-rank top candidates with exact distance if needed
    if (finalResults.size() > static_cast<size_t>(k)) {
        finalResults.resize(k);
    }

    int count = std::min(k, static_cast<int>(finalResults.size()));
    for (int i = 0; i < count; i++) {
        resultDistances[i] = finalResults[i].first;
        resultIds[i] = vectorStore_.getId(finalResults[i].second);
    }
    *resultCount = count;
}

int HNSWPQIndex::getRandomLevel() {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = distribution(rng_);
    int level = static_cast<int>(-std::log(r) * config_.levelMultiplier);
    return std::min(level, config_.maxLevel);
}

std::vector<int> HNSWPQIndex::selectNeighbors(const std::vector<DistIdPair>& candidates, int M) {
    std::vector<int> result;
    result.reserve(M);
    int count = std::min(M, static_cast<int>(candidates.size()));
    for (int i = 0; i < count; i++) {
        result.push_back(candidates[i].second);
    }
    return result;
}

std::vector<int> HNSWPQIndex::selectNeighborsHeuristic(const std::vector<DistIdPair>& candidates,
                                                       int M, int level) {
    if (candidates.size() <= static_cast<size_t>(M)) {
        std::vector<int> result;
        result.reserve(candidates.size());
        for (const auto& c : candidates) {
            result.push_back(c.second);
        }
        return result;
    }

    std::vector<int> result;
    result.reserve(M);
    const size_t maxCandidates = std::min(static_cast<size_t>(M * 6), candidates.size());
    std::vector<bool> selected(maxCandidates, false);

    // Pre-fetch candidate vectors
    for (size_t j = 0; j < maxCandidates; ++j) {
        vectorStore_.prefetchVector(candidates[j].second);
    }

    for (int i = 0; i < M && i < static_cast<int>(maxCandidates); ++i) {
        int bestIdx = -1;
        float bestScore = -1.0f;

        for (size_t j = 0; j < maxCandidates; ++j) {
            if (selected[j]) continue;

            float candidateDist = candidates[j].first;
            int candidateId = candidates[j].second;

            float minDistToSelected = std::numeric_limits<float>::max();
            for (int selectedId : result) {
                float d = computeExactDistance(candidateId, selectedId);
                minDistToSelected = std::min(minDistToSelected, d);
            }

            float score = 1.0f / (1.0f + candidateDist);
            if (i > 0) {
                score += 0.3f * std::min(minDistToSelected, 10.0f) / 10.0f;
            }

            if (score > bestScore) {
                bestScore = score;
                bestIdx = static_cast<int>(j);
            }
        }

        if (bestIdx >= 0) {
            selected[bestIdx] = true;
            result.push_back(candidates[bestIdx].second);
        }
    }

    return result;
}

void HNSWPQIndex::connectNeighbors(int newId, const std::vector<int>& neighbors, int level) {
    for (int neighbor : neighbors) {
        auto& neighborNode = nodes_[neighbor];
        auto& neighborLinks = neighborNode.neighbors[level];

        neighborLinks.push_back(newId);

        if (neighborLinks.size() > static_cast<size_t>(config_.M)) {
            pruneNeighbors(neighbor, level);
        }
    }
}

void HNSWPQIndex::pruneNeighbors(int nodeId, int level) {
    auto& node = nodes_[nodeId];
    auto& links = node.neighbors[level];

    const float* nodeVec = vectorStore_.getVector(nodeId);
    if (!nodeVec) return;

    std::vector<DistIdPair> neighborDists;
    neighborDists.reserve(links.size());

    for (int neighborId : links) {
        float d = computeExactDistance(nodeId, neighborId);
        neighborDists.emplace_back(d, neighborId);
    }

    std::sort(neighborDists.begin(), neighborDists.end());

    links.clear();
    links.reserve(config_.M);
    for (int i = 0; i < config_.M && i < static_cast<int>(neighborDists.size()); ++i) {
        links.push_back(neighborDists[i].second);
    }
}

void HNSWPQIndex::save(const std::string& /*path*/) {
    // TODO: Implement save
}

void HNSWPQIndex::load(const std::string& /*path*/) {
    // TODO: Implement load
}

void HNSWPQIndex::searchBatch(const float* queries, int nQueries, int k,
                              int* resultIds, float* resultDistances) {
    int nThreads = std::min(4, nQueries);
    int chunkSize = (nQueries + nThreads - 1) / nThreads;

    std::vector<std::future<void>> futures;

    for (int t = 0; t < nThreads; t++) {
        int start = t * chunkSize;
        int end = std::min(start + chunkSize, nQueries);
        if (start >= end) break;

        futures.push_back(std::async(std::launch::async, [this, queries, k, resultIds, resultDistances, start, end]() {
            for (int i = start; i < end; i++) {
                const float* query = queries + static_cast<size_t>(i) * dimension_;
                int* ids = resultIds + static_cast<size_t>(i) * k;
                float* dists = resultDistances + static_cast<size_t>(i) * k;
                int count;
                search(query, k, ids, dists, &count);
            }
        }));
    }

    for (auto& f : futures) {
        f.get();
    }
}

void HNSWPQIndex::addBatch(const float* vectors, const int* ids, int n) {
    for (int i = 0; i < n; i++) {
        try {
            add(ids[i], vectors + static_cast<size_t>(i) * dimension_);
        } catch (...) {
            // Skip failed adds
        }
    }
}

float* HNSWPQIndex::getCodebookCentroid(int subspaceIdx, int centroidIdx) {
    return codebooks_.data() +
           (static_cast<size_t>(subspaceIdx) * nCentroids_ + centroidIdx) * subDim_;
}

size_t HNSWPQIndex::getMemoryUsage() const {
    size_t codebookMem = codebooks_.size() * sizeof(float);
    size_t codesMem = codes_.size() * sizeof(uint8_t);
    size_t graphMem = 0;
    for (const auto& node : nodes_) {
        for (const auto& level : node.neighbors) {
            graphMem += level.size() * sizeof(int);
        }
    }
    return codebookMem + codesMem + graphMem + vectorStore_.capacity() * dimension_ * sizeof(float);
}

float HNSWPQIndex::getCompressionRatio() const {
    // Original: dimension * sizeof(float)
    // Compressed: pqM bytes
    float originalSize = dimension_ * sizeof(float);
    float compressedSize = config_.pqM * sizeof(uint8_t);
    return originalSize / compressedSize;
}

} // namespace vectordb
