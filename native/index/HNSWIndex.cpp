#include "HNSWIndex.h"
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

HNSWIndex::HNSWIndex(int dimension, int maxElements)
    : HNSWIndex(dimension, maxElements, HNSWConfig{}) {}

HNSWIndex::HNSWIndex(int dimension, int maxElements, const HNSWConfig& config)
    : vectorStore_(dimension, maxElements), config_(config) {
    distanceFunc_ = getEuclideanDistanceFunc();
    nodes_.reserve(maxElements);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    rng_.seed(seed);
}

void HNSWIndex::add(int id, const float* vector) {
    int newIndex = vectorStore_.size();
    vectorStore_.add(id, vector);

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
    float currDist = computeDistance(vector, currObj);

    for (int level = nodes_[currObj].level; level > newNode.level; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            std::shared_lock<std::shared_mutex> readLock(mutex_);
            const auto& neighbors = nodes_[currObj].neighbors[level];
            readLock.unlock();

            for (int neighbor : neighbors) {
                float d = computeDistance(vector, neighbor);
                if (d < currDist) {
                    currDist = d;
                    currObj = neighbor;
                    changed = true;
                }
            }
        }
    }

    int maxLevelToProcess = std::min(newNode.level, nodes_[currObj].level);
    for (int level = maxLevelToProcess; level >= 0; level--) {
        std::vector<DistIdPair> results;
        int efBuild = config_.efConstruction;
        if (level > 0 && size_ > 1000) {
            efBuild = std::max(config_.M * 2, static_cast<int>(config_.efConstruction * 0.8));
        }
        searchLevel(vector, currObj, efBuild, level, results);

        std::vector<int> selectedNeighbors;
        if (config_.useHeuristicSelection && results.size() > static_cast<size_t>(config_.M)) {
            selectedNeighbors = selectNeighborsHeuristic(vector, results, config_.M, level);
        } else {
            selectedNeighbors = selectNeighbors(results, config_.M);
        }
        newNode.neighbors[level] = selectedNeighbors;
        connectNeighbors(newIndex, selectedNeighbors, level);

        if (!results.empty()) {
            currObj = results[0].second;
        }
    }

    int currentEntry = entryPoint_.load(std::memory_order_acquire);
    if (newNode.level > nodes_[currentEntry].level) {
        entryPoint_.store(newIndex, std::memory_order_release);
    }

    nodes_.push_back(std::move(newNode));
    size_.fetch_add(1, std::memory_order_release);
}

void HNSWIndex::search(const float* query, int k,
                       int* resultIds, float* resultDistances,
                       int* resultCount) {
    if (size_.load() == 0) {
        *resultCount = 0;
        return;
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);

    int currObj = entryPoint_.load(std::memory_order_acquire);
    float currDist = computeDistance(query, currObj);

    int maxLevel = nodes_[currObj].level;

    for (int level = maxLevel; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            const auto& neighbors = nodes_[currObj].neighbors[level];
            for (int neighbor : neighbors) {
                float d = computeDistance(query, neighbor);
                if (d < currDist) {
                    currDist = d;
                    currObj = neighbor;
                    changed = true;
                }
            }
        }
    }

    std::vector<DistIdPair> results;
    int dataSize = size_.load(std::memory_order_acquire);
    int efSearch = config_.getEfSearch(k, dataSize);
    searchLevel(query, currObj, efSearch, 0, results);

    int count = std::min(k, static_cast<int>(results.size()));
    for (int i = 0; i < count; i++) {
        resultDistances[i] = results[i].first;
        resultIds[i] = vectorStore_.getId(results[i].second);
    }
    *resultCount = count;
}

void HNSWIndex::searchLevel(const float* query, int entryPoint, int ef, int level,
                            std::vector<DistIdPair>& results) {
    std::priority_queue<DistIdPair, std::vector<DistIdPair>, CompareByFirst> candidates;
    std::priority_queue<DistIdPair> bestResults;
    std::vector<bool> visited(size_, false);

    float dist = computeDistance(query, entryPoint);
    candidates.emplace(dist, entryPoint);
    bestResults.emplace(dist, entryPoint);
    visited[entryPoint] = true;

    float lowerBound = dist;
    int expansionCount = 0;
    const int maxExpansions = config_.getMaxExpansions(ef);

    while (!candidates.empty()) {
        auto curr = candidates.top();
        candidates.pop();
        expansionCount++;

        if (curr.first > lowerBound && bestResults.size() >= static_cast<size_t>(ef)) {
            break;
        }

        if (config_.useEarlyTermination && expansionCount > maxExpansions) {
            break;
        }

        if (config_.distanceThreshold > 0 && curr.first > config_.distanceThreshold) {
            break;
        }

        const auto& neighbors = nodes_[curr.second].neighbors[level];
        for (size_t ni = 0; ni < std::min(size_t(4), neighbors.size()); ++ni) {
            vectorStore_.prefetchVector(neighbors[ni]);
        }

        for (size_t ni = 0; ni < neighbors.size(); ++ni) {
            int neighbor = neighbors[ni];

            if (ni + 4 < neighbors.size()) {
                vectorStore_.prefetchVector(neighbors[ni + 4]);
            }

            if (visited[neighbor]) continue;
            visited[neighbor] = true;

            float d = computeDistance(query, neighbor);

            if (config_.distanceThreshold > 0 && d > config_.distanceThreshold) {
                continue;
            }

            if (bestResults.size() < static_cast<size_t>(ef) || d < lowerBound) {
                candidates.emplace(d, neighbor);
                bestResults.emplace(d, neighbor);

                if (bestResults.size() > static_cast<size_t>(ef)) {
                    bestResults.pop();
                }

                if (!bestResults.empty()) {
                    lowerBound = bestResults.top().first;
                }
            }
        }
    }

    results.clear();
    while (!bestResults.empty()) {
        results.push_back(bestResults.top());
        bestResults.pop();
    }
    std::reverse(results.begin(), results.end());
}

std::vector<int> HNSWIndex::selectNeighbors(const std::vector<DistIdPair>& candidates, int M) {
    std::vector<int> result;
    result.reserve(M);

    int count = std::min(M, static_cast<int>(candidates.size()));
    for (int i = 0; i < count; i++) {
        result.push_back(candidates[i].second);
    }

    return result;
}

std::vector<int> HNSWIndex::selectNeighborsHeuristic(const float* query,
                                                     const std::vector<DistIdPair>& candidates,
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

    const size_t maxCandidates = std::min(static_cast<size_t>(M * 8), candidates.size());
    std::vector<bool> selected(maxCandidates, false);

    for (int i = 0; i < M && i < static_cast<int>(maxCandidates); ++i) {
        int bestIdx = -1;
        float bestScore = -1.0f;

        for (size_t j = 0; j < maxCandidates; ++j) {
            if (selected[j]) continue;

            int candidateId = candidates[j].second;
            float candidateDist = candidates[j].first;

            float minDistToSelected = std::numeric_limits<float>::max();
            for (int selectedId : result) {
                const float* selectedVec = vectorStore_.getVector(selectedId);
                const float* candidateVec = vectorStore_.getVector(candidateId);
                float d = distanceFunc_(selectedVec, candidateVec, vectorStore_.dimension());
                minDistToSelected = std::min(minDistToSelected, d);
            }

            float score;
            if (result.empty()) {
                score = 1.0f / (1.0f + candidateDist);
            } else {
                score = (1.0f / (1.0f + candidateDist)) + 0.5f * minDistToSelected;
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

void HNSWIndex::connectNeighbors(int newId, const std::vector<int>& neighbors, int level) {
    for (int neighbor : neighbors) {
        auto& neighborNode = nodes_[neighbor];
        auto& neighborLinks = neighborNode.neighbors[level];

        neighborLinks.push_back(newId);

        if (neighborLinks.size() > static_cast<size_t>(config_.M)) {
            pruneNeighbors(neighbor, level);
        }
    }
}

void HNSWIndex::pruneNeighbors(int nodeId, int level) {
    auto& node = nodes_[nodeId];
    auto& links = node.neighbors[level];

    if (links.size() <= static_cast<size_t>(config_.M * config_.pruneOverflowFactor)) {
        return;
    }

    const float* nodeVec = vectorStore_.getVector(nodeId);

    std::vector<DistIdPair> neighborDists;
    neighborDists.reserve(links.size());
    for (int neighborId : links) {
        const float* neighborVec = vectorStore_.getVector(neighborId);
        float d = distanceFunc_(nodeVec, neighborVec, vectorStore_.dimension());
        neighborDists.emplace_back(d, neighborId);
    }

    std::sort(neighborDists.begin(), neighborDists.end());

    links.clear();
    links.reserve(config_.M);
    for (int i = 0; i < config_.M && i < static_cast<int>(neighborDists.size()); ++i) {
        links.push_back(neighborDists[i].second);
    }
}

int HNSWIndex::getRandomLevel() {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = distribution(rng_);
    int level = static_cast<int>(-std::log(r) * config_.levelMultiplier);
    return std::min(level, config_.maxLevel);
}

float HNSWIndex::computeDistance(const float* a, int bIndex) {
    const float* b = vectorStore_.getVector(bIndex);
    return distanceFunc_(a, b, vectorStore_.dimension());
}

void HNSWIndex::save(const std::string& /*path*/) {
    // TODO: Implement save
}

void HNSWIndex::load(const std::string& /*path*/) {
    // TODO: Implement load
}

void HNSWIndex::searchBatch(const float* queries, int nQueries, int k,
                            int* resultIds, float* resultDistances) {
    int dim = vectorStore_.dimension();
    int nThreads = std::min(numThreads_, nQueries);
    int chunkSize = (nQueries + nThreads - 1) / nThreads;

    std::vector<std::future<void>> futures;

    for (int t = 0; t < nThreads; t++) {
        int start = t * chunkSize;
        int end = std::min(start + chunkSize, nQueries);

        if (start >= end) break;

        futures.push_back(std::async(std::launch::async, [this, queries, k, resultIds, resultDistances, dim, start, end]() {
            for (int i = start; i < end; i++) {
                const float* query = queries + static_cast<size_t>(i) * dim;
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

void HNSWIndex::addBatch(const float* vectors, const int* ids, int n,
                        int* failedIndices, int* failedCount) {
    if (failedCount) *failedCount = 0;

    int dim = vectorStore_.dimension();

    for (int i = 0; i < n; i++) {
        try {
            add(ids[i], vectors + static_cast<size_t>(i) * dim);
        } catch (...) {
            if (failedIndices && failedCount) {
                failedIndices[*failedCount] = i;
                (*failedCount)++;
            }
        }
    }
}

void HNSWIndex::setNumThreads(int numThreads) {
    numThreads_ = std::max(1, numThreads);
}

int HNSWIndex::getNumThreads() const {
    return numThreads_;
}

} // namespace vectordb
