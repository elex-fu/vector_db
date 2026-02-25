#include "IVFIndex.h"
#include <stdexcept>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>

namespace vectordb {

IVFIndex::IVFIndex(int dimension, int maxElements)
    : IVFIndex(dimension, maxElements, IVFConfig{}) {}

IVFIndex::IVFIndex(int dimension, int maxElements, const IVFConfig& config)
    : vectorStore_(dimension, maxElements), config_(config) {
    distanceFunc_ = getEuclideanDistanceFunc();
    centroids_.resize(static_cast<size_t>(config.nLists) * dimension);
    invertedLists_.resize(config.nLists);
    idToList_.resize(maxElements, -1);
}

void IVFIndex::train(int nSamples, const float* samples) {
    if (nSamples <= 0 || samples == nullptr) {
        throw std::invalid_argument("Invalid training samples");
    }

    const int dim = vectorStore_.dimension();
    const int nLists = config_.nLists;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, nSamples - 1);

    for (int i = 0; i < nLists; i++) {
        int sampleIdx = dist(rng);
        std::copy(samples + static_cast<size_t>(sampleIdx) * dim,
                  samples + static_cast<size_t>(sampleIdx + 1) * dim,
                  centroids_.begin() + static_cast<size_t>(i) * dim);
    }

    std::vector<int> assignments(nSamples);
    std::vector<int> clusterSizes(nLists);

    for (int iter = 0; iter < config_.maxIterations; iter++) {
        bool changed = false;
        for (int i = 0; i < nSamples; i++) {
            const float* sample = samples + static_cast<size_t>(i) * dim;
            int nearest = findNearestCentroid(sample);
            if (assignments[i] != nearest) {
                assignments[i] = nearest;
                changed = true;
            }
        }

        if (!changed) break;

        std::fill(centroids_.begin(), centroids_.end(), 0.0f);
        std::fill(clusterSizes.begin(), clusterSizes.end(), 0);

        for (int i = 0; i < nSamples; i++) {
            int cluster = assignments[i];
            float* centroid = centroids_.data() + static_cast<size_t>(cluster) * dim;
            const float* sample = samples + static_cast<size_t>(i) * dim;

            for (int d = 0; d < dim; d++) {
                centroid[d] += sample[d];
            }
            clusterSizes[cluster]++;
        }

        for (int i = 0; i < nLists; i++) {
            if (clusterSizes[i] > 0) {
                float* centroid = centroids_.data() + static_cast<size_t>(i) * dim;
                float invSize = 1.0f / clusterSizes[i];
                for (int d = 0; d < dim; d++) {
                    centroid[d] *= invSize;
                }
            }
        }
    }

    trained_ = true;
}

void IVFIndex::add(int id, const float* vector) {
    if (!trained_) {
        throw std::runtime_error("IVF index must be trained before adding vectors");
    }

    int index = vectorStore_.size();
    int listId = findNearestCentroid(vector);

    vectorStore_.add(id, vector);
    invertedLists_[listId].push_back(index);
    idToList_[index] = listId;
    size_++;
}

void IVFIndex::addBatch(const float* vectors, const int* ids, int n) {
    if (!trained_) {
        throw std::runtime_error("IVF index must be trained before adding vectors");
    }

    const int dim = vectorStore_.dimension();

    for (int i = 0; i < n; i++) {
        const float* vec = vectors + static_cast<size_t>(i) * dim;
        add(ids[i], vec);
    }
}

void IVFIndex::search(const float* query, int k,
                     int* resultIds, float* resultDistances,
                     int* resultCount) {
    if (!trained_) {
        *resultCount = 0;
        return;
    }

    const int dim = vectorStore_.dimension();

    std::vector<std::pair<float, int>> centroidDists;
    for (int i = 0; i < config_.nLists; i++) {
        float dist = distanceFunc_(query, centroids_.data() + static_cast<size_t>(i) * dim, dim);
        centroidDists.emplace_back(dist, i);
    }

    std::partial_sort(centroidDists.begin(),
                      centroidDists.begin() + config_.nProbes,
                      centroidDists.end());

    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(k * 10);

    for (int p = 0; p < config_.nProbes && p < config_.nLists; p++) {
        int listId = centroidDists[p].second;
        const auto& list = invertedLists_[listId];

        for (int idx : list) {
            const float* vec = vectorStore_.getVector(idx);
            float dist = distanceFunc_(query, vec, dim);
            candidates.emplace_back(dist, vectorStore_.getId(idx));
        }
    }

    if (candidates.size() > static_cast<size_t>(k)) {
        std::partial_sort(candidates.begin(),
                          candidates.begin() + k,
                          candidates.end());
        candidates.resize(k);
    } else {
        std::sort(candidates.begin(), candidates.end());
    }

    int count = std::min(k, static_cast<int>(candidates.size()));
    for (int i = 0; i < count; i++) {
        resultDistances[i] = candidates[i].first;
        resultIds[i] = candidates[i].second;
    }
    *resultCount = count;
}

int IVFIndex::findNearestCentroid(const float* vector) {
    const int dim = vectorStore_.dimension();
    int nearest = 0;
    float minDist = std::numeric_limits<float>::max();

    for (int i = 0; i < config_.nLists; i++) {
        float dist = distanceFunc_(vector, centroids_.data() + static_cast<size_t>(i) * dim, dim);
        if (dist < minDist) {
            minDist = dist;
            nearest = i;
        }
    }

    return nearest;
}

void IVFIndex::save(const std::string& /*path*/) {}
void IVFIndex::load(const std::string& /*path*/) {}

} // namespace vectordb