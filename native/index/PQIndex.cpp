#include "PQIndex.h"
#include <stdexcept>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <future>

namespace vectordb {

PQIndex::PQIndex(int dimension, int maxElements)
    : PQIndex(dimension, maxElements, PQConfig{}) {}

PQIndex::PQIndex(int dimension, int maxElements, const PQConfig& config)
    : vectorStore_(dimension, maxElements), config_(config) {

    if (dimension % config.M != 0) {
        throw std::invalid_argument("Dimension must be divisible by M");
    }

    subDim_ = dimension / config.M;
    nCentroids_ = 1 << config.nBits;

    codebooks_.resize(static_cast<size_t>(config.M) * nCentroids_ * subDim_);
    distanceFunc_ = getEuclideanDistanceFunc();
}

void PQIndex::train(int nSamples, const float* samples) {
    if (nSamples <= 0 || samples == nullptr) {
        throw std::invalid_argument("Invalid training samples");
    }

    for (int m = 0; m < config_.M; m++) {
        trainSubspace(m, nSamples, samples);
    }

    trained_ = true;
}

void PQIndex::trainSubspace(int subspaceIdx, int nSamples, const float* samples) {
    const int subDim = subDim_;
    const int nCentroids = nCentroids_;
    const int dim = vectorStore_.dimension();

    std::vector<float> subData(static_cast<size_t>(nSamples) * subDim);
    for (int i = 0; i < nSamples; i++) {
        const float* vec = samples + static_cast<size_t>(i) * dim + static_cast<size_t>(subspaceIdx) * subDim;
        std::copy(vec, vec + subDim, subData.data() + static_cast<size_t>(i) * subDim);
    }

    std::mt19937 rng(42 + subspaceIdx);
    std::uniform_int_distribution<int> dist(0, nSamples - 1);

    for (int i = 0; i < nCentroids; i++) {
        int sampleIdx = dist(rng);
        float* centroid = getCodebookCentroid(subspaceIdx, i);
        const float* sample = subData.data() + static_cast<size_t>(sampleIdx) * subDim;
        std::copy(sample, sample + subDim, centroid);
    }

    std::vector<int> assignments(nSamples);
    std::vector<int> clusterSizes(nCentroids);

    for (int iter = 0; iter < config_.maxIterations; iter++) {
        bool changed = false;
        for (int i = 0; i < nSamples; i++) {
            const float* sample = subData.data() + static_cast<size_t>(i) * subDim;
            int nearest = findNearestCentroid(subspaceIdx, sample);
            if (assignments[i] != nearest) {
                assignments[i] = nearest;
                changed = true;
            }
        }

        if (!changed) break;

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

int PQIndex::findNearestCentroid(int subspaceIdx, const float* subVector) {
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

void PQIndex::add(int id, const float* vector) {
    if (!trained_) {
        throw std::runtime_error("PQ index must be trained before adding vectors");
    }

    std::vector<uint8_t> codes(config_.M);
    encode(vector, codes.data());

    int index = vectorStore_.add(id, vector);

    codes_.resize(static_cast<size_t>(index + 1) * config_.M);
    std::copy(codes.begin(), codes.end(), codes_.data() + static_cast<size_t>(index) * config_.M);

    size_++;
}

void PQIndex::encode(const float* vector, uint8_t* codes) {
    for (int m = 0; m < config_.M; m++) {
        const float* subVector = vector + static_cast<size_t>(m) * subDim_;
        codes[m] = static_cast<uint8_t>(findNearestCentroid(m, subVector));
    }
}

void PQIndex::search(const float* query, int k,
                    int* resultIds, float* resultDistances,
                    int* resultCount) {
    if (!trained_) {
        *resultCount = 0;
        return;
    }

    std::vector<float> distanceTable(static_cast<size_t>(config_.M) * nCentroids_);
    BatchDistanceFunc batchDistFunc = getBatchEuclideanDistanceFunc();

    for (int m = 0; m < config_.M; m++) {
        const float* querySub = query + static_cast<size_t>(m) * subDim_;
        float* distTableSub = distanceTable.data() + static_cast<size_t>(m) * nCentroids_;
        const float* codebookStart = codebooks_.data() +
            static_cast<size_t>(m) * nCentroids_ * subDim_;
        batchDistFunc(querySub, codebookStart, nCentroids_, subDim_, distTableSub);
    }

    std::vector<std::pair<float, int>> distances;
    distances.reserve(size_);

    std::vector<const float*> distTableRows(config_.M);
    for (int m = 0; m < config_.M; m++) {
        distTableRows[m] = distanceTable.data() + static_cast<size_t>(m) * nCentroids_;
    }

    const int blockSize = 256;
    for (int blockStart = 0; blockStart < size_; blockStart += blockSize) {
        int blockEnd = std::min(blockStart + blockSize, size_);

        for (int i = blockStart; i < blockEnd; i++) {
            float dist = 0.0f;
            const uint8_t* codePtr = codes_.data() + static_cast<size_t>(i) * config_.M;

            int m = 0;
            for (; m + 8 <= config_.M; m += 8) {
                dist += distTableRows[m][codePtr[m]];
                dist += distTableRows[m + 1][codePtr[m + 1]];
                dist += distTableRows[m + 2][codePtr[m + 2]];
                dist += distTableRows[m + 3][codePtr[m + 3]];
                dist += distTableRows[m + 4][codePtr[m + 4]];
                dist += distTableRows[m + 5][codePtr[m + 5]];
                dist += distTableRows[m + 6][codePtr[m + 6]];
                dist += distTableRows[m + 7][codePtr[m + 7]];
            }
            for (; m < config_.M; m++) {
                dist += distTableRows[m][codePtr[m]];
            }

            distances.emplace_back(dist, vectorStore_.getId(i));
        }
    }

    if (distances.size() > static_cast<size_t>(k)) {
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
        distances.resize(k);
    } else {
        std::sort(distances.begin(), distances.end());
    }

    int count = std::min(k, static_cast<int>(distances.size()));
    for (int i = 0; i < count; i++) {
        resultDistances[i] = distances[i].first;
        resultIds[i] = distances[i].second;
    }
    *resultCount = count;
}

void PQIndex::addBatch(const float* vectors, const int* ids, int n) {
    if (!trained_) {
        throw std::runtime_error("PQ index must be trained before adding vectors");
    }

    const int dim = vectorStore_.dimension();

    std::vector<std::vector<uint8_t>> batchCodes(n);
    for (int i = 0; i < n; i++) {
        batchCodes[i].resize(config_.M);
    }

    int nThreads = std::min(4, n);
    int chunkSize = (n + nThreads - 1) / nThreads;
    std::vector<std::future<void>> futures;

    for (int t = 0; t < nThreads; t++) {
        int start = t * chunkSize;
        int end = std::min(start + chunkSize, n);
        if (start >= end) break;

        futures.push_back(std::async(std::launch::async, [this, vectors, &batchCodes, dim, start, end]() {
            for (int i = start; i < end; i++) {
                encode(vectors + static_cast<size_t>(i) * dim, batchCodes[i].data());
            }
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    for (int i = 0; i < n; i++) {
        const float* vec = vectors + static_cast<size_t>(i) * dim;
        int index = vectorStore_.add(ids[i], vec);

        codes_.resize(static_cast<size_t>(index + 1) * config_.M);
        std::copy(batchCodes[i].begin(), batchCodes[i].end(),
                  codes_.data() + static_cast<size_t>(index) * config_.M);
        size_++;
    }
}

void PQIndex::searchBatch(const float* queries, int nQueries, int k,
                         int* resultIds, float* resultDistances) {
    if (!trained_) {
        for (int i = 0; i < nQueries; i++) {
            int* ids = resultIds + static_cast<size_t>(i) * k;
            float* dists = resultDistances + static_cast<size_t>(i) * k;
            for (int j = 0; j < k; j++) {
                ids[j] = -1;
                dists[j] = -1.0f;
            }
        }
        return;
    }

    const int dim = vectorStore_.dimension();

    int nThreads = std::min(4, nQueries);
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

void PQIndex::save(const std::string& /*path*/) {}
void PQIndex::load(const std::string& /*path*/) {}

} // namespace vectordb
