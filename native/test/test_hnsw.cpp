#include <gtest/gtest.h>
#include "index/HNSWIndex.h"
#include "index/PQIndex.h"
#include "index/IVFIndex.h"
#include <vector>
#include <random>

using namespace vectordb;

class HNSWTest : public ::testing::Test {
protected:
    void SetUp() override {
        dimension = 128;
        nVectors = 1000;
        rng.seed(42);

        // Generate random vectors
        vectors.resize(nVectors);
        for (int i = 0; i < nVectors; i++) {
            vectors[i].resize(dimension);
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = dist(rng);
            }
        }
    }

    int dimension;
    int nVectors;
    std::vector<std::vector<float>> vectors;
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist{0.0f, 1.0f};
};

TEST_F(HNSWTest, BasicAddAndSearch) {
    HNSWIndex index(dimension, nVectors * 2);

    // Add vectors
    for (int i = 0; i < nVectors; i++) {
        index.add(i, vectors[i].data());
    }

    EXPECT_EQ(index.size(), nVectors);

    // Search
    std::vector<float> query = vectors[0];
    int k = 10;
    std::vector<int> ids(k);
    std::vector<float> distances(k);
    int count;

    index.search(query.data(), k, ids.data(), distances.data(), &count);

    EXPECT_GT(count, 0);
    EXPECT_EQ(ids[0], 0);  // First result should be the query itself
}

TEST_F(HNSWTest, BatchSearch) {
    HNSWIndex index(dimension, nVectors * 2);

    for (int i = 0; i < nVectors; i++) {
        index.add(i, vectors[i].data());
    }

    int nQueries = 10;
    int k = 5;
    std::vector<float> queries(nQueries * dimension);
    std::vector<int> resultIds(nQueries * k);
    std::vector<float> resultDistances(nQueries * k);

    for (int i = 0; i < nQueries; i++) {
        for (int j = 0; j < dimension; j++) {
            queries[i * dimension + j] = dist(rng);
        }
    }

    index.searchBatch(queries.data(), nQueries, k, resultIds.data(), resultDistances.data());

    // Verify results
    for (int i = 0; i < nQueries; i++) {
        EXPECT_GE(resultIds[i * k], 0);
    }
}

TEST_F(HNSWTest, PQIndexBasic) {
    PQIndex index(dimension, nVectors * 2);

    // Train
    std::vector<float> trainData;
    for (int i = 0; i < 100; i++) {
        trainData.insert(trainData.end(), vectors[i].begin(), vectors[i].end());
    }
    index.train(100, trainData.data());

    EXPECT_TRUE(index.isTrained());

    // Add vectors
    for (int i = 0; i < nVectors; i++) {
        index.add(i, vectors[i].data());
    }

    // Search
    std::vector<float> query = vectors[0];
    int k = 10;
    std::vector<int> ids(k);
    std::vector<float> distances(k);
    int count;

    index.search(query.data(), k, ids.data(), distances.data(), &count);

    EXPECT_GT(count, 0);
}

TEST_F(HNSWTest, IVFIndexBasic) {
    IVFIndex index(dimension, nVectors * 2);

    // Train
    std::vector<float> trainData;
    for (int i = 0; i < 500; i++) {
        trainData.insert(trainData.end(), vectors[i].begin(), vectors[i].end());
    }
    index.train(500, trainData.data());

    EXPECT_TRUE(index.isTrained());

    // Add vectors
    for (int i = 0; i < nVectors; i++) {
        index.add(i, vectors[i].data());
    }

    // Search
    std::vector<float> query = vectors[0];
    int k = 10;
    std::vector<int> ids(k);
    std::vector<float> distances(k);
    int count;

    index.search(query.data(), k, ids.data(), distances.data(), &count);

    EXPECT_GT(count, 0);
}
