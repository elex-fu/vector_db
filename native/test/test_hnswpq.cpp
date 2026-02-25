#include <gtest/gtest.h>
#include "index/HNSWPQIndex.h"
#include "index/HNSWIndex.h"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace vectordb;

class HNSWPQTest : public ::testing::Test {
protected:
    std::mt19937 rng{42};
    std::uniform_real_distribution<float> dist{0.0f, 1.0f};

    std::vector<float> generateRandomVector(int dim) {
        std::vector<float> vec(dim);
        for (int i = 0; i < dim; i++) {
            vec[i] = dist(rng);
        }
        return vec;
    }
};

TEST_F(HNSWPQTest, BasicAddAndSearch) {
    const int dim = 128;
    const int maxElements = 1000;

    HNSWPQIndex index(dim, maxElements);

    // Generate training data
    std::vector<float> trainData;
    for (int i = 0; i < 500; i++) {
        auto vec = generateRandomVector(dim);
        trainData.insert(trainData.end(), vec.begin(), vec.end());
    }

    // Train the index
    index.train(500, trainData.data());
    EXPECT_TRUE(index.isTrained());

    // Add vectors
    for (int i = 0; i < 100; i++) {
        auto vec = generateRandomVector(dim);
        index.add(i, vec.data());
    }

    EXPECT_EQ(index.size(), 100);

    // Search
    auto query = generateRandomVector(dim);
    std::vector<int> ids(10);
    std::vector<float> distances(10);
    int count;

    index.search(query.data(), 10, ids.data(), distances.data(), &count);

    EXPECT_GT(count, 0);
    EXPECT_LE(count, 10);

    // Verify distances are sorted
    for (int i = 1; i < count; i++) {
        EXPECT_LE(distances[i-1], distances[i]);
    }

    std::cout << "\nHNSWPQIndex Basic Test:" << std::endl;
    std::cout << "Compression ratio: " << index.getCompressionRatio() << "x" << std::endl;
    std::cout << "Memory usage: " << index.getMemoryUsage() / 1024 << " KB" << std::endl;
}

TEST_F(HNSWPQTest, PerformanceBenchmark) {
    const int dim = 128;
    const int maxElements = 10000;

    HNSWPQConfig config;
    config.pqM = 8;
    config.pqBits = 8;

    HNSWPQIndex index(dim, maxElements, config);

    std::cout << "\n========== HNSWPQIndex Performance ==========" << std::endl;

    // Training
    std::vector<float> trainData;
    for (int i = 0; i < 5000; i++) {
        auto vec = generateRandomVector(dim);
        trainData.insert(trainData.end(), vec.begin(), vec.end());
    }

    auto trainStart = std::chrono::high_resolution_clock::now();
    index.train(5000, trainData.data());
    auto trainEnd = std::chrono::high_resolution_clock::now();

    double trainTime = std::chrono::duration<double, std::milli>(trainEnd - trainStart).count();
    std::cout << "Training 5000 samples: " << trainTime << " ms" << std::endl;

    // Add vectors
    auto addStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5000; i++) {
        auto vec = generateRandomVector(dim);
        index.add(i, vec.data());
    }
    auto addEnd = std::chrono::high_resolution_clock::now();

    double addTime = std::chrono::duration<double, std::milli>(addEnd - addStart).count();
    std::cout << "Adding 5000 vectors: " << addTime << " ms ("
              << std::fixed << std::setprecision(3) << addTime / 5000 << " ms/vec)" << std::endl;

    // Search benchmark
    const int nQueries = 1000;
    std::vector<std::vector<float>> queries;
    for (int i = 0; i < nQueries; i++) {
        queries.push_back(generateRandomVector(dim));
    }

    // Warmup
    std::vector<int> ids(10);
    std::vector<float> distances(10);
    int count;
    for (int i = 0; i < 100; i++) {
        index.search(queries[i].data(), 10, ids.data(), distances.data(), &count);
    }

    // Benchmark
    auto searchStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nQueries; i++) {
        index.search(queries[i].data(), 10, ids.data(), distances.data(), &count);
    }
    auto searchEnd = std::chrono::high_resolution_clock::now();

    double searchTime = std::chrono::duration<double, std::milli>(searchEnd - searchStart).count();
    double qps = nQueries * 1000.0 / searchTime;

    std::cout << "Search " << nQueries << " queries: " << searchTime << " ms" << std::endl;
    std::cout << "QPS: " << std::fixed << std::setprecision(1) << qps << std::endl;

    // Memory stats
    std::cout << "\nMemory Statistics:" << std::endl;
    std::cout << "  Compression ratio: " << index.getCompressionRatio() << "x" << std::endl;
    std::cout << "  Total memory: " << index.getMemoryUsage() / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Memory per vector: " << index.getMemoryUsage() / index.size() << " bytes" << std::endl;
}

TEST_F(HNSWPQTest, CompareWithHNSW) {
    const int dim = 128;
    const int nVectors = 5000;

    // Build both indexes
    HNSWIndex hnswIndex(dim, nVectors + 100);
    HNSWPQIndex hnswPqIndex(dim, nVectors + 100);

    // Generate data
    std::vector<std::vector<float>> vectors;
    std::vector<float> trainData;
    for (int i = 0; i < nVectors; i++) {
        auto vec = generateRandomVector(dim);
        vectors.push_back(vec);

        if (i < 5000) {
            trainData.insert(trainData.end(), vec.begin(), vec.end());
        }
    }

    // Train PQ index
    hnswPqIndex.train(5000, trainData.data());

    // Add to both
    for (int i = 0; i < nVectors; i++) {
        hnswIndex.add(i, vectors[i].data());
        hnswPqIndex.add(i, vectors[i].data());
    }

    // Compare search results
    int correct = 0;
    int total = 0;

    for (int q = 0; q < 100; q++) {
        auto query = generateRandomVector(dim);

        std::vector<int> hnswIds(10), pqIds(10);
        std::vector<float> hnswDists(10), pqDists(10);
        int hnswCount, pqCount;

        hnswIndex.search(query.data(), 10, hnswIds.data(), hnswDists.data(), &hnswCount);
        hnswPqIndex.search(query.data(), 10, pqIds.data(), pqDists.data(), &pqCount);

        // Count overlap
        for (int i = 0; i < pqCount && i < 5; i++) {
            for (int j = 0; j < hnswCount && j < 5; j++) {
                if (pqIds[i] == hnswIds[j]) {
                    correct++;
                    break;
                }
            }
            total++;
        }
    }

    float recall = static_cast<float>(correct) / total;
    std::cout << "\nHNSWPQ vs HNSW Recall@5: " << std::fixed << std::setprecision(2)
              << recall * 100 << "%" << std::endl;

    // PQ should use less memory
    std::cout << "HNSW memory: " << nVectors * dim * sizeof(float) / 1024 << " KB (estimated)" << std::endl;
    std::cout << "HNSWPQ memory: " << hnswPqIndex.getMemoryUsage() / 1024 << " KB" << std::endl;
}
