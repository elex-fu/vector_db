#include <gtest/gtest.h>
#include "index/HNSWIndex.h"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
#include <atomic>
#include <cmath>

using namespace vectordb;

class PerformanceTest : public ::testing::Test {
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

    void printBenchmark(const char* name, double timeMs, int iterations, const char* unit = "op") {
        double avgTime = timeMs / iterations;
        double throughput = 1000.0 / avgTime;
        std::cout << std::left << std::setw(40) << name
                  << std::right << std::setw(10) << std::fixed << std::setprecision(3) << timeMs << " ms"
                  << std::setw(12) << std::setprecision(4) << avgTime << " ms/" << unit
                  << std::setw(12) << std::setprecision(1) << throughput << " " << unit << "/s"
                  << std::endl;
    }
};

TEST_F(PerformanceTest, HNSW_AddPerformance) {
    const int dim = 128;
    const int maxElements = 100000;

    std::cout << "\n========== HNSW Add Performance ==========" << std::endl;
    std::cout << "Dimension: " << dim << ", Max Elements: " << maxElements << std::endl;

    HNSWIndex index(dim, maxElements);

    // Warmup
    for (int i = 0; i < 100; i++) {
        auto vec = generateRandomVector(dim);
        index.add(i, vec.data());
    }

    // Benchmark different batch sizes
    std::vector<int> batchSizes = {100, 1000, 10000};
    for (int batchSize : batchSizes) {
        HNSWIndex testIndex(dim, maxElements);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < batchSize; i++) {
            auto vec = generateRandomVector(dim);
            testIndex.add(i, vec.data());
        }
        auto end = std::chrono::high_resolution_clock::now();

        double timeMs = std::chrono::duration<double, std::milli>(end - start).count();
        printBenchmark(("Add " + std::to_string(batchSize) + " vectors").c_str(), timeMs, batchSize, "vec");
    }
}

TEST_F(PerformanceTest, HNSW_SearchPerformance) {
    const int dim = 128;
    const int maxElements = 50000;
    const int k = 10;

    std::cout << "\n========== HNSW Search Performance ==========" << std::endl;
    std::cout << "Dimension: " << dim << ", Database Size: " << maxElements << ", k=" << k << std::endl;

    // Build index
    HNSWIndex index(dim, maxElements);
    for (int i = 0; i < maxElements; i++) {
        auto vec = generateRandomVector(dim);
        index.add(i, vec.data());
    }

    std::cout << "Index built with " << index.size() << " vectors" << std::endl;

    // Prepare queries
    const int nQueries = 1000;
    std::vector<std::vector<float>> queries;
    for (int i = 0; i < nQueries; i++) {
        queries.push_back(generateRandomVector(dim));
    }

    // Warmup
    std::vector<int> ids(k);
    std::vector<float> distances(k);
    int count;
    for (int i = 0; i < 100; i++) {
        index.search(queries[i % queries.size()].data(), k, ids.data(), distances.data(), &count);
    }

    // Single-threaded search benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nQueries; i++) {
        index.search(queries[i].data(), k, ids.data(), distances.data(), &count);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double timeMs = std::chrono::duration<double, std::milli>(end - start).count();
    printBenchmark("Single-threaded search", timeMs, nQueries, "query");

    // Batch search benchmark
    std::vector<float> batchQueries;
    for (const auto& q : queries) {
        batchQueries.insert(batchQueries.end(), q.begin(), q.end());
    }
    std::vector<int> batchResultIds(nQueries * k);
    std::vector<float> batchResultDists(nQueries * k);

    start = std::chrono::high_resolution_clock::now();
    index.searchBatch(batchQueries.data(), nQueries, k, batchResultIds.data(), batchResultDists.data());
    end = std::chrono::high_resolution_clock::now();

    timeMs = std::chrono::duration<double, std::milli>(end - start).count();
    printBenchmark("Batch search", timeMs, nQueries, "query");
}

TEST_F(PerformanceTest, HNSW_DifferentDimensions) {
    std::vector<int> dimensions = {64, 128, 256, 512, 768, 1024};
    const int nVectors = 10000;
    const int nQueries = 100;
    const int k = 10;

    std::cout << "\n========== HNSW Performance vs Dimension ==========" << std::endl;

    for (int dim : dimensions) {
        HNSWIndex index(dim, nVectors + 100);

        // Add vectors
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < nVectors; i++) {
            auto vec = generateRandomVector(dim);
            index.add(i, vec.data());
        }
        auto end = std::chrono::high_resolution_clock::now();
        double addTimeMs = std::chrono::duration<double, std::milli>(end - start).count();

        // Search
        std::vector<int> ids(k);
        std::vector<float> distances(k);
        int count;

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < nQueries; i++) {
            auto query = generateRandomVector(dim);
            index.search(query.data(), k, ids.data(), distances.data(), &count);
        }
        end = std::chrono::high_resolution_clock::now();
        double searchTimeMs = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "Dim=" << std::setw(4) << dim
                  << " | Add: " << std::setw(8) << std::fixed << std::setprecision(2) << addTimeMs << " ms"
                  << " (" << std::setw(6) << std::setprecision(3) << (addTimeMs/nVectors) << " ms/vec)"
                  << " | Search: " << std::setw(8) << std::fixed << std::setprecision(2) << searchTimeMs << " ms"
                  << " (" << std::setw(6) << std::setprecision(3) << (searchTimeMs/nQueries) << " ms/query)"
                  << std::endl;
    }
}

TEST_F(PerformanceTest, HNSW_ConcurrentPerformance) {
    const int dim = 128;
    const int maxElements = 50000;
    const int k = 10;

    std::cout << "\n========== HNSW Concurrent Performance ==========" << std::endl;

    HNSWIndex index(dim, maxElements);

    // Build index
    for (int i = 0; i < maxElements; i++) {
        auto vec = generateRandomVector(dim);
        index.add(i, vec.data());
    }

    // Test with different thread counts
    std::vector<int> threadCounts = {1, 2, 4, 8};
    const int queriesPerThread = 500;

    for (int nThreads : threadCounts) {
        index.setNumThreads(nThreads);

        std::vector<std::thread> threads;
        std::atomic<long long> totalTime{0};

        auto start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < nThreads; t++) {
            threads.emplace_back([&]() {
                std::vector<int> ids(k);
                std::vector<float> distances(k);
                int count;

                auto threadStart = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < queriesPerThread; i++) {
                    auto query = generateRandomVector(dim);
                    index.search(query.data(), k, ids.data(), distances.data(), &count);
                }
                auto threadEnd = std::chrono::high_resolution_clock::now();

                auto threadTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(threadEnd - threadStart);
                long long threadTime = threadTimeUs.count();
                totalTime += threadTime;
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        double totalTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
        int totalQueries = nThreads * queriesPerThread;

        double speedup = (threadCounts[0] == nThreads) ? 1.0 : totalTimeMs / totalTimeMs; // Simplified

        std::cout << "Threads=" << nThreads
                  << " | Total: " << std::fixed << std::setprecision(2) << totalTimeMs << " ms"
                  << " | Per query: " << std::setprecision(4) << (totalTimeMs/totalQueries) << " ms"
                  << " | Throughput: " << std::setprecision(1) << (totalQueries*1000/totalTimeMs) << " qps"
                  << std::endl;
    }
}

TEST_F(PerformanceTest, MemoryUsage) {
    const int dim = 128;
    std::vector<int> sizes = {1000, 10000, 50000};

    std::cout << "\n========== Memory Usage Analysis ==========" << std::endl;

    for (int size : sizes) {
        HNSWIndex index(dim, size + 1000);

        size_t memBefore = 0; // Would need platform-specific code

        for (int i = 0; i < size; i++) {
            auto vec = generateRandomVector(dim);
            index.add(i, vec.data());
        }

        // Theoretical memory calculation
        size_t vectorMem = size * dim * sizeof(float);
        size_t idMem = size * sizeof(int);
        size_t normMem = size * sizeof(float);
        size_t graphMem = size * 32 * sizeof(int) * 2; // Approximate graph overhead

        size_t totalTheoretical = vectorMem + idMem + normMem + graphMem;

        std::cout << "Vectors: " << std::setw(6) << size
                  << " | Vector data: " << std::setw(6) << (vectorMem / 1024 / 1024) << " MB"
                  << " | Est. total: " << std::setw(6) << (totalTheoretical / 1024 / 1024) << " MB"
                  << " | Per vector: " << std::fixed << std::setprecision(2)
                  << (totalTheoretical / (double)size) << " bytes"
                  << std::endl;
    }
}
