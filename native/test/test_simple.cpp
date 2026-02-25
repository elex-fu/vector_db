#include <gtest/gtest.h>
#include "index/HNSWIndex.h"
#include <vector>
#include <random>

using namespace vectordb;

TEST(HNSWSimpleTest, AddOneVector) {
    HNSWIndex index(128, 100);

    std::vector<float> vec(128, 0.5f);
    index.add(0, vec.data());

    EXPECT_EQ(index.size(), 1);
}

TEST(HNSWSimpleTest, AddMultipleVectors) {
    HNSWIndex index(128, 1000);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < 10; i++) {
        std::vector<float> vec(128);
        for (int j = 0; j < 128; j++) {
            vec[j] = dist(rng);
        }
        index.add(i, vec.data());
    }

    EXPECT_EQ(index.size(), 10);
}

TEST(HNSWSimpleTest, SearchOnEmptyIndex) {
    HNSWIndex index(128, 100);

    std::vector<float> query(128, 0.5f);
    int k = 5;
    std::vector<int> ids(k);
    std::vector<float> distances(k);
    int count;

    index.search(query.data(), k, ids.data(), distances.data(), &count);

    EXPECT_EQ(count, 0);
}