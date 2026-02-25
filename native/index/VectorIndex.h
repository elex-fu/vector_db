#pragma once
#include <string>

namespace vectordb {

/**
 * 向量索引基类接口
 */
class VectorIndex {
public:
    virtual ~VectorIndex() = default;

    // 添加向量
    virtual void add(int id, const float* vector) = 0;

    // 搜索最近邻
    virtual void search(const float* query, int k,
                       int* resultIds, float* resultDistances,
                       int* resultCount) = 0;

    // 保存索引
    virtual void save(const std::string& path) = 0;

    // 加载索引
    virtual void load(const std::string& path) = 0;

    // 获取当前向量数量
    virtual int size() const = 0;

    // 获取向量维度
    virtual int dimension() const = 0;

    // 获取最大容量
    virtual int capacity() const = 0;
};

} // namespace vectordb
