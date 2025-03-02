# 向量数据库实现说明

## 项目概述

这是一个用Java实现的本地向量数据库，支持10万条数据，向量维度为1000，并使用优化的索引结构降低计算和存储量。

## 核心组件

### 1. 向量表示 (Vector)

- 使用`float[]`数组表示向量
- 支持向量的基本操作：欧几里得距离、余弦相似度、归一化等
- 缓存向量范数以提高计算效率

### 2. 索引结构 (HNSW)

采用HNSW（Hierarchical Navigable Small World）算法作为索引结构，具有以下特点：

- 多层图结构，每层是一个近似最近邻图
- 搜索复杂度从O(n)降低到O(log n)
- 支持高效的增量构建
- 参数可调整：
  - m: 每个节点的最大连接数
  - efConstruction: 构建时的搜索宽度
  - maxLevel: 最大层数

### 3. 存储层 (VectorStorage)

- 本地文件系统存储
- 使用Jackson进行序列化/反序列化
- 内存缓存减少磁盘IO
- 支持向量的增删改查操作

### 4. 数据库接口 (VectorDatabase)

- 提供简洁的API
- 使用Builder模式创建实例
- 管理索引和存储的协调

## 性能优化

### 1. 索引优化

- HNSW算法大幅降低搜索复杂度
- 多层图结构减少搜索空间
- 并发安全的数据结构

### 2. 存储优化

- 向量量化减少存储空间（可选）
- 内存缓存减少磁盘IO
- 批量操作提高吞吐量

### 3. 计算优化

- 缓存向量范数
- 提前终止搜索
- 并行计算（可扩展）

## 使用方法

```java
// 初始化向量数据库
VectorDatabase db = new VectorDatabase.Builder()
    .withDimension(1000)
    .withMaxElements(100000)
    .withStoragePath("./data")
    .build();

// 添加向量
int id = 1;
float[] vector = new float[1000]; // 填充向量数据
db.addVector(id, vector);

// 搜索最近邻
List<SearchResult> results = db.search(vector, 10); // 查找10个最近邻

// 关闭数据库
db.close();
```

## 性能指标

在标准测试环境下（具体硬件配置），性能表现如下：

- 插入性能：每秒可插入约X条向量
- 搜索性能：
  - Top-1搜索：每秒约X次查询
  - Top-10搜索：每秒约X次查询
  - Top-100搜索：每秒约X次查询
- 存储效率：每个1000维向量约占用X KB存储空间

## 未来改进

1. 支持向量的批量操作
2. 实现更高效的向量量化方法（如PQ量化）
3. 添加更多距离度量方式
4. 支持多线程并行搜索
5. 实现更高效的持久化策略 