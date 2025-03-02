# 向量数据库

这是一个用Java实现的本地向量数据库，支持10万条数据，向量维度为1000，并使用优化的索引结构降低计算和存储量。

## 特性

- 支持高维向量（1000维）的存储和检索
- 支持多种索引算法：
  - HNSW（Hierarchical Navigable Small World）算法
  - ANNOY（Approximate Nearest Neighbors Oh Yeah）算法
  - LSH（Locality-Sensitive Hashing）算法
  - IVF（Inverted File）算法
  - PQ（Product Quantization）算法
- 索引优化功能，支持批量添加和索引重建
- 本地持久化存储
- 支持向量的增删改查操作
- 优化的内存使用和计算效率

## 项目结构

- `core`: 核心数据结构和接口
- `index`: 索引实现
  - `HnswIndex`: HNSW算法实现
  - `AnnoyIndex`: ANNOY算法实现
  - `LshIndex`: LSH算法实现
  - `IvfIndex`: IVF算法实现
  - `PqIndex`: PQ算法实现
- `storage`: 数据持久化
- `util`: 工具类

## 使用方法

```java
// 初始化向量数据库（默认使用HNSW索引）
VectorDatabase db = new VectorDatabase.Builder()
    .withDimension(1000)
    .withMaxElements(100000)
    .withStoragePath("./data")
    .build();

// 使用其他索引类型初始化数据库
VectorDatabase annoyDb = new VectorDatabase.Builder()
    .withDimension(1000)
    .withMaxElements(100000)
    .withStoragePath("./data_annoy")
    .withIndexType(IndexType.ANNOY)
    .build();

// 使用LSH索引
VectorDatabase lshDb = new VectorDatabase.Builder()
    .withDimension(1000)
    .withMaxElements(100000)
    .withStoragePath("./data_lsh")
    .withIndexType(IndexType.LSH)
    .build();

// 添加向量
int id = 1;
float[] vector = new float[1000]; // 填充向量数据
db.addVector(id, vector);

// 搜索最近邻
List<SearchResult> results = db.search(vector, 10); // 查找10个最近邻

// 重建索引（在批量添加或删除向量后优化搜索效率）
db.rebuildIndex();

// 关闭数据库
db.close();
```

## 性能优化

- 使用HNSW算法降低搜索复杂度，从O(n)降低到O(log n)
- 向量量化减少存储空间
- 批量操作提高吞吐量
- 多级缓存减少磁盘IO

## 索引对比

本项目实现了五种主流的向量索引方法：HNSW、ANNOY、LSH、IVF和PQ。以下是它们的性能对比：

### 1. 内存占用

| 索引类型 | 内存占用特点 | 相对大小 |
|---------|------------|---------|
| HNSW    | 存储完整的多层图结构，每节点最多16个连接 | 较大 (约为ANNOY的1.5-2倍) |
| ANNOY   | 存储多棵二叉树结构，默认10棵树 | 中等 |
| LSH     | 存储多个哈希表和哈希函数 | 中等 |
| IVF     | 存储聚类中心和倒排列表 | 较小 |
| PQ      | 存储子空间聚类中心和量化编码 | 非常小 (可压缩至原始大小的1/8-1/16) |

### 2. 添加速度

| 索引类型 | 单个添加 | 批量添加 | 10000个向量添加时间 |
|---------|---------|---------|-------------------|
| HNSW    | 较慢    | 不支持原生批量添加 | 2662毫秒 |
| ANNOY   | 较快    | 支持高效批量添加 | 93237毫秒（包含重建时间） |
| LSH     | 快      | 支持批量添加 | 9423毫秒 |
| IVF     | 快      | 支持批量添加 | 1307毫秒 |
| PQ      | 快      | 支持批量添加 | 1277毫秒 |

### 3. 查询性能

| 索引类型 | 准确度 | 查询速度 | 搜索时间 |
|---------|-------|---------|------------------|
| HNSW    | 很高  | 非常快   | <1毫秒 |
| ANNOY   | 中等  | 较快     | 37毫秒（优化后约26毫秒） |
| LSH     | 中等  | 快       | 1毫秒 |
| IVF     | 高    | 快       | 7毫秒 |
| PQ      | 中低  | 中等     | 38毫秒 |

### 4. 删除性能

| 索引类型 | 删除操作复杂度 | 删除速度 |
|---------|--------------|---------|
| HNSW    | 较复杂，需更新图连接 | <1毫秒 |
| ANNOY   | 较简单，标记删除即可 | <1毫秒 |
| LSH     | 简单，从哈希桶中移除 | 1毫秒 |
| IVF     | 简单，从聚类中移除 | <1毫秒 |
| PQ      | 简单，移除编码 | <1毫秒 |

### 5. 索引重建性能

| 索引类型 | 100个向量重建时间 | 10000个向量重建时间 |
|---------|-----------------|-------------------|
| HNSW    | 1毫秒 | 1,118毫秒 |
| ANNOY   | 110毫秒 | 17,335毫秒 |
| LSH     | 83毫秒 | 8,613毫秒 |
| IVF     | 93毫秒 | 7,175毫秒 |
| PQ      | <1毫秒（数量太少未进行量化） | 13,372毫秒 |

### 适用场景

#### HNSW适用场景
- 高精度要求的应用（医疗图像检索、人脸识别）
- 查询频繁的应用（搜索引擎、实时推荐系统）
- 数据相对稳定的场景（主要以查询为主）
- 内存资源充足的环境（服务器端应用、云计算环境）

#### ANNOY适用场景
- 需要持久化索引结构的场景
- 对查询速度要求适中的应用（内容推荐、相似商品查找）
- 资源受限的环境（移动设备、嵌入式系统）

#### LSH适用场景
- 大规模数据集（需处理海量向量数据）
- 可接受近似结果的应用（相似图片检索、重复检测）
- 需要快速添加和删除的动态数据集
- 对内存消耗有一定限制的环境

#### IVF适用场景
- 大规模数据集（百万级以上）
- 需要平衡精度和效率的应用
- 数据分布相对均匀的场景
- 需要快速添加新数据的场景

#### PQ适用场景
- 超大规模数据集（千万级以上）
- 内存严重受限的环境
- 可接受一定精度损失的应用
- 存储空间受限的场景（边缘设备、移动应用）

## 性能比较总结

| 索引类型 | 索引重建时间 | 搜索时间 | 内存占用 | 适用场景 |
|---------|------------|---------|---------|---------|
| HNSW    | 非常快      | 非常快   | 较大     | 高精度搜索，中小规模数据集 |
| ANNOY   | 慢         | 中等     | 中等     | 平衡精度和效率，可持久化 |
| LSH     | 中等       | 快       | 中等     | 大规模数据集的近似搜索 |
| IVF     | 中等       | 快       | 较小     | 大规模数据集的高效搜索 |
| PQ      | 慢（大数据集）| 中等    | 非常小   | 超大规模数据集，内存受限场景 |

## 大规模测试结果（10,000个向量）

在大规模测试中，各索引类型的表现如下：

| 索引类型 | 添加10000个向量 | 重建索引时间 | 搜索时间 | 最佳匹配相似度 |
|---------|---------------|------------|---------|-------------|
| HNSW    | 2662毫秒      | 1,118毫秒   | <1毫秒   | 0.206 |
| ANNOY   | 93237毫秒     | 17,335毫秒  | 37毫秒   | 0.2118 |
| LSH     | 9423毫秒      | 8,613毫秒   | 1毫秒    | 0.1914 |
| IVF     | 1307毫秒      | 7,175毫秒   | 7毫秒    | 0.2157 |
| PQ      | 1277毫秒      | 13,372毫秒  | 38毫秒   | 0.0442 |

这些结果表明：
- **HNSW**：搜索速度最快，重建索引也较快，但内存消耗较大
- **ANNOY**：添加和重建索引最慢，但提供了持久化能力
- **LSH**：性能均衡，适合大规模近似搜索
- **IVF**：添加速度最快，搜索性能好，适合大规模数据集
- **PQ**：添加速度快，内存占用最小，但搜索精度较低

## 依赖

- Java 11+
- Jackson (用于序列化)
- SLF4J (日志) 