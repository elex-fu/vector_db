# VectorDB 详细修改计划

**版本**: 1.0
**日期**: 2026-02-25
**状态**: 待实施

---

## 1. 修改计划总览

### 1.1 修改优先级矩阵

```
紧急程度 ↑
    │
 P0 │ [召回率修复]      [PQ参数优化]      [efSearch调整]
    │     🔴              🔴                🔴
    │
 P1 │ [SIMD优化]        [细粒度锁]        [批量查询]
    │     🟡              🟡                🟡
    │
 P2 │ [OPQ实现]         [IVF索引]         [MMAP存储]
    │     🟢              🟢                🟢
    │
 P3 │ [GPU加速]         [混合精度]        [分布式]
    │     🔵              🔵                🔵
    └─────────────────────────────────────────────────→
         Week1          Week2-3          Week4+
                      实施时间
```

### 1.2 修改文件清单

| 优先级 | 文件 | 修改类型 | 预计工时 | 依赖 |
|--------|------|----------|----------|------|
| P0 | CompressionConfig.java | 修改 | 4h | - |
| P0 | HnswPqIndex.java | 修改 | 8h | CompressionConfig |
| P0 | HNSWPQIndex.cpp | 修改 | 8h | - |
| P1 | DistanceUtils.java | 新增 | 6h | - |
| P1 | HnswPqIndex.java | 修改 | 6h | DistanceUtils |
| P1 | BatchSearchUtils.java | 新增 | 8h | - |
| P2 | OPQIndex.java | 新增 | 16h | - |
| P2 | IvfPqIndex.java | 新增 | 20h | - |

---

## 2. P0 优先级修改 (立即执行)

### 2.1 Fix #1: PQ参数优化

**问题**: 当前PQ配置压缩比过高，量化误差大，导致召回率低

**当前配置** (512维):
```java
pqSubspaces = 256  // 512/256 = 2维/子空间
subDim = 2
compressionRatio = 8x
```

**目标配置**:
```java
pqSubspaces = 64   // 512/64 = 8维/子空间
subDim = 8
compressionRatio = 32x
```

**修改步骤**:

1. **修改 CompressionConfig.java**

```java
// 文件: src/main/java/com/vectordb/config/CompressionConfig.java
// 行号: 85-95

// 原代码:
public static CompressionConfig recommendedConfig(int dimension) {
    int pqSubspaces = Math.max(8, dimension / 2);
    while (dimension % pqSubspaces != 0 && pqSubspaces > 1) {
        pqSubspaces--;
    }
    return hnswPqConfig(pqSubspaces, 8);
}

// 修改为:
public static CompressionConfig recommendedConfig(int dimension) {
    // 目标: 每个子空间8-16维，平衡精度和压缩
    int targetSubDim = 8;  // 8维/子空间
    int pqSubspaces = dimension / targetSubDim;

    // 确保能整除
    while (dimension % pqSubspaces != 0 && pqSubspaces > 1) {
        pqSubspaces--;
    }

    // 如果无法得到合适的子空间数，尝试其他配置
    if (pqSubspaces < 8) {
        // 维度太小，使用1维/子空间
        pqSubspaces = dimension;
    }

    return hnswPqConfig(pqSubspaces, 8);
}

// 新增方法: 高精度配置 (低压缩比，高召回)
public static CompressionConfig highPrecisionConfig(int dimension) {
    int pqSubspaces = dimension;  // 1维/子空间
    return hnswPqConfig(pqSubspaces, 8);
}

// 新增方法: 高压缩配置 (高压缩比，可能降低召回)
public static CompressionConfig highCompressionConfig(int dimension) {
    int targetSubDim = 16;  // 16维/子空间
    int pqSubspaces = dimension / targetSubDim;
    while (dimension % pqSubspaces != 0 && pqSubspaces > 1) {
        pqSubspaces--;
    }
    return hnswPqConfig(Math.max(pqSubspaces, 8), 8);
}
```

2. **修改 HnswPqIndex.java 自动调整逻辑**

```java
// 文件: src/main/java/com/vectordb/index/HnswPqIndex.java
// 行号: 77-86

// 原代码:
if (dimension % pqSubspaces != 0) {
    this.pqSubspaces = findBestSubspaceDivisor(dimension);
    log.warn("维度{}不能被PQ子空间数{}整除，自动调整为{}",
            dimension, compressionConfig.getPqSubspaces(), this.pqSubspaces);
}

// 修改为:
if (dimension % pqSubspaces != 0) {
    // 优先保持目标子空间维度(8-16)，寻找最接近的可整除数
    int targetSubDim = Math.max(4, dimension / pqSubspaces);
    this.pqSubspaces = findOptimalSubspaces(dimension, targetSubDim);

    if (this.pqSubspaces != compressionConfig.getPqSubspaces()) {
        log.warn("维度{}不能被PQ子空间数{}整除，自动调整为{} ({}维/子空间)",
                dimension, compressionConfig.getPqSubspaces(),
                this.pqSubspaces, dimension / this.pqSubspaces);
    }
}

// 新增方法:
private int findOptimalSubspaces(int dimension, int targetSubDim) {
    // 寻找最接近targetSubDim的可整除子空间数
    int optimalSubspaces = dimension / targetSubDim;

    // 确保至少为1
    optimalSubspaces = Math.max(1, optimalSubspaces);

    // 向上或向下寻找可整除的数
    for (int offset = 0; offset < dimension / 4; offset++) {
        int candidate = optimalSubspaces - offset;
        if (candidate > 0 && dimension % candidate == 0) {
            return candidate;
        }
        candidate = optimalSubspaces + offset;
        if (candidate <= dimension && dimension % candidate == 0) {
            return candidate;
        }
    }

    // fallback: 使用dimension本身 (1维/子空间)
    return dimension;
}
```

**测试验证**:

```java
@Test
public void testPqConfiguration() {
    // 512维应该使用64个子空间 (8维/子空间)
    CompressionConfig config = CompressionConfig.recommendedConfig(512);
    assertEquals(64, config.getPqSubspaces());
    assertEquals(32.0, config.getCompressionRatio(512), 0.1);

    // 128维应该使用16个子空间 (8维/子空间)
    config = CompressionConfig.recommendedConfig(128);
    assertEquals(16, config.getPqSubspaces());

    // 验证HnswPqIndex使用正确配置
    HnswPqIndex index = new HnswPqIndex(512, 10000,
        CompressionConfig.recommendedConfig(512));
    assertEquals(64, index.getPqSubspaces());
}
```

**预期效果**:
- 压缩比: 8x → 32x (512维)
- Recall: 8% → 50-70%
- 训练时间: 减少50%

---

### 2.2 Fix #2: efSearch动态调整

**问题**: efSearch太小，只访问1%数据，召回率不足

**当前实现**:
```cpp
// HNSWPQIndex.cpp 第512行
int efSearch = std::max(k * 50, std::min(dataSize / 10, 2000));
// k=10, dataSize=10000 => efSearch = max(500, 1000) = 1000
// 只访问 1000/100000 = 1% 数据
```

**修改方案**:

1. **修改 HNSWPQIndex.cpp**

```cpp
// 文件: native/index/HNSWPQIndex.cpp
// 行号: 509-512

// 原代码:
int dataSize = size_.load(std::memory_order_acquire);
int efSearch = std::max(k * 50, std::min(dataSize / 10, 2000));

// 修改为:
int dataSize = size_.load(std::memory_order_acquire);

// 新策略: 至少访问10%数据，确保召回率
int minEfByRatio = static_cast<int>(dataSize * 0.10);  // 10%数据
int minEfByK = k * 100;  // 100倍k
int maxEf = std::min(dataSize, 5000);  // 上限5000

int efSearch = std::max({minEfByRatio, minEfByK, k * 50});
efSearch = std::min(efSearch, maxEf);

// 记录日志
static std::atomic<int> logCounter{0};
if (++logCounter % 100 == 1) {
    std::cout << "[HNSWPQ] efSearch=" << efSearch
              << " (dataSize=" << dataSize
              << ", k=" << k << ")" << std::endl;
}
```

2. **添加自适应efSearch配置**

```cpp
// 文件: native/index/HNSWPQIndex.h
// 在 HNSWPQConfig 结构体中添加:

struct HNSWPQConfig {
    // ... 现有字段 ...

    // 新增: efSearch策略
    enum EfSearchStrategy {
        FIXED,           // 固定值
        DYNAMIC_RATIO,   // 基于数据比例 (默认)
        DYNAMIC_K_BASED  // 基于k值
    };
    EfSearchStrategy efStrategy = DYNAMIC_RATIO;

    float minSearchRatio = 0.10f;   // 最小搜索比例 (10%)
    int minEfMultiplier = 100;       // k的最小倍数
    int maxEf = 5000;               // ef上限

    // 计算efSearch的方法
    int calculateEfSearch(int k, int dataSize) const {
        switch (efStrategy) {
            case FIXED:
                return efSearch;  // 使用配置中的固定值

            case DYNAMIC_RATIO: {
                int efByRatio = static_cast<int>(dataSize * minSearchRatio);
                int efByK = k * minEfMultiplier;
                int ef = std::max({efByRatio, efByK, k * 50});
                return std::min(ef, maxEf);
            }

            case DYNAMIC_K_BASED:
                return std::min(k * minEfMultiplier, maxEf);
        }
        return k * 50;  // 默认
    }
};
```

3. **修改搜索方法使用配置**

```cpp
// 文件: native/index/HNSWPQIndex.cpp
// 行号: 509

// 原代码:
int efSearch = std::max(k * 50, std::min(dataSize / 10, 2000));

// 修改为:
int efSearch = config_.calculateEfSearch(k, dataSize);
```

**测试验证**:

```cpp
// 测试不同策略
TEST(HNSWPQConfigTest, EfSearchCalculation) {
    HNSWPQConfig config;

    // 10000数据, k=10
    // DYNAMIC_RATIO: max(1000, 1000, 500) = 1000
    config.efStrategy = HNSWPQConfig::DYNAMIC_RATIO;
    EXPECT_EQ(config.calculateEfSearch(10, 10000), 1000);

    // DYNAMIC_K_BASED: 10 * 100 = 1000
    config.efStrategy = HNSWPQConfig::DYNAMIC_K_BASED;
    EXPECT_EQ(config.calculateEfSearch(10, 10000), 1000);

    // 大k值应该受maxEf限制
    EXPECT_EQ(config.calculateEfSearch(100, 100000), 5000);
}
```

**预期效果**:
- 访问数据比例: 1% → 10%
- Recall: +20-30%
- 搜索延迟: +50% (可接受)

---

### 2.3 Fix #3: 双层重排序

**问题**: 仅对20*k个候选进行重排序，召回的候选集不够大

**当前实现**:
```cpp
// HNSWPQIndex.cpp 第627行
const int refineFactor = 20;
int nRefine = std::min(static_cast<int>(finalResults.size()), k * refineFactor);
// 只取Top-200进行精确重排序
```

**修改方案**:

1. **修改 HNSWPQIndex.cpp**

```cpp
// 文件: native/index/HNSWPQIndex.cpp
// 行号: 554-657 (search方法)

// 修改1: 扩大候选池
// 原代码 (第555行):
const int candidatePoolSize = k * 200;

// 修改为:
const int candidatePoolSize = k * 500;  // 扩大到500倍

// 修改2: 双层重排序
// 原代码 (第617-649行):
const int refineFactor = 20;
int nRefine = std::min(static_cast<int>(finalResults.size()), k * refineFactor);

// 修改为:
// 第一层: 从候选池选择Top-(100*k)
const int firstLevelSize = k * 100;
std::vector<DistIdPair> firstLevelResults;
firstLevelResults.reserve(std::min(static_cast<int>(finalResults.size()), firstLevelSize));

// 第二层: 从Top-(100*k)中选择Top-(20*k)进行精确距离计算
const int secondLevelSize = k * 20;

// 第三层: 最终Top-k

// 实现代码:
// 步骤1: 使用PQ距离排序候选
std::partial_sort(finalResults.begin(),
                  finalResults.begin() + std::min(firstLevelSize, static_cast<int>(finalResults.size())),
                  finalResults.end());

// 步骤2: 对Top-(100*k)使用精确距离重排序
int nFirstLevel = std::min(static_cast<int>(finalResults.size()), firstLevelSize);
std::vector<DistIdPair> refinedResults;
refinedResults.reserve(nFirstLevel);

for (int i = 0; i < nFirstLevel; i++) {
    int nodeId = finalResults[i].second;
    float exactDist = computeExactDistanceToQuery(query, nodeId);
    refinedResults.emplace_back(exactDist, nodeId);
}

// 步骤3: 按精确距离排序
std::sort(refinedResults.begin(), refinedResults.end());

// 步骤4: 取Top-k作为最终结果
int nFinal = std::min(k, static_cast<int>(refinedResults.size()));
for (int i = 0; i < nFinal; i++) {
    resultDistances[i] = refinedResults[i].first;
    resultIds[i] = vectorStore_.getId(refinedResults[i].second);
}
*resultCount = nFinal;
```

2. **添加配置参数**

```cpp
// 文件: native/index/HNSWPQIndex.h
// 在 HNSWPQConfig 中添加:

struct HNSWPQConfig {
    // ... 现有字段 ...

    // 重排序配置
    int candidatePoolMultiplier = 500;   // 候选池大小 = k * 500
    int firstLevelMultiplier = 100;       // 第一层 = k * 100
    int secondLevelMultiplier = 20;       // 第二层 = k * 20
    bool useTwoLevelRefinement = true;    // 启用双层重排序
};
```

**测试验证**:

```cpp
TEST(HNSWPQIndexTest, TwoLevelRefinement) {
    HNSWPQConfig config;
    config.useTwoLevelRefinement = true;
    config.candidatePoolMultiplier = 500;
    config.firstLevelMultiplier = 100;
    config.secondLevelMultiplier = 20;

    HNSWPQIndex index(128, 10000, config);

    // 添加测试数据
    // ...

    // 搜索k=10
    int resultIds[10];
    float resultDists[10];
    int resultCount;

    float query[128] = {0};
    index.search(query, 10, resultIds, resultDists, &resultCount);

    // 验证返回正确数量
    EXPECT_EQ(resultCount, 10);
}
```

**预期效果**:
- 精确距离计算候选: 200 → 2000
- Recall: +10-15%
- 延迟: +30% (增加90次精确距离计算)

---

### 2.4 Fix #4: 精确距离构建图

**问题**: 使用PQ近似距离构建HNSW图，图质量差

**当前实现**:
```cpp
// HNSWPQIndex.cpp 第337行
float currDist = computeExactDistance(newIndex, currObj);
// ...
for (int i = 0; i < levelInfo.size; i++) {
    int neighbor = levelInfo.data[i];
    float d = computeExactDistance(newIndex, neighbor);  // 精确距离
}
```

**注意**: 经检查，C++实现已经使用精确距离构建图。但Java层HnswPqIndex使用的是PQ距离。

**修改方案 (Java层)**:

```java
// 文件: src/main/java/com/vectordb/index/HnswPqIndex.java
// 方法: addVectorCompressed (第280行)

// 原代码: 使用PQ距离进行邻居选择
for (int currentLevel = Math.min(level, maxLevel - 1); currentLevel >= 0; currentLevel--) {
    List<SearchResult> neighbors = searchLayerCompressed(...);
}

// 修改为: 构建时使用精确距离
private boolean addVectorWithExactDistance(Vector vector) {
    int id = vector.getId();
    int index = currentSize++;

    vectors.put(id, vector);
    idToIndex.put(id, index);

    // PQ编码 (仅用于存储和搜索加速)
    byte[] code = encodeVector(vector);
    System.arraycopy(code, 0, codes[index], 0, pqSubspaces);

    int level = assignLevel();
    idToLevel.put(id, level);

    if (entryPoint == -1) {
        entryPoint = id;
        for (int i = 0; i <= level; i++) {
            graph.get(i).put(id, new ArrayList<>());
        }
        return true;
    }

    // HNSW插入 - 使用精确距离构建图
    int currentEntryPoint = entryPoint;

    for (int currentLevel = maxLevel - 1; currentLevel > level; currentLevel--) {
        currentEntryPoint = searchLayerClosestExact(vector, currentEntryPoint, currentLevel);
    }

    for (int currentLevel = Math.min(level, maxLevel - 1); currentLevel >= 0; currentLevel--) {
        // 关键修改: 使用精确距离搜索邻居
        List<SearchResult> neighbors = searchLayerExact(vector, currentEntryPoint, efConstruction, currentLevel);
        List<Integer> selectedNeighbors = selectNeighbors(vector, neighbors, m);

        // 设置邻居...
    }

    return true;
}

// 新增方法: 使用精确距离搜索
private List<SearchResult> searchLayerExact(Vector query, int entryPointId, int ef, int level) {
    // 类似searchLayer，但使用calculateDistance而不是computePQDistance
    PriorityQueue<SearchResult> resultSet = new PriorityQueue<>(
        Comparator.comparing(SearchResult::getDistance).reversed());

    float distance = calculateDistance(query, vectors.get(entryPointId));
    resultSet.add(new SearchResult(entryPointId, distance));

    Set<Integer> visited = new HashSet<>();
    visited.add(entryPointId);

    // ... 其余逻辑相同，但使用calculateDistance
}
```

**预期效果**:
- 图质量提升
- Recall: +10-20%
- 构建时间: 增加50% (因为使用精确距离)

---

## 3. P1 优先级修改 (本周内)

### 3.1 Enhancement #1: Java层SIMD优化

**问题**: Java层距离计算无SIMD优化，速度慢

**方案A: Java Vector API (JDK 16+)**

```java
// 文件: src/main/java/com/vectordb/util/SIMDVectorUtils.java (新增)

package com.vectordb.util;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class SIMDVectorUtils {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_256;

    // SIMD欧氏距离计算
    public static float euclideanDistanceSIMD(float[] a, float[] b) {
        int i = 0;
        float sum = 0;

        // SIMD批量处理
        for (; i <= a.length - SPECIES.length(); i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            FloatVector diff = va.sub(vb);
            sum += diff.mul(diff).reduceLanes(VectorOperators.ADD);
        }

        // 处理剩余元素
        for (; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }

        return sum;
    }

    // SIMD点积计算 (用于余弦相似度)
    public static float dotProductSIMD(float[] a, float[] b) {
        int i = 0;
        float sum = 0;

        for (; i <= a.length - SPECIES.length(); i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            sum += va.mul(vb).reduceLanes(VectorOperators.ADD);
        }

        for (; i < a.length; i++) {
            sum += a[i] * b[i];
        }

        return sum;
    }
}
```

**方案B: JNI调用C++ SIMD实现**

```cpp
// 文件: native/bridge/VectorDBJNI.cpp (新增)

JNIEXPORT jfloat JNICALL Java_com_vectordb_util_NativeUtils_euclideanDistanceNative(
    JNIEnv* env,
    jclass clazz,
    jfloatArray a,
    jfloatArray b,
    jint dim) {

    jfloat* aArray = env->GetFloatArrayElements(a, nullptr);
    jfloat* bArray = env->GetFloatArrayElements(b, nullptr);

    float dist = euclideanDistanceAVX2(aArray, bArray, dim);

    env->ReleaseFloatArrayElements(a, aArray, JNI_ABORT);
    env->ReleaseFloatArrayElements(b, bArray, JNI_ABORT);

    return dist;
}
```

**修改 HnswPqIndex.java 使用SIMD**:

```java
// 文件: src/main/java/com/vectordb/index/HnswPqIndex.java

// 原代码 (第126-132行):
private float calculateDistance(Vector v1, Vector v2) {
    if (useCosineSimilarity) {
        return 1.0f - v1.cosineSimilarity(v2);
    } else {
        return v1.euclideanDistance(v2);
    }
}

// 修改为:
private float calculateDistance(Vector v1, Vector v2) {
    if (useCosineSimilarity) {
        float dot = SIMDVectorUtils.dotProductSIMD(v1.getValues(), v2.getValues());
        return 1.0f - dot;
    } else {
        return SIMDVectorUtils.euclideanDistanceSIMD(v1.getValues(), v2.getValues());
    }
}
```

**预期效果**:
- 距离计算速度: 4-8x
- QPS: +50-100%

---

### 3.2 Enhancement #2: 细粒度锁优化

**问题**: `synchronized` 方法使用全局锁，并发受限

**修改方案**:

```java
// 文件: src/main/java/com/vectordb/index/HnswPqIndex.java

// 原代码:
public synchronized boolean addVector(Vector vector) { ... }
public synchronized boolean removeVector(int id) { ... }

// 修改为: 分段锁
private static final int NUM_SEGMENTS = 16;
private final ReadWriteLock[] segmentLocks = new ReentrantReadWriteLock[NUM_SEGMENTS];

public HnswPqIndex(...) {
    // 初始化锁
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        segmentLocks[i] = new ReentrantReadWriteLock();
    }
}

private int getSegment(int id) {
    return Math.abs(id) % NUM_SEGMENTS;
}

public boolean addVector(Vector vector) {
    int segment = getSegment(vector.getId());
    Lock writeLock = segmentLocks[segment].writeLock();
    writeLock.lock();
    try {
        // 添加逻辑
        return addVectorInternal(vector);
    } finally {
        writeLock.unlock();
    }
}

public boolean removeVector(int id) {
    int segment = getSegment(id);
    Lock writeLock = segmentLocks[segment].writeLock();
    writeLock.lock();
    try {
        // 删除逻辑
        return removeVectorInternal(id);
    } finally {
        writeLock.unlock();
    }
}
```

**预期效果**:
- 并发度: 1 → 16+
- QPS (多线程): +200-400%

---

### 3.3 Enhancement #3: 批量查询接口

**问题**: 单次查询JNI开销大

**修改方案**:

```java
// 文件: src/main/java/com/vectordb/core/VectorDatabase.java (新增方法)

/**
 * 批量搜索
 * @param queries 查询向量列表
 * @param k 每个查询返回的结果数
 * @return 每个查询的搜索结果
 */
public List<List<SearchResult>> searchBatch(List<float[]> queries, int k) {
    List<List<SearchResult>> results = new ArrayList<>(queries.size());

    // 并行处理批量查询
    int numThreads = Runtime.getRuntime().availableProcessors();
    ExecutorService executor = Executors.newFixedThreadPool(numThreads);

    List<Future<List<SearchResult>>> futures = new ArrayList<>();
    for (float[] query : queries) {
        futures.add(executor.submit(() -> search(query, k)));
    }

    for (Future<List<SearchResult>> future : futures) {
        try {
            results.add(future.get());
        } catch (Exception e) {
            log.error("批量搜索失败", e);
            results.add(Collections.emptyList());
        }
    }

    executor.shutdown();
    return results;
}
```

**预期效果**:
- 批量查询QPS: 10,000+
- 单查询平均延迟: 降低30%

---

## 4. 测试计划

### 4.1 单元测试

```java
// CompressionConfigTest.java
@Test
public void testRecommendedConfig() {
    CompressionConfig config = CompressionConfig.recommendedConfig(512);
    assertEquals(64, config.getPqSubspaces());
    assertEquals(8, config.getPqBits());
    assertEquals(32.0, config.getCompressionRatio(512), 0.1);
}

// HnswPqIndexTest.java
@Test
public void testRecallRate() {
    HnswPqIndex index = new HnswPqIndex(512, 10000,
        CompressionConfig.recommendedConfig(512));

    // 添加10000个向量
    // ...

    // 搜索并计算召回率
    List<float[]> queries = generateQueries(100);
    double avgRecall = calculateAverageRecall(index, queries, 10);

    assertTrue("召回率应 >= 85%", avgRecall >= 0.85);
}

// SIMDVectorUtilsTest.java
@Test
public void testSIMDDistance() {
    float[] a = generateRandomVector(512);
    float[] b = generateRandomVector(512);

    float expected = euclideanDistance(a, b);
    float actual = SIMDVectorUtils.euclideanDistanceSIMD(a, b);

    assertEquals(expected, actual, 0.001);
}
```

### 4.2 性能基准测试

```java
// PerformanceBenchmarkTest.java
@Test
public void benchmarkQPS() {
    VectorDatabase db = new VectorDatabase.Builder()
        .withDimension(512)
        .withMaxElements(100000)
        .withCompressionEnabled(true)
        .build();

    // 添加10万向量
    // ...

    // 测试QPS
    int numQueries = 10000;
    long start = System.nanoTime();

    for (int i = 0; i < numQueries; i++) {
        db.search(generateRandomVector(512), 10);
    }

    long elapsed = (System.nanoTime() - start) / 1_000_000; // ms
    double qps = numQueries * 1000.0 / elapsed;

    System.out.println("QPS: " + qps);
    assertTrue("QPS应 >= 5000", qps >= 5000);
}
```

---

## 5. 实施时间表

| 周次 | 任务 | 负责人 | 验收标准 |
|------|------|--------|----------|
| Week 1 | PQ参数优化 | TBD | Recall >= 50% |
| Week 1 | efSearch调整 | TBD | 访问10%数据 |
| Week 1 | 双层重排序 | TBD | Recall >= 70% |
| Week 1 | 精确距离建图 | TBD | Recall >= 85% |
| Week 2 | SIMD优化 | TBD | QPS >= 4000 |
| Week 2 | 细粒度锁 | TBD | 并发QPS >= 8000 |
| Week 2 | 批量查询 | TBD | 批量QPS >= 10000 |
| Week 3 | 集成测试 | TBD | 所有测试通过 |
| Week 3 | 性能回归 | TBD | Recall >= 90%, QPS >= 5000 |

---

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| SIMD优化引入bug | 中 | 高 | 完善单元测试，保留fallback |
| Recall提升不达预期 | 中 | 极高 | 备选方案: 纯HNSW |
| 性能优化导致不稳定 | 低 | 高 | 渐进式优化，充分测试 |
| 时间表延期 | 中 | 中 | 分阶段交付，优先P0 |

---

*文档创建时间: 2026-02-25*
*作者: Claude Code*
