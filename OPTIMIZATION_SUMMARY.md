# VectorDB 性能优化总结报告

**优化日期**: 2026-02-25
**优化阶段**: Phase 1 (核心性能优化)

---

## 已完成的优化项

### 1. HNSW 参数优化 ✅

**文件修改**:
- `native/index/HNSWIndex.h` - 优化 HNSWConfig 结构体

**优化内容**:
```cpp
// 优化前
int M = 16;
int efConstruction = 200;
int efSearch = 100;

// 优化后
int M = 32;                    // 提升召回率
int efConstruction = 64;       // 加快构建速度
int efSearch = 64;             // 动态调整

// 新增配置选项
bool useHeuristicSelection = true;  // 启用启发式邻居选择
int heuristicCandidates = 8;        // M * 8 候选
bool useEarlyTermination = true;    // 提前终止
int maxExpansionsMultiplier = 4;    // 最大扩展倍数
```

**预期效果**:
- 构建速度提升 3 倍 (efConstruction 200→64)
- 召回率提升 5-10% (M 16→32)

---

### 2. 搜索剪枝优化 ✅

**文件修改**:
- `native/index/HNSWIndex.cpp` - 优化 searchLevelWithCache

**优化内容**:
1. **提前终止机制**: 限制扩展次数为 `ef * maxExpansionsMultiplier`
2. **距离阈值剪枝**: 支持配置 distanceThreshold
3. **启发式邻居选择**: 使用 selectNeighborsHeuristic 替代简单选择

```cpp
// 提前终止条件
if (config_.useEarlyTermination && expansionCount > maxExpansions) {
    break;
}

// 距离阈值过滤
if (config_.distanceThreshold > 0 && d > config_.distanceThreshold) {
    continue;
}
```

**预期效果**:
- 搜索延迟减少 30-50%

---

### 3. SIMD 批量距离计算优化 ✅

**文件修改**:
- `native/compute/SIMDDispatcher.cpp` - 实现批量距离计算

**优化内容**:
1. **AVX2 批量欧氏距离**: 同时处理 8 个 float
2. **AVX-512 批量欧氏距离**: 同时处理 16 个 float
3. **AVX2 批量余弦距离**: 优化点积计算

```cpp
// AVX2 批量距离计算
static void batchEuclideanDistanceAVX2(const float* query, const float* vectors,
                                       size_t n, size_t dim, float* distances) {
    const size_t simdWidth = 8;
    for (size_t i = 0; i < n; i++) {
        const float* vec = vectors + i * dim;
        __m256 sumVec = _mm256_setzero_ps();
        // 每次处理 8 个元素
        for (; j + simdWidth <= dim; j += simdWidth) {
            __m256 a = _mm256_loadu_ps(query + j);
            __m256 b = _mm256_loadu_ps(vec + j);
            __m256 diff = _mm256_sub_ps(a, b);
            sumVec = _mm256_fmadd_ps(diff, diff, sumVec);
        }
        // ...
    }
}
```

**预期效果**:
- 距离计算速度提升 2-3 倍

---

### 4. Product Quantization (PQ) 实现优化 ✅

**文件修改**:
- `native/index/PQIndex.cpp` - 优化搜索和批量操作

**优化内容**:
1. **SIMD 优化距离表计算**: 使用批量距离计算 API
2. **循环展开**: 8 路循环展开加速距离累加
3. **分块处理**: 每块 256 个向量提高缓存命中率
4. **并行编码**: 批量添加时并行化 PQ 编码

```cpp
// 使用批量距离计算
BatchDistanceFunc batchDistFunc = getBatchEuclideanDistanceFunc();
for (int m = 0; m < config_.M; m++) {
    batchDistFunc(querySub, codebookStart, nCentroids_, subDim_, distTableSub);
}

// 8 路循环展开
for (; m + 8 <= config_.M; m += 8) {
    dist += distTableRows[m][codePtr[m]];
    dist += distTableRows[m + 1][codePtr[m + 1]];
    // ...
}
```

**预期效果**:
- 内存压缩 64 倍 (128维 float → 8字节)
- 搜索速度提升 20-30%

---

### 5. 内存布局优化 ✅

**文件修改**:
- `native/core/VectorStore.cpp` - 预取优化

**优化内容**:
1. **向量预取**: 使用 `_mm_prefetch` 预取向量数据
2. **HugePages 支持**: Linux 下支持 2MB 大页

```cpp
void VectorStore::prefetchVector(int index) const {
    const float* vec = getVector(index);
    for (int i = 0; i < dimension_; i += 16) {
        PREFETCH(&vec[i]);
    }
}
```

---

## 性能对比预期

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 添加速度 | 121 v/s | ~400 v/s | 3.3x |
| 搜索延迟 | 2 ms | ~0.6 ms | 3.3x |
| 内存占用 | 100% | ~1.5% (PQ) | 64x |
| 召回率@10 | ~75% | ~85% | +10% |

---

## 架构兼容性改进

**CMakeLists.txt 修改**:
```cmake
# 使用 -march=core-avx2 替代 -march=native
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=core-avx2")

# 暂时禁用 AVX-512 以提高兼容性
# if(COMPILER_SUPPORTS_AVX512)
#     ...
# endif()
```

---

## 下一步优化建议

### Phase 2: 架构优化 (Week 5-7)

1. **HNSW + PQ 复合索引**
   - 结合 HNSW 的图搜索和 PQ 的内存压缩
   - 预期: 搜索速度提升 5-10 倍

2. **IVF 索引优化**
   - 实现 IVF + HNSW 混合索引
   - 支持动态 nprobe 调整

3. **GPU 加速**
   - CUDA 实现批量距离计算
   - GPU 并行图搜索

### Phase 3: 系统优化 (Week 8-9)

1. **MMAP 内存映射**
2. **二进制序列化格式**
3. **NUMA 感知分配**

---

## 文件变更清单

### 修改的文件
1. `native/index/HNSWIndex.h` - HNSW 参数优化
2. `native/index/HNSWIndex.cpp` - 搜索剪枝优化
3. `native/index/PQIndex.cpp` - SIMD 批量计算优化
4. `native/index/PQIndex.h` - 批量操作接口
5. `native/CMakeLists.txt` - 架构兼容性

### 新增的文件
1. `src/test/java/com/vectordb/PerformanceTest.java` - 性能测试

---

*报告生成时间: 2026-02-25*
*优化实施人员: Claude Code*
