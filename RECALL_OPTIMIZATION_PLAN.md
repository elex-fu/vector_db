# 召回率优化实施计划

**目标**: Recall@10 从 8.56% 提升到 90%+
**时间**: 3天
**日期**: 2026-02-26

---

## 1. 优化策略概览

```
当前状态: Recall 8.56%

优化步骤:
Step 1: PQ参数优化 (256→64子空间)     +40%  → 48%
Step 2: efSearch调整 (1%→10%)          +20%  → 68%
Step 3: 双层重排序 (20x→100x)          +15%  → 83%
Step 4: 精确距离建图                   +10%  → 93%

目标: Recall >90%
```

---

## 2. 详细修改清单

### Fix #1: PQ参数优化 (预计收益 +40%)

**问题**: 当前512维使用256个子空间，2维/子空间，量化误差过大

**修改文件**:
1. `CompressionConfig.java` - 修改recommendedConfig
2. `HnswPqIndex.java` - 优化子空间计算逻辑

**修改内容**:
```java
// 原配置: pqSubspaces = 256 (2维/子空间)
// 新配置: pqSubspaces = 64  (8维/子空间)
// 压缩比: 8x → 32x (512维)
```

### Fix #2: efSearch调整 (预计收益 +20%)

**问题**: 当前efSearch=1000，仅访问1%数据

**修改文件**:
1. `HnswPqIndex.java` - 修改searchLayer方法
2. 调整efSearch计算逻辑

**修改内容**:
```java
// 原: efSearch = max(k*50, min(dataSize/10, 2000))
// 新: efSearch = max(dataSize*0.15, k*100)
// 访问数据: 1% → 15%
```

### Fix #3: 双层重排序 (预计收益 +15%)

**问题**: 仅对20*k候选进行精确重排序

**修改文件**:
1. `HnswPqIndex.java` - 修改searchNearest方法
2. 实现双层重排序逻辑

**修改内容**:
```java
// Layer 1: 收集500*k候选 (PQ距离)
// Layer 2: Top-100*k精确距离重排序
// Layer 3: 返回Top-k
```

### Fix #4: 精确距离建图 (预计收益 +10%)

**问题**: 使用PQ近似距离构建HNSW图

**修改文件**:
1. `HnswPqIndex.java` - 修改addVector方法
2. 建图时使用精确距离

**修改内容**:
```java
// 建图时使用calculateDistance (精确距离)
// 搜索时使用computePQDistance (快速近似)
```

---

## 3. 验证计划

### 3.1 单元测试

```java
@Test
public void testRecallRate() {
    // 测试Recall是否达到90%
}
```

### 3.2 性能基准

```java
@Test
public void benchmarkRecallAndQPS() {
    // 测试Recall和QPS的平衡
}
```

### 3.3 对比测试

- 对比优化前后Recall
- 对比优化前后QPS
- 对比优化前后内存使用

---

## 4. 时间表

| 时间 | 任务 | 负责人 | 验收标准 |
|------|------|--------|----------|
| Day 1 AM | Fix #1 PQ参数优化 | TBD | 压缩比32x |
| Day 1 PM | Fix #2 efSearch调整 | TBD | 访问15%数据 |
| Day 2 AM | Fix #3 双层重排序 | TBD | 100x候选池 |
| Day 2 PM | Fix #4 精确距离建图 | TBD | 建图使用精确距离 |
| Day 3 | 集成测试 & 验证 | TBD | Recall >90% |

---

## 5. 风险控制

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 优化后QPS下降过多 | 高 | 中 | 监控QPS，必要时调整参数 |
| 内存使用超标 | 中 | 中 | 监控内存，调整压缩比 |
| Recall提升不达预期 | 低 | 高 | 备选方案: 纯HNSW |

---

*计划创建时间: 2026-02-26*
*执行人: Claude Code*
