# 向量数据库测试说明

本文档提供了关于向量数据库测试的详细说明，包括如何设置测试环境、运行测试以及解释测试结果。

## 测试结构

测试代码组织在以下目录结构中：

```
src/test/java/com/vectordb/
├── core/                  # 核心类测试
│   ├── VectorTest.java    # Vector类测试
│   └── SearchResultTest.java # SearchResult类测试
├── index/                 # 索引测试
│   └── HnswIndexTest.java # HNSW索引测试
├── storage/               # 存储测试
│   └── VectorStorageTest.java # 向量存储测试
├── util/                  # 工具类测试
│   └── VectorUtilsTest.java # 向量工具类测试
├── benchmark/             # 性能基准测试
│   └── PerformanceBenchmarkTest.java # 性能测试
├── VectorDatabaseIntegrationTest.java # 集成测试
└── VectorDatabaseTestSuite.java       # 测试套件
```

## 测试依赖

测试需要以下依赖：

- JUnit 4.12 或更高版本
- 其他项目依赖（如Jackson等）

确保在项目的`pom.xml`（Maven）或`build.gradle`（Gradle）中包含这些依赖。

## 运行测试

### 运行所有测试

使用测试套件运行所有单元测试：

```bash
# 使用Maven
mvn test -Dtest=com.vectordb.VectorDatabaseTestSuite

# 使用Gradle
./gradlew test --tests com.vectordb.VectorDatabaseTestSuite
```

### 运行特定测试类

运行特定的测试类：

```bash
# 使用Maven
mvn test -Dtest=com.vectordb.core.VectorTest

# 使用Gradle
./gradlew test --tests com.vectordb.core.VectorTest
```

### 运行性能基准测试

性能基准测试可能需要较长时间运行，建议单独运行：

```bash
# 使用Maven
mvn test -Dtest=com.vectordb.benchmark.PerformanceBenchmarkTest

# 使用Gradle
./gradlew test --tests com.vectordb.benchmark.PerformanceBenchmarkTest
```

## 测试说明

### 单元测试

单元测试验证各个组件的功能正确性：

- **VectorTest**: 测试向量基本属性和操作
- **SearchResultTest**: 测试搜索结果的排序和比较
- **HnswIndexTest**: 测试HNSW索引的添加、删除和搜索功能
- **VectorStorageTest**: 测试向量的持久化存储和检索
- **VectorUtilsTest**: 测试向量工具类的各种计算方法

### 集成测试

**VectorDatabaseIntegrationTest** 测试整个系统的功能，包括：

- 添加和检索向量
- 删除向量
- 相似性搜索
- 数据库持久化和重新加载

### 性能基准测试

**PerformanceBenchmarkTest** 测试系统在不同条件下的性能：

- 批量插入性能
- 搜索性能随数据库大小的变化
- 不同维度的搜索性能
- 内存使用情况

## 测试结果解释

### 单元测试和集成测试

这些测试应该全部通过，表明系统功能正常。如果有测试失败，错误信息将指示问题所在。

### 性能基准测试

性能测试会输出各种指标：

- 插入时间：每个向量的平均插入时间
- 查询时间：每次查询的平均时间
- 内存使用：数据库占用的内存量

这些指标可以用来评估系统性能，并与不同的实现或配置进行比较。

## 注意事项

1. 测试会创建临时文件和目录，测试完成后会自动清理
2. 性能测试结果可能因硬件配置而异
3. 在生产环境中，建议使用更大的数据集进行性能测试

## 故障排除

如果测试失败，请检查：

1. 依赖是否正确配置
2. 测试环境是否有足够的权限创建和删除文件
3. 系统是否有足够的内存运行测试，特别是性能测试

如有其他问题，请参考项目文档或联系开发团队。 