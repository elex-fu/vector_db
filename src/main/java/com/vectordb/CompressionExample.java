package com.vectordb;

import com.vectordb.config.CompressionConfig;
import com.vectordb.core.SearchResult;
import com.vectordb.core.VectorDatabase;
import com.vectordb.index.HnswPqIndex;
import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.Random;

/**
 * 向量数据库压缩功能示例
 * 演示如何启用/禁用压缩，以及不同配置的效果
 */
@Slf4j
public class CompressionExample {

    public static void main(String[] args) {
        CompressionExample example = new CompressionExample();

        // 示例1: 不使用压缩
        example.demoWithoutCompression();

        // 示例2: 使用推荐压缩配置
        example.demoWithCompression();

        // 示例3: 自定义压缩参数
        example.demoCustomCompression();

        // 示例4: 性能对比
        example.demoPerformanceComparison();
    }

    /**
     * 示例1: 不使用压缩
     */
    private void demoWithoutCompression() {
        log.info("\n==============================================");
        log.info("示例1: 不使用压缩");
        log.info("==============================================");

        // 创建数据库 (不启用压缩)
        VectorDatabase db = new VectorDatabase.Builder()
                .withDimension(512)
                .withMaxElements(10000)
                .withStoragePath("./example_no_compression")
                .withCompressionEnabled(false)  // 显式禁用压缩
                .build();

        try {
            // 添加向量
            addSampleVectors(db, 1000, 512);

            // 搜索
            float[] query = generateRandomVector(512);
            List<SearchResult> results = db.search(query, 10);

            log.info("索引类型: {}", db.getIndexType());
            log.info("压缩启用: {}", db.isCompressionEnabled());
            log.info("向量数量: {}", db.size());
            log.info("搜索结果: {} 个", results.size());

            db.close();
        } catch (Exception e) {
            log.error("示例1出错: {}", e.getMessage(), e);
        }
        cleanup("./example_no_compression");
    }

    /**
     * 示例2: 使用推荐压缩配置
     */
    private void demoWithCompression() {
        log.info("\n==============================================");
        log.info("示例2: 使用推荐压缩配置");
        log.info("==============================================");

        int dimension = 512;

        // 获取推荐配置
        CompressionConfig config = CompressionConfig.recommendedConfig(dimension);
        log.info("推荐配置: PQ子空间数={}, 位数={}, 压缩比={}x",
                config.getPqSubspaces(),
                config.getPqBits(),
                String.format("%.2f", config.getCompressionRatio(dimension)));

        // 创建数据库 (启用压缩)
        VectorDatabase db = new VectorDatabase.Builder()
                .withDimension(dimension)
                .withMaxElements(10000)
                .withStoragePath("./example_with_compression")
                .withCompressionEnabled(true)  // 启用推荐压缩配置
                .build();

        try {
            // 添加向量
            addSampleVectors(db, 1000, dimension);

            // 搜索
            float[] query = generateRandomVector(dimension);
            List<SearchResult> results = db.search(query, 10);

            log.info("索引类型: {}", db.getIndexType());
            log.info("压缩启用: {}", db.isCompressionEnabled());
            log.info("压缩比: {}x", String.format("%.2f", db.getCompressionRatio()));
            log.info("向量数量: {}", db.size());
            log.info("搜索结果: {} 个", results.size());

            // 显示详细统计
            if (db.getIndex() instanceof HnswPqIndex) {
                HnswPqIndex index = (HnswPqIndex) db.getIndex();
                log.info("是否已训练: {}", index.isTrained());
                log.info("内存节省: {}%", String.format("%.1f", index.getMemorySavings()));
                log.info("\n索引统计:\n{}", index.getIndexStats());
            }

            db.close();
        } catch (Exception e) {
            log.error("示例2出错: {}", e.getMessage(), e);
        }
        cleanup("./example_with_compression");
    }

    /**
     * 示例3: 自定义压缩参数
     */
    private void demoCustomCompression() {
        log.info("\n==============================================");
        log.info("示例3: 自定义压缩参数");
        log.info("==============================================");

        int dimension = 512;

        // 创建自定义压缩配置
        // 使用128个子空间，每个子空间4维 (512/128=4)
        CompressionConfig customConfig = CompressionConfig.builder()
                .enabled(true)
                .type(CompressionConfig.CompressionType.HNSWPQ)
                .pqSubspaces(128)      // 更多子空间 = 更高精度，更低压缩比
                .pqBits(8)             // 256个聚类中心
                .pqIterations(50)      // 更多迭代 = 更好聚类质量
                .build();

        log.info("自定义配置: PQ子空间数={}, 位数={}, 迭代次数={}",
                customConfig.getPqSubspaces(),
                customConfig.getPqBits(),
                customConfig.getPqIterations());
        log.info("预期压缩比: {}x", String.format("%.2f", customConfig.getCompressionRatio(dimension)));
        log.info("预期内存节省: {}%", String.format("%.1f", customConfig.getMemorySavings(dimension)));

        // 创建数据库
        VectorDatabase db = new VectorDatabase.Builder()
                .withDimension(dimension)
                .withMaxElements(10000)
                .withStoragePath("./example_custom_compression")
                .withCompression(customConfig)  // 使用自定义配置
                .build();

        try {
            // 添加向量
            addSampleVectors(db, 1000, dimension);

            // 搜索
            float[] query = generateRandomVector(dimension);
            List<SearchResult> results = db.search(query, 10);

            log.info("实际压缩比: {}x", String.format("%.2f", db.getCompressionRatio()));
            log.info("搜索结果: {} 个", results.size());

            db.close();
        } catch (Exception e) {
            log.error("示例3出错: {}", e.getMessage(), e);
        }
        cleanup("./example_custom_compression");
    }

    /**
     * 示例4: 性能对比
     */
    private void demoPerformanceComparison() {
        log.info("\n==============================================");
        log.info("示例4: 压缩 vs 不压缩 性能对比");
        log.info("==============================================");

        int dimension = 512;
        int numVectors = 5000;
        int numQueries = 100;

        // 准备测试数据
        float[][] vectors = new float[numVectors][dimension];
        for (int i = 0; i < numVectors; i++) {
            vectors[i] = generateRandomVector(dimension);
        }

        float[][] queries = new float[numQueries][dimension];
        for (int i = 0; i < numQueries; i++) {
            queries[i] = generateRandomVector(dimension);
        }

        long buildTimeNoCompression = 0;
        long searchTimeNoCompression = 0;
        long buildTimeWithCompression = 0;
        long searchTimeWithCompression = 0;
        double compressionRatio = 1.0;

        // 测试无压缩
        log.info("\n--- 测试无压缩 ---");
        VectorDatabase dbNoCompression = new VectorDatabase.Builder()
                .withDimension(dimension)
                .withMaxElements(numVectors * 2)
                .withStoragePath("./perf_test_no_compression")
                .withCompressionEnabled(false)
                .build();

        try {
            long startTime = System.currentTimeMillis();
            for (int i = 0; i < numVectors; i++) {
                dbNoCompression.addVector(i, vectors[i]);
            }
            buildTimeNoCompression = System.currentTimeMillis() - startTime;

            startTime = System.nanoTime();
            for (float[] query : queries) {
                dbNoCompression.search(query, 10);
            }
            searchTimeNoCompression = (System.nanoTime() - startTime) / 1_000_000;

            log.info("构建时间: {} ms", buildTimeNoCompression);
            log.info("搜索时间: {} ms ({} 次查询)", searchTimeNoCompression, numQueries);
            log.info("平均每次查询: {} ms", String.format("%.2f", (double) searchTimeNoCompression / numQueries));

            dbNoCompression.close();
        } catch (Exception e) {
            log.error("无压缩测试出错: {}", e.getMessage(), e);
        }
        cleanup("./perf_test_no_compression");

        // 测试有压缩
        log.info("\n--- 测试有压缩 ---");
        VectorDatabase dbWithCompression = new VectorDatabase.Builder()
                .withDimension(dimension)
                .withMaxElements(numVectors * 2)
                .withStoragePath("./perf_test_with_compression")
                .withCompressionEnabled(true)
                .build();

        try {
            long startTime = System.currentTimeMillis();
            for (int i = 0; i < numVectors; i++) {
                dbWithCompression.addVector(i, vectors[i]);
            }
            buildTimeWithCompression = System.currentTimeMillis() - startTime;

            startTime = System.nanoTime();
            for (float[] query : queries) {
                dbWithCompression.search(query, 10);
            }
            searchTimeWithCompression = (System.nanoTime() - startTime) / 1_000_000;

            compressionRatio = dbWithCompression.getCompressionRatio();

            log.info("构建时间: {} ms", buildTimeWithCompression);
            log.info("搜索时间: {} ms ({} 次查询)", searchTimeWithCompression, numQueries);
            log.info("平均每次查询: {} ms", String.format("%.2f", (double) searchTimeWithCompression / numQueries));
            log.info("压缩比: {}x", String.format("%.2f", compressionRatio));

            dbWithCompression.close();
        } catch (Exception e) {
            log.error("有压缩测试出错: {}", e.getMessage(), e);
        }
        cleanup("./perf_test_with_compression");

        // 对比结果
        log.info("\n--- 对比结果 ---");
        System.out.printf("构建时间: 无压缩=%dms, 有压缩=%dms (%.2fx)%n",
                buildTimeNoCompression, buildTimeWithCompression,
                buildTimeNoCompression > 0 ? (double) buildTimeWithCompression / buildTimeNoCompression : 0);
        System.out.printf("搜索时间: 无压缩=%dms, 有压缩=%dms (%.2fx)%n",
                searchTimeNoCompression, searchTimeWithCompression,
                searchTimeNoCompression > 0 ? (double) searchTimeWithCompression / searchTimeNoCompression : 0);
        System.out.printf("内存节省: %.1f%%%n", (1.0 - 1.0 / compressionRatio) * 100);
    }

    /**
     * 添加示例向量
     */
    private void addSampleVectors(VectorDatabase db, int count, int dimension) {
        for (int i = 0; i < count; i++) {
            float[] vector = generateRandomVector(dimension);
            db.addVector(i, vector);
        }
        log.info("已添加 {} 个向量", count);
    }

    /**
     * 生成随机向量
     */
    private float[] generateRandomVector(int dimension) {
        Random random = new Random();
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = random.nextFloat() * 2 - 1; // -1 到 1
        }
        return vector;
    }

    /**
     * 清理测试目录
     */
    private void cleanup(String path) {
        java.io.File dir = new java.io.File(path);
        if (dir.exists()) {
            deleteDirectory(dir);
        }
    }

    private void deleteDirectory(java.io.File directory) {
        java.io.File[] files = directory.listFiles();
        if (files != null) {
            for (java.io.File file : files) {
                if (file.isDirectory()) {
                    deleteDirectory(file);
                } else {
                    file.delete();
                }
            }
        }
        directory.delete();
    }
}
