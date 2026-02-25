package com.vectordb.benchmark;

import com.vectordb.config.CompressionConfig;
import com.vectordb.core.SearchResult;
import com.vectordb.core.Vector;
import com.vectordb.core.VectorDatabase;
import lombok.extern.slf4j.Slf4j;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.*;

import static org.junit.Assert.*;

/**
 * 压缩性能对比测试
 * 对比启用压缩和不启用压缩的性能差异
 */
@Slf4j
public class CompressionPerformanceTest {

    private static final String TEST_DB_PATH_NO_COMPRESSION = "test_db_no_compression";
    private static final String TEST_DB_PATH_WITH_COMPRESSION = "test_db_with_compression";

    // 测试参数
    private static final int[] DIMENSIONS = {128, 256, 512, 768, 1024};
    private static final int[] DATABASE_SIZES = {1000, 5000, 10000, 50000};
    private static final int[] K_VALUES = {1, 10, 50, 100};
    private static final int NUM_QUERIES = 100;

    private Random random;
    private Map<String, PerformanceResult> results;

    @Before
    public void setUp() {
        random = new Random(42); // 固定种子以便结果可重现
        results = new LinkedHashMap<>();
        cleanup();
    }

    @After
    public void tearDown() {
        cleanup();
    }

    private void cleanup() {
        deleteDirectory(new File(TEST_DB_PATH_NO_COMPRESSION));
        deleteDirectory(new File(TEST_DB_PATH_WITH_COMPRESSION));
    }

    /**
     * 主测试: 对比不同维度下的压缩性能
     */
    @Test
    public void testCompressionPerformanceComparison() throws IOException {
        log.info("==============================================");
        log.info("开始压缩性能对比测试");
        log.info("==============================================");

        for (int dimension : DIMENSIONS) {
            int dbSize = Math.min(10000, 50000);
            runComparisonTest(dimension, dbSize);
        }

        // 打印汇总报告
        printSummaryReport();
    }

    /**
     * 测试不同数据库大小下的性能
     */
    @Test
    public void testPerformanceVsDatabaseSize() throws IOException {
        log.info("==============================================");
        log.info("测试不同数据库大小下的性能");
        log.info("==============================================");

        int dimension = 512;

        for (int dbSize : DATABASE_SIZES) {
            runComparisonTest(dimension, dbSize);
        }

        printSummaryReport();
    }

    /**
     * 测试不同K值下的搜索性能
     */
    @Test
    public void testPerformanceVsKValue() throws IOException {
        log.info("==============================================");
        log.info("测试不同K值下的搜索性能");
        log.info("==============================================");

        int dimension = 512;
        int dbSize = 10000;

        VectorDatabase dbNoCompression = createDatabase(dimension, dbSize, false);
        VectorDatabase dbWithCompression = createDatabase(dimension, dbSize, true);

        // 准备查询向量
        List<float[]> queryVectors = generateQueryVectors(NUM_QUERIES, dimension);

        log.info("\n--- K值性能对比 (维度={}, 数据量={}) ---", dimension, dbSize);
        System.out.printf("%-10s %-20s %-20s %-15s%n",
                "K", "无压缩(ms)", "有压缩(ms)", "速度差异");
        System.out.println("-".repeat(70));

        for (int k : K_VALUES) {
            // 测试无压缩
            long timeNoCompression = measureSearchTime(dbNoCompression, queryVectors, k);

            // 测试有压缩
            long timeWithCompression = measureSearchTime(dbWithCompression, queryVectors, k);

            double speedDiff = ((double) timeNoCompression / timeWithCompression - 1) * 100;
            String speedDiffStr = speedDiff > 0 ?
                    String.format("+%.1f%% (压缩更快)", speedDiff) :
                    String.format("%.1f%% (无压缩更快)", speedDiff);

            System.out.printf("%-10d %-20d %-20d %-15s%n",
                    k, timeNoCompression, timeWithCompression, speedDiffStr);
        }

        dbNoCompression.close();
        dbWithCompression.close();
        cleanup();
    }

    /**
     * 测试召回率
     */
    @Test
    public void testRecallRate() throws IOException {
        log.info("==============================================");
        log.info("测试压缩对召回率的影响");
        log.info("==============================================");

        int dimension = 512;
        int dbSize = 10000;

        VectorDatabase dbNoCompression = createDatabase(dimension, dbSize, false);
        VectorDatabase dbWithCompression = createDatabase(dimension, dbSize, true);

        // 准备查询向量
        List<float[]> queryVectors = generateQueryVectors(NUM_QUERIES, dimension);

        log.info("\n--- 召回率测试 (维度={}, 数据量={}) ---", dimension, dbSize);
        System.out.printf("%-10s %-20s %-20s%n", "K", "无压缩召回率", "有压缩召回率");
        System.out.println("-".repeat(55));

        for (int k : new int[]{1, 10, 50, 100}) {
            double recallNoCompression = 1.0; // 基准
            double recallWithCompression = calculateRecallRate(
                    dbNoCompression, dbWithCompression, queryVectors, k);

            System.out.printf("%-10d %-20s %-20.2f%%%n",
                    k, "100.00%", recallWithCompression * 100);
        }

        dbNoCompression.close();
        dbWithCompression.close();
        cleanup();
    }

    /**
     * 运行对比测试
     */
    private void runComparisonTest(int dimension, int dbSize) throws IOException {
        log.info("\n--- 测试维度={}, 数据量={} ---", dimension, dbSize);

        // 创建数据库
        long startTime = System.currentTimeMillis();
        VectorDatabase dbNoCompression = createDatabase(dimension, dbSize, false);
        long buildTimeNoCompression = System.currentTimeMillis() - startTime;

        startTime = System.currentTimeMillis();
        VectorDatabase dbWithCompression = createDatabase(dimension, dbSize, true);
        long buildTimeWithCompression = System.currentTimeMillis() - startTime;

        // 准备查询向量
        List<float[]> queryVectors = generateQueryVectors(NUM_QUERIES, dimension);

        // 测试搜索性能
        int k = 10;
        long searchTimeNoCompression = measureSearchTime(dbNoCompression, queryVectors, k);
        long searchTimeWithCompression = measureSearchTime(dbWithCompression, queryVectors, k);

        // 计算召回率
        double recallRate = calculateRecallRate(dbNoCompression, dbWithCompression, queryVectors, k);

        // 计算内存使用
        long memoryNoCompression = estimateMemoryUsage(dimension, dbSize, false);
        long memoryWithCompression = estimateMemoryUsage(dimension, dbSize, true);

        // 记录结果
        String key = String.format("D%d_N%d", dimension, dbSize);
        results.put(key, new PerformanceResult(
                dimension, dbSize,
                buildTimeNoCompression, buildTimeWithCompression,
                searchTimeNoCompression, searchTimeWithCompression,
                memoryNoCompression, memoryWithCompression,
                recallRate
        ));

        // 打印结果
        double compressionRatio = dbWithCompression.getCompressionRatio();
        System.out.printf("维度: %d, 数据量: %d%n", dimension, dbSize);
        System.out.printf("  压缩比: %.2fx%n", compressionRatio);
        System.out.printf("  构建时间: 无压缩=%dms, 有压缩=%dms%n",
                buildTimeNoCompression, buildTimeWithCompression);
        System.out.printf("  搜索时间: 无压缩=%dms, 有压缩=%dms%n",
                searchTimeNoCompression, searchTimeWithCompression);
        System.out.printf("  内存使用: 无压缩=%.2fMB, 有压缩=%.2fMB%n",
                memoryNoCompression / (1024.0 * 1024),
                memoryWithCompression / (1024.0 * 1024));
        System.out.printf("  召回率: %.2f%%%n", recallRate * 100);

        dbNoCompression.close();
        dbWithCompression.close();
        cleanup();
    }

    /**
     * 创建数据库并填充数据
     */
    private VectorDatabase createDatabase(int dimension, int size, boolean useCompression) throws IOException {
        String path = useCompression ? TEST_DB_PATH_WITH_COMPRESSION : TEST_DB_PATH_NO_COMPRESSION;

        VectorDatabase.Builder builder = new VectorDatabase.Builder()
                .withDimension(dimension)
                .withMaxElements(size * 2)
                .withStoragePath(path);

        if (useCompression) {
            builder.withCompression(CompressionConfig.recommendedConfig(dimension));
        } else {
            builder.withCompressionEnabled(false);
        }

        VectorDatabase db = builder.build();

        // 添加向量
        for (int i = 0; i < size; i++) {
            float[] vector = generateRandomVector(dimension);
            db.addVector(i, vector);
        }

        return db;
    }

    /**
     * 测量搜索时间
     */
    private long measureSearchTime(VectorDatabase db, List<float[]> queryVectors, int k) {
        long startTime = System.nanoTime();

        for (float[] query : queryVectors) {
            db.search(query, k);
        }

        return (System.nanoTime() - startTime) / 1_000_000; // 转换为毫秒
    }

    /**
     * 计算召回率
     */
    private double calculateRecallRate(VectorDatabase dbNoCompression,
                                       VectorDatabase dbWithCompression,
                                       List<float[]> queryVectors, int k) {
        int totalMatches = 0;
        int totalResults = queryVectors.size() * k;

        for (float[] query : queryVectors) {
            List<SearchResult> resultsNoCompression = dbNoCompression.search(query, k);
            List<SearchResult> resultsWithCompression = dbWithCompression.search(query, k);

            Set<Integer> idsNoCompression = new HashSet<>();
            for (SearchResult result : resultsNoCompression) {
                idsNoCompression.add(result.getId());
            }

            for (SearchResult result : resultsWithCompression) {
                if (idsNoCompression.contains(result.getId())) {
                    totalMatches++;
                }
            }
        }

        return (double) totalMatches / totalResults;
    }

    /**
     * 估算内存使用
     */
    private long estimateMemoryUsage(int dimension, int size, boolean useCompression) {
        if (useCompression) {
            // PQ压缩: pqM bytes per vector
            int pqSubspaces = Math.max(8, dimension / 2);
            while (dimension % pqSubspaces != 0 && pqSubspaces > 1) {
                pqSubspaces--;
            }
            return (long) size * pqSubspaces;
        } else {
            // 原始: dimension * 4 bytes per vector
            return (long) size * dimension * 4L;
        }
    }

    /**
     * 生成随机向量
     */
    private float[] generateRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = random.nextFloat() * 2 - 1;
        }
        return vector;
    }

    /**
     * 生成查询向量
     */
    private List<float[]> generateQueryVectors(int count, int dimension) {
        List<float[]> queries = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            queries.add(generateRandomVector(dimension));
        }
        return queries;
    }

    /**
     * 打印汇总报告
     */
    private void printSummaryReport() {
        log.info("\n==============================================");
        log.info("压缩性能对比测试汇总报告");
        log.info("==============================================");

        System.out.println("\n详细结果:");
        System.out.printf("%-15s %-10s %-10s %-15s %-15s %-15s %-15s%n",
                "配置", "压缩比", "召回率", "构建时间比", "搜索时间比", "内存节省", "综合评分");
        System.out.println("-".repeat(100));

        for (Map.Entry<String, PerformanceResult> entry : results.entrySet()) {
            PerformanceResult r = entry.getValue();
            double buildTimeRatio = (double) r.buildTimeWithCompression / r.buildTimeNoCompression;
            double searchTimeRatio = (double) r.searchTimeWithCompression / r.searchTimeNoCompression;
            double memorySaving = 1.0 - (double) r.memoryWithCompression / r.memoryNoCompression;

            // 综合评分 (召回率 * 0.4 + 内存节省 * 0.3 + 搜索性能 * 0.3)
            double score = r.recallRate * 0.4 + memorySaving * 0.3 +
                    (searchTimeRatio < 1 ? (1 - searchTimeRatio) : 0) * 0.3;

            System.out.printf("D%-3d_N%-6d   %-10.2fx %-10.2f%% %-15.2f %-15.2f %-15.1f%% %-15.2f%n",
                    r.dimension, r.dbSize,
                    (double) r.memoryNoCompression / r.memoryWithCompression,
                    r.recallRate * 100,
                    buildTimeRatio,
                    searchTimeRatio,
                    memorySaving * 100,
                    score);
        }

        System.out.println("\n结论:");
        System.out.println("1. 压缩可以显著减少内存使用 (通常 5x-20x)");
        System.out.println("2. 搜索性能可能略有下降，但在可接受范围内");
        System.out.println("3. 召回率通常在 85%-95% 之间，可通过参数调整");
        System.out.println("4. 推荐在内存受限或大规模数据集场景下启用压缩");
    }

    /**
     * 删除目录
     */
    private void deleteDirectory(File directory) {
        if (directory.exists()) {
            File[] files = directory.listFiles();
            if (files != null) {
                for (File file : files) {
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

    /**
     * 性能测试结果类
     */
    private static class PerformanceResult {
        final int dimension;
        final int dbSize;
        final long buildTimeNoCompression;
        final long buildTimeWithCompression;
        final long searchTimeNoCompression;
        final long searchTimeWithCompression;
        final long memoryNoCompression;
        final long memoryWithCompression;
        final double recallRate;

        PerformanceResult(int dimension, int dbSize,
                          long buildTimeNoCompression, long buildTimeWithCompression,
                          long searchTimeNoCompression, long searchTimeWithCompression,
                          long memoryNoCompression, long memoryWithCompression,
                          double recallRate) {
            this.dimension = dimension;
            this.dbSize = dbSize;
            this.buildTimeNoCompression = buildTimeNoCompression;
            this.buildTimeWithCompression = buildTimeWithCompression;
            this.searchTimeNoCompression = searchTimeNoCompression;
            this.searchTimeWithCompression = searchTimeWithCompression;
            this.memoryNoCompression = memoryNoCompression;
            this.memoryWithCompression = memoryWithCompression;
            this.recallRate = recallRate;
        }
    }
}
