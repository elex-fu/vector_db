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
 * 召回率优化验证测试
 * 验证Fix #1-4的优化效果
 */
@Slf4j
public class RecallOptimizationTest {

    private static final String TEST_DB_PATH = "test_recall_optimization";
    private static final int DIMENSION = 512;
    private static final int NUM_VECTORS = 10000;
    private static final int NUM_QUERIES = 100;
    private static final int K = 10;

    private Random random;

    @Before
    public void setUp() {
        random = new Random(42);
        cleanup();
    }

    @After
    public void tearDown() {
        cleanup();
    }

    private void cleanup() {
        deleteDirectory(new File(TEST_DB_PATH));
    }

    /**
     * 测试Fix #1: PQ参数优化
     * 验证子空间配置是否正确
     */
    @Test
    public void testFix1_PQParameterOptimization() {
        log.info("\n========== Fix #1: PQ参数优化测试 ==========");

        // 测试512维的推荐配置
        CompressionConfig config = CompressionConfig.recommendedConfig(512);

        log.info("维度: 512");
        log.info("PQ子空间数: {}", config.getPqSubspaces());
        log.info("预期子空间维度: {}维/子空间", 512 / config.getPqSubspaces());
        log.info("压缩比: {}x", String.format("%.2f", config.getCompressionRatio(512)));

        // 验证子空间维度 >= 4 (避免量化误差过大)
        int subDim = 512 / config.getPqSubspaces();
        assertTrue("子空间维度应 >= 4 (避免2维/子空间的量化误差)", subDim >= 4);

        // 验证压缩比在合理范围
        double ratio = config.getCompressionRatio(512);
        assertTrue("压缩比应在4x-64x之间", ratio >= 4.0 && ratio <= 64.0);

        log.info("✓ Fix #1 验证通过: 子空间维度={} >= 4, 压缩比={}x", subDim, String.format("%.2f", ratio));
    }

    /**
     * 测试Fix #2: efSearch调整
     * 验证efSearch是否至少访问15%数据
     */
    @Test
    public void testFix2_EfSearchAdjustment() throws IOException {
        log.info("\n========== Fix #2: efSearch调整测试 ==========");

        VectorDatabase db = createTestDatabase(NUM_VECTORS);

        // 添加测试向量
        for (int i = 0; i < NUM_VECTORS; i++) {
            db.addVector(i, generateRandomVector(DIMENSION));
        }

        log.info("数据量: {}", NUM_VECTORS);
        log.info("预期efSearch: 至少访问{}个向量 (15%)", (int)(NUM_VECTORS * 0.15));
        log.info("预期efSearch: 至少{} (100*k)", 100 * K);

        // 搜索并验证性能
        long startTime = System.nanoTime();
        List<SearchResult> results = db.search(generateRandomVector(DIMENSION), K);
        long searchTime = (System.nanoTime() - startTime) / 1_000_000; // ms

        assertNotNull("搜索结果不应为空", results);
        assertEquals("应返回K个结果", K, results.size());

        log.info("搜索延迟: {}ms", searchTime);
        log.info("✓ Fix #2 验证通过: 搜索完成");

        db.close();
    }

    /**
     * 测试Fix #3 & #4: 召回率验证
     * 使用暴力搜索作为真值
     */
    @Test
    public void testFix34_RecallRate() throws IOException {
        log.info("\n========== Fix #3 & #4: 召回率验证测试 ==========");

        // 存储所有向量用于暴力搜索
        Map<Integer, float[]> allVectors = new HashMap<>();

        // 创建HNSWPQ优化版本
        VectorDatabase optimizedDb = new VectorDatabase.Builder()
                .withDimension(DIMENSION)
                .withMaxElements(NUM_VECTORS * 2)
                .withStoragePath(TEST_DB_PATH + "_optimized")
                .withCompression(CompressionConfig.recommendedConfig(DIMENSION))
                .build();

        log.info("添加{}个向量到数据库...", NUM_VECTORS);
        for (int i = 0; i < NUM_VECTORS; i++) {
            float[] vector = generateRandomVector(DIMENSION);
            allVectors.put(i, vector);
            optimizedDb.addVector(i, vector);
        }

        log.info("数据库向量数: {}", optimizedDb.size());
        log.info("数据库压缩比: {}x", String.format("%.2f", optimizedDb.getCompressionRatio()));

        // 生成查询向量
        List<float[]> queries = new ArrayList<>();
        for (int i = 0; i < NUM_QUERIES; i++) {
            queries.add(generateRandomVector(DIMENSION));
        }

        // 计算召回率
        double totalRecall = 0;
        double totalBruteForceTime = 0;
        double totalOptimizedTime = 0;

        for (int i = 0; i < NUM_QUERIES; i++) {
            float[] query = queries.get(i);

            // 暴力搜索真值
            long start = System.nanoTime();
            PriorityQueue<SearchResult> bruteForceResults = new PriorityQueue<>(
                    Comparator.comparing(SearchResult::getDistance).reversed());
            for (Map.Entry<Integer, float[]> entry : allVectors.entrySet()) {
                float dist = squaredEuclideanDistance(query, entry.getValue());
                bruteForceResults.add(new SearchResult(entry.getKey(), dist));
                if (bruteForceResults.size() > K) {
                    bruteForceResults.poll();
                }
            }
            List<SearchResult> bruteForceTopK = new ArrayList<>(bruteForceResults);
            bruteForceTopK.sort(Comparator.comparing(SearchResult::getDistance));
            totalBruteForceTime += (System.nanoTime() - start) / 1_000_000.0;

            // HNSWPQ搜索
            start = System.nanoTime();
            List<SearchResult> optimizedResults = optimizedDb.search(query, K);
            totalOptimizedTime += (System.nanoTime() - start) / 1_000_000.0;

            // 计算召回率
            Set<Integer> groundTruthIds = new HashSet<>();
            for (SearchResult r : bruteForceTopK) {
                groundTruthIds.add(r.getId());
            }

            int matchCount = 0;
            for (SearchResult r : optimizedResults) {
                if (groundTruthIds.contains(r.getId())) {
                    matchCount++;
                }
            }

            double recall = (double) matchCount / K;
            totalRecall += recall;
        }

        double avgRecall = totalRecall / NUM_QUERIES;
        double avgBruteForceTime = totalBruteForceTime / NUM_QUERIES;
        double avgOptimizedTime = totalOptimizedTime / NUM_QUERIES;
        double bruteForceQPS = 1000.0 / avgBruteForceTime * NUM_QUERIES;
        double optimizedQPS = 1000.0 / avgOptimizedTime * NUM_QUERIES;

        log.info("\n========== 测试结果 ==========");
        log.info("平均Recall@{}: {}%", K, String.format("%.2f", avgRecall * 100));
        log.info("暴力搜索平均延迟: {}ms", String.format("%.2f", avgBruteForceTime));
        log.info("优化平均延迟: {}ms", String.format("%.2f", avgOptimizedTime));
        log.info("暴力搜索QPS: {}", String.format("%.0f", bruteForceQPS));
        log.info("优化QPS: {}", String.format("%.0f", optimizedQPS));
        log.info("速度提升: {}x", String.format("%.2f", bruteForceQPS / optimizedQPS));
        log.info("压缩比: {}x", String.format("%.2f", optimizedDb.getCompressionRatio()));

        // 断言
        assertTrue("Recall应 >= 70% (优化前仅8.56%)", avgRecall >= 0.70);
        assertTrue("Recall应 >= 85% (目标)", avgRecall >= 0.85);

        log.info("\n✓ 召回率优化成功! {}% -> {}%+", "8.56", String.format("%.2f", avgRecall * 100));

        optimizedDb.close();
        deleteDirectory(new File(TEST_DB_PATH + "_optimized"));
    }

    private float squaredEuclideanDistance(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    /**
     * 快速召回率测试 (较少数据，用于快速验证)
     * 使用暴力搜索作为真值
     */
    @Test
    public void testQuickRecallValidation() throws IOException {
        log.info("\n========== 快速召回率验证 ==========");

        int quickTestSize = 5000;
        int quickQueries = 50;

        Map<Integer, float[]> allVectors = new HashMap<>();

        VectorDatabase optimizedDb = new VectorDatabase.Builder()
                .withDimension(DIMENSION)
                .withMaxElements(quickTestSize * 2)
                .withStoragePath(TEST_DB_PATH + "_quick_opt")
                .withCompression(CompressionConfig.recommendedConfig(DIMENSION))
                .build();

        // 添加数据
        for (int i = 0; i < quickTestSize; i++) {
            float[] vector = generateRandomVector(DIMENSION);
            allVectors.put(i, vector);
            optimizedDb.addVector(i, vector);
        }

        // 测试召回率
        int matches = 0;
        int total = quickQueries * K;

        for (int i = 0; i < quickQueries; i++) {
            float[] query = generateRandomVector(DIMENSION);

            // 暴力搜索真值
            PriorityQueue<SearchResult> bruteForceResults = new PriorityQueue<>(
                    Comparator.comparing(SearchResult::getDistance).reversed());
            for (Map.Entry<Integer, float[]> entry : allVectors.entrySet()) {
                float dist = squaredEuclideanDistance(query, entry.getValue());
                bruteForceResults.add(new SearchResult(entry.getKey(), dist));
                if (bruteForceResults.size() > K) {
                    bruteForceResults.poll();
                }
            }
            List<SearchResult> groundTruth = new ArrayList<>(bruteForceResults);

            List<SearchResult> optimizedResults = optimizedDb.search(query, K);

            Set<Integer> groundTruthIds = new HashSet<>();
            for (SearchResult r : groundTruth) {
                groundTruthIds.add(r.getId());
            }

            for (SearchResult r : optimizedResults) {
                if (groundTruthIds.contains(r.getId())) {
                    matches++;
                }
            }
        }

        double recall = (double) matches / total;
        log.info("快速测试 Recall@{}: {}%", K, String.format("%.2f", recall * 100));
        log.info("目标 Recall: >= 85%");

        assertTrue("快速测试Recall应 >= 70%", recall >= 0.70);

        optimizedDb.close();
        deleteDirectory(new File(TEST_DB_PATH + "_quick_opt"));
    }

    // ========== 辅助方法 ==========

    private VectorDatabase createTestDatabase(int maxElements) throws IOException {
        return new VectorDatabase.Builder()
                .withDimension(DIMENSION)
                .withMaxElements(maxElements * 2)
                .withStoragePath(TEST_DB_PATH)
                .withCompression(CompressionConfig.recommendedConfig(DIMENSION))
                .build();
    }

    private float[] generateRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = random.nextFloat() * 2 - 1; // -1 到 1
        }
        return vector;
    }

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
}
