package com.vectordb.benchmark;

import com.vectordb.config.CompressionConfig;
import com.vectordb.core.SearchResult;
import com.vectordb.core.Vector;
import com.vectordb.core.VectorDatabase;
import com.vectordb.index.HnswPqIndex;
import lombok.extern.slf4j.Slf4j;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.*;

import static org.junit.Assert.*;

/**
 * 召回率问题诊断测试
 */
@Slf4j
public class RecallDiagnosticTest {

    private static final String TEST_DB_PATH = "test_recall_diagnostic";
    private static final int DIMENSION = 128;  // 使用较小维度便于测试
    private static final int NUM_VECTORS = 1000;
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
     * 诊断测试1: 检查基础功能是否正常
     */
    @Test
    public void testBasicFunctionality() throws IOException {
        log.info("\n========== 诊断测试1: 基础功能 ==========");

        VectorDatabase db = new VectorDatabase.Builder()
                .withDimension(DIMENSION)
                .withMaxElements(NUM_VECTORS * 2)
                .withStoragePath(TEST_DB_PATH)
                .withCompression(CompressionConfig.highRecallConfig(DIMENSION))
                .build();

        log.info("索引类型: {}", db.getIndexType());
        log.info("是否启用压缩: {}", db.isCompressionEnabled());
        log.info("压缩比: {}x", String.format("%.2f", db.getCompressionRatio()));

        if (db.getIndex() instanceof HnswPqIndex) {
            HnswPqIndex index = (HnswPqIndex) db.getIndex();
            log.info("是否已训练: {}", index.isTrained());
        }

        // 添加一些向量
        log.info("添加{}个向量...", NUM_VECTORS);
        for (int i = 0; i < NUM_VECTORS; i++) {
            float[] vector = generateRandomVector(DIMENSION);
            db.addVector(i, vector);
        }

        log.info("数据库中向量数: {}", db.size());

        if (db.getIndex() instanceof HnswPqIndex) {
            HnswPqIndex index = (HnswPqIndex) db.getIndex();
            log.info("训练后是否已训练: {}", index.isTrained());
        }

        // 搜索测试
        float[] query = generateRandomVector(DIMENSION);
        List<SearchResult> results = db.search(query, K);

        log.info("搜索结果数量: {}", results.size());
        if (!results.isEmpty()) {
            log.info("第一个结果ID: {}, 距离: {}",
                    results.get(0).getId(),
                    String.format("%.4f", results.get(0).getDistance()));
        }

        assertTrue("应该有搜索结果", !results.isEmpty());
        db.close();
    }

    /**
     * 诊断测试2: 暴力搜索对比
     */
    @Test
    public void testBruteForceComparison() throws IOException {
        log.info("\n========== 诊断测试2: 暴力搜索对比 ==========");

        // 创建内存中的向量
        Map<Integer, float[]> vectorMap = new HashMap<>();
        for (int i = 0; i < NUM_VECTORS; i++) {
            vectorMap.put(i, generateRandomVector(DIMENSION));
        }

        // 创建数据库
        VectorDatabase db = new VectorDatabase.Builder()
                .withDimension(DIMENSION)
                .withMaxElements(NUM_VECTORS * 2)
                .withStoragePath(TEST_DB_PATH)
                .withCompression(CompressionConfig.highRecallConfig(DIMENSION))
                .build();

        // 添加向量
        for (Map.Entry<Integer, float[]> entry : vectorMap.entrySet()) {
            db.addVector(entry.getKey(), entry.getValue());
        }

        // 测试查询
        float[] query = generateRandomVector(DIMENSION);

        // 暴力搜索Top-10 (使用最大堆来保留最小的K个)
        PriorityQueue<SearchResult> bruteForceResults = new PriorityQueue<>(
                Comparator.comparing(SearchResult::getDistance).reversed());

        for (Map.Entry<Integer, float[]> entry : vectorMap.entrySet()) {
            float dist = euclideanDistance(query, entry.getValue());
            bruteForceResults.add(new SearchResult(entry.getKey(), dist));
            if (bruteForceResults.size() > K) {
                bruteForceResults.poll(); // 移除最远的（最大堆的堆顶）
            }
        }

        List<SearchResult> bruteForceTopK = new ArrayList<>(bruteForceResults);
        bruteForceTopK.sort(Comparator.comparing(SearchResult::getDistance));

        // HNSWPQ搜索
        List<SearchResult> hnswpqResults = db.search(query, K);

        // 调试：检查ID=5在暴力搜索中的位置
        float[] vec5 = vectorMap.get(5);
        if (vec5 != null) {
            float dist5 = euclideanDistance(query, vec5);
            log.info("调试 - ID=5在暴力搜索中的距离: {}", String.format("%.4f", dist5));
        }

        // 调试：验证HNSWPQ搜索返回的第一个结果的实际距离
        if (!hnswpqResults.isEmpty()) {
            SearchResult firstResult = hnswpqResults.get(0);
            SearchResult bfResult = bruteForceTopK.get(0);
            float[] actualVector = vectorMap.get(firstResult.getId());
            float[] bfActualVector = vectorMap.get(bfResult.getId());
            if (actualVector != null) {
                float actualDist = euclideanDistance(query, actualVector);
                log.info("调试 - HNSWPQ第一个结果ID: {}, 报告距离: {}, 实际计算距离: {}",
                        firstResult.getId(), String.format("%.4f", firstResult.getDistance()),
                        String.format("%.4f", actualDist));
                // 验证暴力搜索第一个结果的实际距离
                float bfDist = euclideanDistance(query, bfActualVector);
                log.info("调试 - 暴力搜索第一个结果ID: {}, 报告距离: {}, 实际计算距离: {}",
                        bfResult.getId(), String.format("%.4f", bfResult.getDistance()),
                        String.format("%.4f", bfDist));
            }
        }

        log.info("暴力搜索Top-10:");
        for (int i = 0; i < Math.min(5, bruteForceTopK.size()); i++) {
            SearchResult r = bruteForceTopK.get(i);
            log.info("  {}: ID={}, 距离={}", i, r.getId(), String.format("%.4f", r.getDistance()));
        }

        log.info("HNSWPQ搜索Top-10:");
        for (int i = 0; i < Math.min(5, hnswpqResults.size()); i++) {
            SearchResult r = hnswpqResults.get(i);
            log.info("  {}: ID={}, 距离={}", i, r.getId(), String.format("%.4f", r.getDistance()));
        }

        // 计算召回率
        Set<Integer> bruteForceIds = new HashSet<>();
        for (SearchResult r : bruteForceTopK) {
            bruteForceIds.add(r.getId());
        }

        int matches = 0;
        for (SearchResult r : hnswpqResults) {
            if (bruteForceIds.contains(r.getId())) {
                matches++;
            }
        }

        double recall = (double) matches / K;
        log.info("Recall@{}: {}%", K, String.format("%.2f", recall * 100));

        db.close();
    }

    /**
     * 诊断测试3: 小数据集精确验证
     */
    @Test
    public void testSmallDatasetExactMatch() throws IOException {
        log.info("\n========== 诊断测试3: 小数据集精确验证 ==========");

        int smallSize = 100;

        // 使用固定种子确保可重现
        Random fixedRandom = new Random(12345);

        VectorDatabase db = new VectorDatabase.Builder()
                .withDimension(DIMENSION)
                .withMaxElements(smallSize * 2)
                .withStoragePath(TEST_DB_PATH)
                .withCompression(CompressionConfig.highRecallConfig(DIMENSION))
                .build();

        // 添加100个向量
        for (int i = 0; i < smallSize; i++) {
            float[] vector = new float[DIMENSION];
            for (int j = 0; j < DIMENSION; j++) {
                vector[j] = fixedRandom.nextFloat();
            }
            db.addVector(i, vector);
        }

        // 使用第0个向量作为查询（应该能找到自己）
        float[] query = new float[DIMENSION];
        fixedRandom = new Random(12345);  // 重置随机数生成器
        for (int j = 0; j < DIMENSION; j++) {
            query[j] = fixedRandom.nextFloat();
        }

        List<SearchResult> results = db.search(query, K);

        log.info("查询向量ID: 0 (第一个添加的向量)");
        log.info("搜索结果:");
        boolean foundSelf = false;
        for (int i = 0; i < results.size(); i++) {
            SearchResult r = results.get(i);
            log.info("  {}: ID={}, 距离={}", i, r.getId(), String.format("%.4f", r.getDistance()));
            if (r.getId() == 0) {
                foundSelf = true;
            }
        }

        log.info("是否找到自身: {}", foundSelf);

        // 应该能找到自己（距离为0）
        assertTrue("应该能找到查询向量自身", foundSelf);

        db.close();
    }

    // ========== 辅助方法 ==========

    private float[] generateRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = random.nextFloat() * 2 - 1;
        }
        return vector;
    }

    private float euclideanDistance(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
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
