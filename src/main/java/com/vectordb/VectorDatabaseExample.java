package com.vectordb;

import com.vectordb.core.SearchResult;
import com.vectordb.core.Vector;
import com.vectordb.core.VectorDatabase;
import com.vectordb.core.VectorDatabase.IndexType;
import com.vectordb.index.AnnoyIndex;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;

/**
 * 向量数据库示例
 * 演示向量数据库的基本操作
 */
public class VectorDatabaseExample {
    // 向量维度
    private static final int VECTOR_DIMENSION = 128;
    
    // 向量数量
    private static final int VECTOR_COUNT = 10000;
    
    // 随机数生成器
    private static final Random random = new Random(42); // 使用固定种子以保证结果可重现
    
    public static void main(String[] args) {
        // 创建数据库目录
        File dbDir = new File("vector_db");
        if (!dbDir.exists()) {
            dbDir.mkdirs();
        }
        
        // 测试HNSW索引
        System.out.println("=== 测试HNSW索引 ===");
        testVectorDatabase(IndexType.HNSW);
        
        // 测试ANNOY索引
        System.out.println("\n=== 测试ANNOY索引 ===");
        testVectorDatabase(IndexType.ANNOY);
        
        // 测试优化后的ANNOY索引
        System.out.println("\n=== 测试优化后的ANNOY索引 ===");
        testOptimizedAnnoy();
        
        // 测试LSH索引
        System.out.println("\n=== 测试LSH索引 ===");
        testVectorDatabase(IndexType.LSH);
        
        // 测试IVF索引
        System.out.println("\n=== 测试IVF索引 ===");
        testVectorDatabase(IndexType.IVF);
        
        // 测试PQ索引
        System.out.println("\n=== 测试PQ索引 ===");
        testVectorDatabase(IndexType.PQ);
        
        // 比较所有索引类型的性能
        System.out.println("\n=== 索引性能比较 ===");
        compareIndexPerformance();
    }
    
    /**
     * 测试向量数据库
     * 
     * @param indexType 索引类型
     */
    private static void testVectorDatabase(IndexType indexType) {
        // 创建数据库目录
        String dbPath = "vector_data/vector_db_" + indexType.name().toLowerCase();
        File dbDir = new File(dbPath);
        if (!dbDir.exists()) {
            dbDir.mkdirs();
        }
        
        System.out.println("初始化向量数据库...");
        
        // 初始化向量数据库，使用Builder模式
        try (VectorDatabase db = new VectorDatabase.Builder()
                .withStoragePath(dbDir.getPath())
                .withDimension(VECTOR_DIMENSION)
                .withMaxElements(VECTOR_COUNT)
                .withIndexType(indexType)
                .build()) {
            
            System.out.println("向量数据库初始化完成，使用索引: " + db.getIndexType());
            
            // 添加随机向量
            System.out.println("添加随机向量...");
            long startTime = System.currentTimeMillis();
            
            for (int i = 1; i <= VECTOR_COUNT; i++) {
                float[] values = generateRandomVector(VECTOR_DIMENSION);
                try {
                    db.addVector(i, values);
                } catch (Exception e) {
                    System.err.println("添加向量时出错 (ID: " + i + "): " + e.getMessage());
                }
                
                if (i % 1000 == 0) {
                    System.out.println("已添加 " + i + " 个向量");
                }
            }
            
            long endTime = System.currentTimeMillis();
            System.out.println("添加 " + VECTOR_COUNT + " 个向量耗时: " + (endTime - startTime) + " 毫秒");
            
            // 获取存储文件大小
            File vectorFile = new File(dbDir, "vectors.json");
            if (vectorFile.exists()) {
                double fileSizeMB = vectorFile.length() / (1024.0 * 1024.0);
                System.out.printf("存储文件大小: %.2f MB\n", fileSizeMB);
            }
            
            // 重建索引测试
            System.out.println("\n执行重建索引测试...");
            startTime = System.currentTimeMillis();
            boolean rebuildSuccess = db.rebuildIndex();
            endTime = System.currentTimeMillis();
            
            System.out.println("重建索引" + (rebuildSuccess ? "成功" : "失败") + 
                    "，耗时: " + (endTime - startTime) + " 毫秒");
            
            // 搜索测试
            System.out.println("\n执行搜索测试...");
            float[] queryVector = generateRandomVector(VECTOR_DIMENSION);
            
            startTime = System.currentTimeMillis();
            List<SearchResult> results = db.search(queryVector, 10);
            endTime = System.currentTimeMillis();
            
            System.out.println("搜索耗时: " + (endTime - startTime) + " 毫秒");
            System.out.println("搜索结果:");
            
            for (int i = 0; i < results.size(); i++) {
                SearchResult result = results.get(i);
                System.out.println((i + 1) + ". ID: " + result.getId() + 
                        ", 距离: " + result.getDistance() + 
                        ", 相似度: " + result.getSimilarity());
            }
            
            // 删除测试
            System.out.println("\n执行删除测试...");
            int idToDelete = 100;
            
            startTime = System.currentTimeMillis();
            boolean deleted = db.deleteVector(idToDelete);
            endTime = System.currentTimeMillis();
            
            System.out.println("删除向量 ID: " + idToDelete + " " + 
                    (deleted ? "成功" : "失败") + ", 耗时: " + (endTime - startTime) + " 毫秒");
            
            // 验证删除
            Optional<Vector> deletedVector = db.getVector(idToDelete);
            System.out.println("验证删除: 向量 ID: " + idToDelete + " " + 
                    (deletedVector.isPresent() ? "仍然存在" : "已被删除"));
            
            // 再次搜索，确认删除的向量不会出现在结果中
            results = db.search(queryVector, 10);
            boolean containsDeleted = results.stream().anyMatch(r -> r.getId() == idToDelete);
            System.out.println("搜索结果中" + (containsDeleted ? "包含" : "不包含") + "已删除的向量");
        } catch (Exception e) {
            System.err.println("操作向量数据库时出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试优化后的ANNOY索引
     */
    private static void testOptimizedAnnoy() {
        System.out.println("初始化优化后的ANNOY索引...");
        
        // 创建ANNOY索引实例
        AnnoyIndex annoyIndex = new AnnoyIndex(VECTOR_DIMENSION, VECTOR_COUNT * 2);
        
        // 设置ANNOY索引参数
        System.out.println("设置ANNOY索引参数...");
        annoyIndex.setIndexParameters(5, 20, true);
        
        // 准备批量添加的向量
        List<Vector> batchVectors = new ArrayList<>();
        for (int i = 1; i <= VECTOR_COUNT; i++) {
            float[] values = generateRandomVector(VECTOR_DIMENSION);
            batchVectors.add(new Vector(i, values));
        }
        
        // 批量添加向量
        System.out.println("批量添加向量...");
        long startTime = System.currentTimeMillis();
        int addedCount = annoyIndex.addVectors(batchVectors);
        long endTime = System.currentTimeMillis();
        
        System.out.println("批量添加 " + addedCount + " 个向量耗时: " + (endTime - startTime) + " 毫秒");
        
        // 搜索测试
        System.out.println("\n执行搜索测试...");
        Vector queryVector = new Vector(-1, generateRandomVector(VECTOR_DIMENSION));
        
        startTime = System.currentTimeMillis();
        List<SearchResult> results = annoyIndex.searchNearest(queryVector, 10);
        endTime = System.currentTimeMillis();
        
        System.out.println("搜索耗时: " + (endTime - startTime) + " 毫秒");
        System.out.println("搜索结果:");
        
        for (int i = 0; i < results.size(); i++) {
            SearchResult result = results.get(i);
            System.out.println((i + 1) + ". ID: " + result.getId() + 
                    ", 距离: " + result.getDistance() + 
                    ", 相似度: " + result.getSimilarity());
        }
        
        // 单个添加向量测试
        System.out.println("\n执行单个添加向量测试...");
        for (int i = VECTOR_COUNT + 1; i <= VECTOR_COUNT + 10; i++) {
            float[] values = generateRandomVector(VECTOR_DIMENSION);
            Vector vector = new Vector(i, values);
            
            startTime = System.currentTimeMillis();
            boolean added = annoyIndex.addVector(vector);
            endTime = System.currentTimeMillis();
            
            System.out.println("添加向量 ID: " + i + " " + 
                    (added ? "成功" : "失败") + ", 耗时: " + (endTime - startTime) + " 毫秒");
        }
        
        // 重建索引测试
        System.out.println("\n执行重建索引测试...");
        startTime = System.currentTimeMillis();
        boolean rebuildSuccess = annoyIndex.buildIndex();
        endTime = System.currentTimeMillis();
        
        System.out.println("重建索引" + (rebuildSuccess ? "成功" : "失败") + 
                "，耗时: " + (endTime - startTime) + " 毫秒");
        
        // 删除测试
        System.out.println("\n执行删除测试...");
        int idToDelete = 50;
        
        startTime = System.currentTimeMillis();
        boolean deleted = annoyIndex.removeVector(idToDelete);
        endTime = System.currentTimeMillis();
        
        System.out.println("删除向量 ID: " + idToDelete + " " + 
                (deleted ? "成功" : "失败") + ", 耗时: " + (endTime - startTime) + " 毫秒");
        
        // 再次搜索，确认删除的向量不会出现在结果中
        results = annoyIndex.searchNearest(queryVector, 10);
        boolean containsDeleted = results.stream().anyMatch(r -> r.getId() == idToDelete);
        System.out.println("搜索结果中" + (containsDeleted ? "包含" : "不包含") + "已删除的向量");
    }
    
    /**
     * 比较不同索引类型的性能
     */
    private static void compareIndexPerformance() {
        System.out.println("比较不同索引类型的性能（向量数量：" + VECTOR_COUNT + "，维度：" + VECTOR_DIMENSION + "）");
        
        // 准备测试数据
        List<Vector> testVectors = new ArrayList<>();
        for (int i = 1; i <= 100; i++) {
            testVectors.add(new Vector(i, generateRandomVector(VECTOR_DIMENSION)));
        }
        float[] queryVectorValues = generateRandomVector(VECTOR_DIMENSION);
        
        // 测试各种索引类型
        IndexType[] indexTypes = IndexType.values();
        for (IndexType indexType : indexTypes) {
            long startTime = System.currentTimeMillis();
            
            // 创建数据库
            try (VectorDatabase db = new VectorDatabase.Builder()
                    .withDimension(VECTOR_DIMENSION)
                    .withMaxElements(VECTOR_COUNT)
                    .withStoragePath("vector_db/" + indexType.name().toLowerCase())
                    .withIndexType(indexType)
                    .build()) {
                
                // 添加向量
                for (Vector vector : testVectors) {
                    db.addVector(vector.getId(), vector.getValues());
                }
                
                // 重建索引
                long rebuildStart = System.currentTimeMillis();
                boolean rebuildSuccess = db.rebuildIndex();
                long rebuildTime = System.currentTimeMillis() - rebuildStart;
                
                // 搜索测试
                long searchStart = System.currentTimeMillis();
                List<SearchResult> results = db.search(queryVectorValues, 10);
                long searchTime = System.currentTimeMillis() - searchStart;
                
                // 输出结果
                System.out.println(indexType.name() + " 索引：");
                System.out.println("  - 索引重建时间：" + rebuildTime + " 毫秒");
                System.out.println("  - 搜索时间：" + searchTime + " 毫秒");
                System.out.println("  - 搜索结果数量：" + results.size());
                if (!results.isEmpty()) {
                    System.out.println("  - 最佳匹配：ID=" + results.get(0).getId() + 
                            ", 距离=" + results.get(0).getDistance() + 
                            ", 相似度=" + results.get(0).getSimilarity());
                }
            } catch (Exception e) {
                System.err.println("测试 " + indexType.name() + " 索引时发生错误：" + e.getMessage());
                e.printStackTrace();
            }
            
            long totalTime = System.currentTimeMillis() - startTime;
            System.out.println("  - 总耗时：" + totalTime + " 毫秒\n");
        }
    }
    
    /**
     * 生成随机向量
     * 
     * @param dimension 向量维度
     * @return 随机向量
     */
    private static float[] generateRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = random.nextFloat() * 2 - 1; // 生成-1到1之间的随机数
        }
        return vector;
    }
} 