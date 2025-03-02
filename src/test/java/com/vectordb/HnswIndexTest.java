package com.vectordb;

import com.vectordb.core.SearchResult;
import com.vectordb.core.Vector;
import com.vectordb.index.HnswIndex;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import lombok.extern.slf4j.Slf4j;

/**
 * HNSW索引优化测试类
 * 用于测试优化后的HNSW索引在高维向量上的性能
 */
@Slf4j
public class HnswIndexTest {
    
    /**
     * 生成随机向量
     * 
     * @param id 向量ID
     * @param dimension 向量维度
     * @return 随机向量
     */
    private static Vector generateRandomVector(int id, int dimension) {
        Random random = new Random();
        float[] values = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            values[i] = random.nextFloat() * 2 - 1; // 生成-1到1之间的随机值
        }
        return new Vector(id, values);
    }
    
    /**
     * 测试HNSW索引在高维向量上的性能
     * 
     * @param dimension 向量维度
     * @param vectorCount 向量数量
     * @param queryCount 查询次数
     * @param k 每次查询返回的结果数量
     */
    public static void testHnswPerformance(int dimension, int vectorCount, int queryCount, int k) {
        log.info("开始测试HNSW索引在{}维向量上的性能...", dimension);
        
        // 创建索引
        HnswIndex index = new HnswIndex(dimension, vectorCount);
        
        // 优化高维向量参数
        index.optimizeForHighDimension(dimension);
        
        // 生成随机向量并添加到索引
        List<Vector> vectors = new ArrayList<>();
        long startTime = System.currentTimeMillis();
        
        for (int i = 0; i < vectorCount; i++) {
            Vector vector = generateRandomVector(i, dimension);
            vectors.add(vector);
            index.addVector(vector);
            
            if ((i + 1) % 1000 == 0 || i == vectorCount - 1) {
                log.info("已添加 {}/{} 个向量", (i + 1), vectorCount);
            }
        }
        
        long endTime = System.currentTimeMillis();
        log.info("添加 {} 个向量耗时: {} 毫秒", vectorCount, (endTime - startTime));
        
        // 输出索引统计信息
        log.info(index.getIndexStats());
        
        // 测试搜索性能
        log.info("开始测试搜索性能...");
        startTime = System.currentTimeMillis();
        
        int totalResults = 0;
        for (int i = 0; i < queryCount; i++) {
            // 随机选择一个向量作为查询向量
            Vector queryVector = generateRandomVector(-1, dimension);
            
            // 搜索最近邻
            List<SearchResult> results = index.searchNearest(queryVector, k);
            totalResults += results.size();
            
            if ((i + 1) % 100 == 0 || i == queryCount - 1) {
                log.info("已完成 {}/{} 次查询", (i + 1), queryCount);
            }
        }
        
        endTime = System.currentTimeMillis();
        double avgQueryTime = (double)(endTime - startTime) / queryCount;
        
        log.info("查询性能统计:");
        log.info("- 总查询次数: {}", queryCount);
        log.info("- 总查询时间: {} 毫秒", (endTime - startTime));
        log.info("- 平均查询时间: {} 毫秒", avgQueryTime);
        log.info("- 平均返回结果数: {}", (double)totalResults / queryCount);
    }
    
    /**
     * 测试不同参数对HNSW索引准确率的影响
     * 
     * @param dimension 向量维度
     * @param vectorCount 向量数量
     * @param queryCount 查询次数
     * @param k 每次查询返回的结果数量
     */
    public static void testHnswAccuracy(int dimension, int vectorCount, int queryCount, int k) {
        log.info("开始测试不同参数对HNSW索引准确率的影响...");
        
        // 生成随机向量
        List<Vector> vectors = new ArrayList<>();
        for (int i = 0; i < vectorCount; i++) {
            vectors.add(generateRandomVector(i, dimension));
        }
        
        // 生成查询向量
        List<Vector> queryVectors = new ArrayList<>();
        for (int i = 0; i < queryCount; i++) {
            queryVectors.add(generateRandomVector(-i-1, dimension));
        }
        
        // 测试不同参数组合
        int[][] paramSets = {
            {16, 200, 200, 0, 0},  // 默认参数
            {32, 400, 400, 1, 1},  // 优化参数1
            {48, 600, 600, 1, 1}   // 优化参数2
        };
        
        for (int[] params : paramSets) {
            int m = params[0];
            int efConstruction = params[1];
            int ef = params[2];
            boolean useCosineSimilarity = params[3] == 1;
            boolean normalizeVectors = params[4] == 1;
            
            log.info("\n测试参数组合: M={}, efConstruction={}, ef={}, 使用余弦相似度={}, 向量归一化={}", 
                    m, efConstruction, ef, useCosineSimilarity, normalizeVectors);
            
            // 创建索引
            HnswIndex index = new HnswIndex(dimension, vectorCount);
            index.setIndexParameters(m, efConstruction, ef, useCosineSimilarity, normalizeVectors);
            
            // 添加向量
            for (Vector vector : vectors) {
                index.addVector(vector);
            }
            
            // 测试搜索性能
            long startTime = System.currentTimeMillis();
            
            for (Vector queryVector : queryVectors) {
                index.searchNearest(queryVector, k);
            }
            
            long endTime = System.currentTimeMillis();
            double avgQueryTime = (double)(endTime - startTime) / queryCount;
            
            log.info("- 平均查询时间: {} 毫秒", avgQueryTime);
        }
    }
    
    /**
     * 主方法
     */
    public static void main(String[] args) {
        // 测试1536维向量（OpenAI嵌入向量维度）
        int dimension = 1536;
        int vectorCount = 10000;
        int queryCount = 100;
        int k = 10;
        
        // 测试性能
        testHnswPerformance(dimension, vectorCount, queryCount, k);
        
        // 测试不同参数的准确率
        testHnswAccuracy(dimension, 5000, 50, k);
    }
} 