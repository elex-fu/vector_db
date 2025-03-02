package com.vectordb.benchmark;

import com.vectordb.core.SearchResult;
import com.vectordb.core.Vector;
import com.vectordb.core.VectorDatabase;
import com.vectordb.index.HnswIndex;
import com.vectordb.storage.VectorStorage;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.ArrayList;

/**
 * 向量数据库性能基准测试
 * 注意：这些测试可能需要较长时间运行，不适合作为常规单元测试
 */
public class PerformanceBenchmarkTest {
    
    private VectorDatabase vectorDb;
    private static final String TEST_DB_PATH = "benchmark_vector_db";
    private static final int DIMENSION = 10;
    private static final int MAX_ELEMENTS = 100;
    private Random random;
    
    @Before
    public void setUp() throws IOException {
        // 创建测试目录
        File dbDir = new File(TEST_DB_PATH);
        if (!dbDir.exists()) {
            dbDir.mkdirs();
        }
        
        // 初始化向量存储和索引
        VectorStorage storage = new VectorStorage(TEST_DB_PATH, DIMENSION, MAX_ELEMENTS);
        HnswIndex index = new HnswIndex(DIMENSION, MAX_ELEMENTS);
        
        // 初始化向量数据库
        vectorDb = new VectorDatabase(DIMENSION, MAX_ELEMENTS, TEST_DB_PATH, storage, index);
        
        // 初始化随机数生成器
        random = new Random(42); // 使用固定种子以便结果可重现
    }
    
    @After
    public void tearDown() throws IOException {
        // 关闭向量数据库
        if (vectorDb != null) {
            vectorDb.close();
        }
        
        // 清理测试目录
        deleteDirectory(new File(TEST_DB_PATH));
    }
    
    /**
     * 测试批量插入性能
     */
    @Test
    public void testBatchInsertionPerformance() throws IOException {
        // 测试不同批量大小
        int[] batchSizes = {100, 1000, 10000};
        
        for (int batchSize : batchSizes) {
            // 准备向量
            List<Vector> vectors = new ArrayList<>();
            for (int i = 0; i < batchSize; i++) {
                vectors.add(new Vector(i, generateRandomVector(DIMENSION)));
            }
            
            // 测量插入时间
            long startTime = System.currentTimeMillis();
            
            for (Vector vector : vectors) {
                vectorDb.addVector(vector.getId(), vector.getValues());
            }
            
            long endTime = System.currentTimeMillis();
            long duration = endTime - startTime;
            
            // 输出结果
            System.out.printf("批量插入 %d 个向量耗时: %d 毫秒 (平均每个向量 %.2f 毫秒)%n", 
                    batchSize, duration, (double) duration / batchSize);
            
            // 清空数据库以便下一次测试
            vectorDb.close();
            deleteDirectory(new File(TEST_DB_PATH));
            setUp();
        }
    }
    
    /**
     * 测试搜索性能随数据库大小的变化
     */
    @Test
    public void testSearchPerformanceVsDatabaseSize() throws IOException {
        // 测试不同数据库大小
        int[] databaseSizes = {100, 1000, 10000};
        int numQueries = 100;
        int k = 10;
        
        for (int dbSize : databaseSizes) {
            // 准备向量并添加到数据库
            for (int i = 0; i < dbSize; i++) {
                Vector vector = new Vector(i, generateRandomVector(DIMENSION));
                vectorDb.addVector(vector.getId(), vector.getValues());
            }
            
            // 准备查询向量
            List<Vector> queryVectors = new ArrayList<>();
            for (int i = 0; i < numQueries; i++) {
                queryVectors.add(new Vector(-i-1, generateRandomVector(DIMENSION)));
            }
            
            // 测量搜索时间
            long startTime = System.currentTimeMillis();
            
            for (Vector queryVector : queryVectors) {
                List<SearchResult> results = vectorDb.search(queryVector.getValues(), k);
                assertNotNull(results);
                assertTrue(results.size() <= k);
            }
            
            long endTime = System.currentTimeMillis();
            long duration = endTime - startTime;
            
            // 输出结果
            System.out.printf("数据库大小: %d, 执行 %d 次查询耗时: %d 毫秒 (平均每次查询 %.2f 毫秒)%n", 
                    dbSize, numQueries, duration, (double) duration / numQueries);
            
            // 清空数据库以便下一次测试
            vectorDb.close();
            deleteDirectory(new File(TEST_DB_PATH));
            setUp();
        }
    }
    
    /**
     * 测试不同维度的搜索性能
     */
    @Test
    public void testSearchPerformanceVsDimension() throws IOException {
        // 测试不同维度
        int[] dimensions = {10, 100, 500, 1000};
        int dbSize = 1000;
        int numQueries = 100;
        int k = 10;
        
        for (int dim : dimensions) {
            // 重新初始化数据库以适应新维度
            vectorDb.close();
            deleteDirectory(new File(TEST_DB_PATH));
            
            // 创建测试目录
            File dbDir = new File(TEST_DB_PATH);
            if (!dbDir.exists()) {
                dbDir.mkdirs();
            }
            
            // 初始化向量存储和索引
            VectorStorage storage = new VectorStorage(TEST_DB_PATH, dim, MAX_ELEMENTS);
            HnswIndex index = new HnswIndex(dim, MAX_ELEMENTS);
            
            // 初始化向量数据库
            vectorDb = new VectorDatabase(dim, MAX_ELEMENTS, TEST_DB_PATH, storage, index);
            
            // 准备向量并添加到数据库
            for (int i = 0; i < dbSize; i++) {
                Vector vector = new Vector(i, generateRandomVector(dim));
                vectorDb.addVector(vector.getId(), vector.getValues());
            }
            
            // 准备查询向量
            List<Vector> queryVectors = new ArrayList<>();
            for (int i = 0; i < numQueries; i++) {
                queryVectors.add(new Vector(-i-1, generateRandomVector(dim)));
            }
            
            // 测量搜索时间
            long startTime = System.currentTimeMillis();
            
            for (Vector queryVector : queryVectors) {
                List<SearchResult> results = vectorDb.search(queryVector.getValues(), k);
                assertNotNull(results);
                assertTrue(results.size() <= k);
            }
            
            long endTime = System.currentTimeMillis();
            long duration = endTime - startTime;
            
            // 输出结果
            System.out.printf("维度: %d, 执行 %d 次查询耗时: %d 毫秒 (平均每次查询 %.2f 毫秒)%n", 
                    dim, numQueries, duration, (double) duration / numQueries);
        }
    }
    
    /**
     * 测试内存使用情况
     */
    @Test
    public void testMemoryUsage() throws IOException {
        // 测试不同数据库大小
        int[] databaseSizes = {1000, 10000, 50000};
        
        for (int dbSize : databaseSizes) {
            // 记录初始内存使用
            System.gc(); // 尝试触发垃圾回收
            long initialMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            
            // 准备向量并添加到数据库
            for (int i = 0; i < dbSize; i++) {
                Vector vector = new Vector(i, generateRandomVector(DIMENSION));
                vectorDb.addVector(vector.getId(), vector.getValues());
            }
            
            // 记录最终内存使用
            System.gc(); // 尝试触发垃圾回收
            long finalMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            
            // 计算内存使用增量
            long memoryUsed = finalMemory - initialMemory;
            
            // 输出结果
            System.out.printf("数据库大小: %d, 内存使用: %.2f MB (平均每个向量 %.2f KB)%n", 
                    dbSize, memoryUsed / (1024.0 * 1024.0), memoryUsed / (dbSize * 1024.0));
            
            // 清空数据库以便下一次测试
            vectorDb.close();
            deleteDirectory(new File(TEST_DB_PATH));
            setUp();
        }
    }
    
    /**
     * 生成随机向量
     */
    private float[] generateRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = random.nextFloat() * 2 - 1; // 生成-1到1之间的随机值
        }
        return vector;
    }
    
    /**
     * 递归删除目录
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
} 