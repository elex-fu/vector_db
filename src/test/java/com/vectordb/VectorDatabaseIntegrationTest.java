package com.vectordb;

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

/**
 * 向量数据库的集成测试
 */
public class VectorDatabaseIntegrationTest {
    
    private VectorDatabase vectorDb;
    private static final String TEST_DB_PATH = "test_vector_db";
    private static final int DIMENSION = 10;
    private static final int MAX_ELEMENTS = 1000;
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
     * 测试添加和检索向量
     */
    @Test
    public void testAddAndRetrieveVector() throws IOException {
        // 创建测试向量
        float[] values = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            values[i] = i * 0.1f;
        }
        Vector vector = new Vector(1, values);
        
        // 添加向量
        assertTrue(vectorDb.addVector(vector.getId(), vector.getValues()));
        
        // 检索向量
        Vector retrieved = vectorDb.getVector(1).orElse(null);
        
        // 验证向量
        assertNotNull(retrieved);
        assertEquals(vector.getId(), retrieved.getId());
        assertEquals(vector.getDimension(), retrieved.getDimension());
        
        // 验证向量值
        for (int i = 0; i < DIMENSION; i++) {
            assertEquals(vector.getValues()[i], retrieved.getValues()[i], 0.0001f);
        }
    }
    
    /**
     * 测试删除向量
     */
    @Test
    public void testDeleteVector() throws IOException {
        // 创建测试向量
        float[] values = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            values[i] = i * 0.1f;
        }
        Vector vector = new Vector(1, values);
        
        // 添加向量
        assertTrue(vectorDb.addVector(vector.getId(), vector.getValues()));
        
        // 验证向量存在
        assertTrue(vectorDb.getVector(1).isPresent());
        
        // 删除向量
        assertTrue(vectorDb.deleteVector(1));
        
        // 验证向量不存在
        assertFalse(vectorDb.getVector(1).isPresent());
        
        // 删除不存在的向量应该返回false
        assertFalse(vectorDb.deleteVector(2));
    }
    
    /**
     * 测试相似性搜索
     */
    @Test
    public void testSimilaritySearch() throws IOException {
        // 添加多个测试向量
        int numVectors = 100;
        for (int id = 1; id <= numVectors; id++) {
            Vector vector = new Vector(id, generateRandomVector(DIMENSION));
            vectorDb.addVector(vector.getId(), vector.getValues());
        }
        
        // 创建一个特殊的目标向量
        float[] targetValues = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            targetValues[i] = 1.0f; // 全1向量
        }
        Vector target = new Vector(numVectors + 1, targetValues);
        vectorDb.addVector(target.getId(), target.getValues());
        
        // 创建一个接近目标向量的查询向量
        float[] queryValues = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            queryValues[i] = 0.9f; // 接近全1向量
        }
        Vector queryVector = new Vector(-1, queryValues);
        
        // 搜索最相似的向量
        int k = 5;
        List<SearchResult> results = vectorDb.search(queryVector.getValues(), k);
        
        // 验证结果数量
        assertEquals(k, results.size());
        
        // 验证目标向量在结果中
        boolean foundTarget = false;
        for (SearchResult result : results) {
            if (result.getId() == numVectors + 1) {
                foundTarget = true;
                break;
            }
        }
        assertTrue("目标向量应该在搜索结果中", foundTarget);
        
        // 验证结果是按距离升序排序的
        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i).getDistance() >= results.get(i-1).getDistance());
        }
    }
    
    /**
     * 测试数据库持久化和重新加载
     */
    @Test
    public void testPersistenceAndReload() throws IOException {
        // 添加多个测试向量
        int numVectors = 10;
        for (int id = 1; id <= numVectors; id++) {
            Vector vector = new Vector(id, generateRandomVector(DIMENSION));
            vectorDb.addVector(vector.getId(), vector.getValues());
        }
        
        // 关闭数据库
        vectorDb.close();
        
        // 重新打开数据库
        VectorStorage storage = new VectorStorage(TEST_DB_PATH, DIMENSION, MAX_ELEMENTS);
        HnswIndex index = new HnswIndex(DIMENSION, MAX_ELEMENTS);
        vectorDb = new VectorDatabase(DIMENSION, MAX_ELEMENTS, TEST_DB_PATH, storage, index);
        
        // 加载所有向量
        vectorDb.loadFromStorage();
        
        // 验证所有向量都被加载
        for (int id = 1; id <= numVectors; id++) {
            Vector vector = vectorDb.getVector(id).orElse(null);
            assertNotNull("向量 " + id + " 应该存在", vector);
            assertEquals(id, vector.getId());
            assertEquals(DIMENSION, vector.getDimension());
        }
        
        // 验证搜索功能仍然有效
        Vector queryVector = new Vector(-1, generateRandomVector(DIMENSION));
        List<SearchResult> results = vectorDb.search(queryVector.getValues(), 5);
        assertEquals(5, results.size());
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