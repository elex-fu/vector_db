package com.vectordb.index;

import com.vectordb.core.SearchResult;
import com.vectordb.core.Vector;

import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

import java.util.List;
import java.util.Random;

/**
 * HnswIndex类的单元测试
 */
public class HnswIndexTest {
    
    private HnswIndex index;
    private static final int DIMENSION = 10;
    private static final int MAX_ELEMENTS = 1000;
    private Random random;
    
    @Before
    public void setUp() {
        index = new HnswIndex(DIMENSION, MAX_ELEMENTS);
        random = new Random(42); // 使用固定种子以便结果可重现
    }
    
    /**
     * 测试添加向量
     */
    @Test
    public void testAddVector() {
        // 创建测试向量
        Vector v1 = new Vector(1, generateRandomVector(DIMENSION));
        Vector v2 = new Vector(2, generateRandomVector(DIMENSION));
        
        // 添加向量
        assertTrue(index.addVector(v1));
        assertTrue(index.addVector(v2));
        
        // 验证索引大小
        assertEquals(2, index.size());
        
        // 重复添加应该失败
        assertFalse(index.addVector(v1));
        
        // 索引大小应该不变
        assertEquals(2, index.size());
    }
    
    /**
     * 测试移除向量
     */
    @Test
    public void testRemoveVector() {
        // 创建并添加测试向量
        Vector v1 = new Vector(1, generateRandomVector(DIMENSION));
        Vector v2 = new Vector(2, generateRandomVector(DIMENSION));
        index.addVector(v1);
        index.addVector(v2);
        
        // 验证索引大小
        assertEquals(2, index.size());
        
        // 移除向量
        assertTrue(index.removeVector(1));
        
        // 验证索引大小
        assertEquals(1, index.size());
        
        // 移除不存在的向量应该失败
        assertFalse(index.removeVector(3));
        
        // 索引大小应该不变
        assertEquals(1, index.size());
        
        // 移除剩余向量
        assertTrue(index.removeVector(2));
        
        // 验证索引为空
        assertEquals(0, index.size());
    }
    
    /**
     * 测试搜索最近邻
     */
    @Test
    public void testSearchNearest() {
        // 创建一组测试向量
        int numVectors = 100;
        for (int i = 0; i < numVectors; i++) {
            Vector vector = new Vector(i, generateRandomVector(DIMENSION));
            index.addVector(vector);
        }
        
        // 验证索引大小
        assertEquals(numVectors, index.size());
        
        // 创建查询向量
        Vector queryVector = new Vector(-1, generateRandomVector(DIMENSION));
        
        // 搜索最近的10个向量
        int k = 10;
        List<SearchResult> results = index.searchNearest(queryVector, k);
        
        // 验证结果数量
        assertEquals(k, results.size());
        
        // 验证结果是按距离升序排序的
        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i).getDistance() >= results.get(i-1).getDistance());
        }
    }
    
    /**
     * 测试空索引的搜索
     */
    @Test
    public void testSearchEmptyIndex() {
        // 创建查询向量
        Vector queryVector = new Vector(-1, generateRandomVector(DIMENSION));
        
        // 搜索最近的10个向量
        List<SearchResult> results = index.searchNearest(queryVector, 10);
        
        // 结果应该是空列表
        assertTrue(results.isEmpty());
    }
    
    /**
     * 测试搜索精度
     */
    @Test
    public void testSearchAccuracy() {
        // 创建一组测试向量，包括一个已知的最近邻
        float[] targetValues = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            targetValues[i] = 1.0f; // 全1向量
        }
        Vector target = new Vector(0, targetValues);
        index.addVector(target);
        
        // 添加其他随机向量
        int numVectors = 100;
        for (int i = 1; i < numVectors; i++) {
            Vector vector = new Vector(i, generateRandomVector(DIMENSION));
            index.addVector(vector);
        }
        
        // 创建一个接近目标向量的查询向量
        float[] queryValues = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            queryValues[i] = 0.9f; // 接近全1向量
        }
        Vector queryVector = new Vector(-1, queryValues);
        
        // 搜索最近邻
        List<SearchResult> results = index.searchNearest(queryVector, 1);
        
        // 验证结果
        assertEquals(1, results.size());
        assertEquals(0, results.get(0).getId()); // 应该找到目标向量
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
} 