package com.vectordb.core;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Vector类的单元测试
 */
public class VectorTest {

    /**
     * 测试向量的基本属性
     */
    @Test
    public void testVectorBasicProperties() {
        int id = 1;
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        Vector vector = new Vector(id, values);
        
        // 测试ID
        assertEquals(id, vector.getId());
        
        // 测试维度
        assertEquals(values.length, vector.getDimension());
        
        // 测试值的副本
        float[] retrievedValues = vector.getValues();
        assertArrayEquals(values, retrievedValues, 0.0001f);
        
        // 确保返回的是副本而不是引用
        retrievedValues[0] = 100.0f;
        assertNotEquals(retrievedValues[0], vector.getValues()[0], 0.0001f);
    }
    
    /**
     * 测试欧几里得距离计算
     */
    @Test
    public void testEuclideanDistance() {
        Vector v1 = new Vector(1, new float[]{1.0f, 0.0f, 0.0f});
        Vector v2 = new Vector(2, new float[]{0.0f, 1.0f, 0.0f});
        
        // 两个单位向量之间的欧几里得距离应为√2
        float expectedDistance = (float) Math.sqrt(2);
        assertEquals(expectedDistance, v1.euclideanDistance(v2), 0.0001f);
        
        // 距离应该是对称的
        assertEquals(v1.euclideanDistance(v2), v2.euclideanDistance(v1), 0.0001f);
        
        // 与自身的距离应为0
        assertEquals(0.0f, v1.euclideanDistance(v1), 0.0001f);
    }
    
    /**
     * 测试余弦相似度计算
     */
    @Test
    public void testCosineSimilarity() {
        Vector v1 = new Vector(1, new float[]{1.0f, 0.0f, 0.0f});
        Vector v2 = new Vector(2, new float[]{0.0f, 1.0f, 0.0f});
        Vector v3 = new Vector(3, new float[]{1.0f, 1.0f, 0.0f});
        
        // 正交向量的余弦相似度应为0
        assertEquals(0.0f, v1.cosineSimilarity(v2), 0.0001f);
        
        // 相似度应该是对称的
        assertEquals(v1.cosineSimilarity(v3), v3.cosineSimilarity(v1), 0.0001f);
        
        // 与自身的相似度应为1
        assertEquals(1.0f, v1.cosineSimilarity(v1), 0.0001f);
        
        // v1和v3的相似度应为1/√2
        float expectedSimilarity = 1.0f / (float) Math.sqrt(2);
        assertEquals(expectedSimilarity, v1.cosineSimilarity(v3), 0.0001f);
    }
    
    /**
     * 测试向量范数计算
     */
    @Test
    public void testNorm() {
        Vector v1 = new Vector(1, new float[]{3.0f, 4.0f});
        Vector v2 = new Vector(2, new float[]{1.0f, 1.0f, 1.0f});
        
        // 向量[3,4]的范数应为5
        assertEquals(5.0f, v1.getNorm(), 0.0001f);
        
        // 向量[1,1,1]的范数应为√3
        float expectedNorm = (float) Math.sqrt(3);
        assertEquals(expectedNorm, v2.getNorm(), 0.0001f);
    }
    
    /**
     * 测试向量归一化
     */
    @Test
    public void testNormalize() {
        Vector v1 = new Vector(1, new float[]{3.0f, 4.0f});
        Vector normalized = v1.normalize();
        
        // 归一化后的向量应该是[3/5, 4/5]
        float[] expectedValues = {3.0f/5.0f, 4.0f/5.0f};
        assertArrayEquals(expectedValues, normalized.getValues(), 0.0001f);
        
        // 归一化向量的范数应为1
        assertEquals(1.0f, normalized.getNorm(), 0.0001f);
        
        // 原向量不应被修改
        assertEquals(5.0f, v1.getNorm(), 0.0001f);
    }
    
    /**
     * 测试维度不匹配的异常
     */
    @Test(expected = IllegalArgumentException.class)
    public void testDimensionMismatchDistance() {
        Vector v1 = new Vector(1, new float[]{1.0f, 2.0f});
        Vector v2 = new Vector(2, new float[]{1.0f, 2.0f, 3.0f});
        
        // 这应该抛出异常
        v1.euclideanDistance(v2);
    }
    
    /**
     * 测试维度不匹配的异常（余弦相似度）
     */
    @Test(expected = IllegalArgumentException.class)
    public void testDimensionMismatchSimilarity() {
        Vector v1 = new Vector(1, new float[]{1.0f, 2.0f});
        Vector v2 = new Vector(2, new float[]{1.0f, 2.0f, 3.0f});
        
        // 这应该抛出异常
        v1.cosineSimilarity(v2);
    }
} 