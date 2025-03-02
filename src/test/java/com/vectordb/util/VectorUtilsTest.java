package com.vectordb.util;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * VectorUtils类的单元测试
 */
public class VectorUtilsTest {

    /**
     * 测试欧几里得距离计算
     */
    @Test
    public void testEuclideanDistance() {
        float[] v1 = {1.0f, 0.0f, 0.0f};
        float[] v2 = {0.0f, 1.0f, 0.0f};
        
        // 两个单位向量之间的欧几里得距离应为√2
        float expectedDistance = (float) Math.sqrt(2);
        assertEquals(expectedDistance, VectorUtils.euclideanDistance(v1, v2), 0.0001f);
        
        // 距离应该是对称的
        assertEquals(VectorUtils.euclideanDistance(v1, v2), VectorUtils.euclideanDistance(v2, v1), 0.0001f);
        
        // 与自身的距离应为0
        assertEquals(0.0f, VectorUtils.euclideanDistance(v1, v1), 0.0001f);
    }
    
    /**
     * 测试余弦相似度计算
     */
    @Test
    public void testCosineSimilarity() {
        float[] v1 = {1.0f, 0.0f, 0.0f};
        float[] v2 = {0.0f, 1.0f, 0.0f};
        float[] v3 = {1.0f, 1.0f, 0.0f};
        
        // 正交向量的余弦相似度应为0
        assertEquals(0.0f, VectorUtils.cosineSimilarity(v1, v2), 0.0001f);
        
        // 相似度应该是对称的
        assertEquals(VectorUtils.cosineSimilarity(v1, v3), VectorUtils.cosineSimilarity(v3, v1), 0.0001f);
        
        // 与自身的相似度应为1
        assertEquals(1.0f, VectorUtils.cosineSimilarity(v1, v1), 0.0001f);
        
        // v1和v3的相似度应为1/√2
        float expectedSimilarity = 1.0f / (float) Math.sqrt(2);
        assertEquals(expectedSimilarity, VectorUtils.cosineSimilarity(v1, v3), 0.0001f);
    }
    
    /**
     * 测试向量范数计算
     */
    @Test
    public void testNorm() {
        float[] v1 = {3.0f, 4.0f};
        float[] v2 = {1.0f, 1.0f, 1.0f};
        
        // 向量[3,4]的范数应为5
        assertEquals(5.0f, VectorUtils.norm(v1), 0.0001f);
        
        // 向量[1,1,1]的范数应为√3
        float expectedNorm = (float) Math.sqrt(3);
        assertEquals(expectedNorm, VectorUtils.norm(v2), 0.0001f);
    }
    
    /**
     * 测试向量归一化
     */
    @Test
    public void testNormalize() {
        float[] v1 = {3.0f, 4.0f};
        float[] normalized = VectorUtils.normalize(v1);
        
        // 归一化后的向量应该是[3/5, 4/5]
        float[] expectedValues = {3.0f/5.0f, 4.0f/5.0f};
        assertArrayEquals(expectedValues, normalized, 0.0001f);
        
        // 归一化向量的范数应为1
        assertEquals(1.0f, VectorUtils.norm(normalized), 0.0001f);
        
        // 原向量不应被修改
        assertEquals(3.0f, v1[0], 0.0001f);
        assertEquals(4.0f, v1[1], 0.0001f);
    }
    
    /**
     * 测试向量量化和反量化
     */
    @Test
    public void testQuantizationAndDequantization() {
        float[] original = {-1.0f, 0.0f, 0.5f, 1.0f};
        float minValue = -1.0f;
        float maxValue = 1.0f;
        
        // 量化
        byte[] quantized = VectorUtils.quantize(original, minValue, maxValue);
        
        // 检查量化结果
        assertEquals(0, quantized[0] & 0xFF); // -1.0应该量化为0
        assertEquals(127, quantized[1] & 0xFF, 1); // 0.0应该量化为127左右
        assertEquals(191, quantized[2] & 0xFF, 1); // 0.5应该量化为191左右
        assertEquals(255, quantized[3] & 0xFF); // 1.0应该量化为255
        
        // 反量化
        float[] dequantized = VectorUtils.dequantize(quantized, minValue, maxValue);
        
        // 检查反量化结果（会有一些精度损失）
        assertArrayEquals(original, dequantized, 0.01f);
    }
    
    /**
     * 测试维度不匹配的异常
     */
    @Test(expected = IllegalArgumentException.class)
    public void testDimensionMismatchDistance() {
        float[] v1 = {1.0f, 2.0f};
        float[] v2 = {1.0f, 2.0f, 3.0f};
        
        // 这应该抛出异常
        VectorUtils.euclideanDistance(v1, v2);
    }
    
    /**
     * 测试维度不匹配的异常（余弦相似度）
     */
    @Test(expected = IllegalArgumentException.class)
    public void testDimensionMismatchSimilarity() {
        float[] v1 = {1.0f, 2.0f};
        float[] v2 = {1.0f, 2.0f, 3.0f};
        
        // 这应该抛出异常
        VectorUtils.cosineSimilarity(v1, v2);
    }
} 