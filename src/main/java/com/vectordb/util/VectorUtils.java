package com.vectordb.util;

import com.vectordb.core.Vector;

/**
 * 向量工具类，提供向量操作的辅助方法
 */
public class VectorUtils {
    
    /**
     * 计算两个向量的欧几里得距离
     */
    public static float euclideanDistance(float[] v1, float[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("向量维度不匹配");
        }
        
        float sum = 0;
        for (int i = 0; i < v1.length; i++) {
            float diff = v1[i] - v2[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }
    
    /**
     * 计算两个向量的余弦相似度
     */
    public static float cosineSimilarity(float[] v1, float[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("向量维度不匹配");
        }
        
        float dotProduct = 0;
        float norm1 = 0;
        float norm2 = 0;
        
        for (int i = 0; i < v1.length; i++) {
            dotProduct += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }
        
        return dotProduct / ((float) Math.sqrt(norm1) * (float) Math.sqrt(norm2));
    }
    
    /**
     * 归一化向量（使其长度为1）
     */
    public static float[] normalize(float[] vector) {
        float[] result = new float[vector.length];
        
        float norm = 0;
        for (float v : vector) {
            norm += v * v;
        }
        norm = (float) Math.sqrt(norm);
        
        for (int i = 0; i < vector.length; i++) {
            result[i] = vector[i] / norm;
        }
        
        return result;
    }
    
    /**
     * 向量量化（将浮点向量转换为字节向量，减少存储空间）
     * 使用简单的线性量化
     */
    public static byte[] quantize(float[] vector, float minValue, float maxValue) {
        byte[] result = new byte[vector.length];
        float range = maxValue - minValue;
        
        for (int i = 0; i < vector.length; i++) {
            // 将浮点值映射到0-255范围
            float normalized = (vector[i] - minValue) / range;
            result[i] = (byte) (Math.min(Math.max(normalized * 255, 0), 255));
        }
        
        return result;
    }
    
    /**
     * 向量反量化（将字节向量转换回浮点向量）
     */
    public static float[] dequantize(byte[] vector, float minValue, float maxValue) {
        float[] result = new float[vector.length];
        float range = maxValue - minValue;
        
        for (int i = 0; i < vector.length; i++) {
            // 将0-255范围映射回浮点值
            float normalized = (vector[i] & 0xFF) / 255.0f;
            result[i] = normalized * range + minValue;
        }
        
        return result;
    }
    
    /**
     * 计算向量的L2范数（欧几里得范数）
     */
    public static float norm(float[] vector) {
        float sum = 0;
        for (float v : vector) {
            sum += v * v;
        }
        return (float) Math.sqrt(sum);
    }
} 