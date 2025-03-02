package com.vectordb.core;

import java.io.Serializable;
import java.util.Arrays;

/**
 * 表示一个向量及其相关操作
 */
public class Vector implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final int id;
    private final float[] values;
    private transient float norm; // 缓存向量的范数，不序列化

    /**
     * 无参构造函数，用于Jackson反序列化
     */
    public Vector() {
        this.id = -1;
        this.values = new float[0];
        this.norm = -1;
    }

    /**
     * 创建一个新向量
     * 
     * @param id 向量ID
     * @param values 向量值
     */
    public Vector(int id, float[] values) {
        this.id = id;
        this.values = Arrays.copyOf(values, values.length);
        this.norm = -1; // 标记为未计算
    }

    /**
     * 获取向量ID
     */
    public int getId() {
        return id;
    }

    /**
     * 获取向量值的副本
     */
    public float[] getValues() {
        return Arrays.copyOf(values, values.length);
    }

    /**
     * 获取向量维度
     */
    public int getDimension() {
        return values.length;
    }

    /**
     * 计算与另一个向量的欧几里得距离
     */
    public float euclideanDistance(Vector other) {
        if (this.getDimension() != other.getDimension()) {
            throw new IllegalArgumentException("向量维度不匹配");
        }
        
        float sum = 0;
        for (int i = 0; i < values.length; i++) {
            float diff = this.values[i] - other.values[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }

    /**
     * 计算与另一个向量的余弦相似度
     */
    public float cosineSimilarity(Vector other) {
        if (this.getDimension() != other.getDimension()) {
            throw new IllegalArgumentException("向量维度不匹配");
        }
        
        float dotProduct = 0;
        for (int i = 0; i < values.length; i++) {
            dotProduct += this.values[i] * other.values[i];
        }
        
        return dotProduct / (getNorm() * other.getNorm());
    }

    /**
     * 计算向量的L2范数（欧几里得范数）
     */
    public float getNorm() {
        if (norm < 0) {
            float sum = 0;
            for (float value : values) {
                sum += value * value;
            }
            norm = (float) Math.sqrt(sum);
        }
        return norm;
    }

    /**
     * 归一化向量（使其长度为1）
     */
    public Vector normalize() {
        float norm = getNorm();
        float[] normalizedValues = new float[values.length];
        
        for (int i = 0; i < values.length; i++) {
            normalizedValues[i] = values[i] / norm;
        }
        
        return new Vector(id, normalizedValues);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Vector vector = (Vector) o;
        return id == vector.id && Arrays.equals(values, vector.values);
    }

    @Override
    public int hashCode() {
        int result = id;
        result = 31 * result + Arrays.hashCode(values);
        return result;
    }

    @Override
    public String toString() {
        return "Vector{" +
                "id=" + id +
                ", dimension=" + values.length +
                '}';
    }
} 