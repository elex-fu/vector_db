package com.vectordb.util;

import java.nio.charset.StandardCharsets;
import java.util.Random;

/**
 * 文字转向量工具类
 * 将文本转换为固定维度的向量表示
 */
public class TextVectorizer {
    
    private static final Random random = new Random(42); // 使用固定种子以保证结果可重现
    
    /**
     * 将文本转换为向量
     * 改进版本：使用字符级别的特征提取，使相似文本生成相似向量
     * 特别优化了汉字处理，增强了相似文本的向量相似度
     * 
     * @param text 输入文本
     * @param dimension 向量维度
     * @return 文本的向量表示
     */
    public static float[] textToVector(String text, int dimension) {
        if (text == null || text.isEmpty()) {
            throw new IllegalArgumentException("文本不能为空");
        }
        
        // 初始化向量
        float[] vector = new float[dimension];
        
        // 获取文本的字符数组
        char[] chars = text.toCharArray();
        
        // 计算文本的整体特征
        long textHash = computeTextHash(text);
        
        // 为每个字符分配维度空间，确保每个字符有足够的维度表示
        int charsPerDimension = Math.max(1, dimension / (chars.length * 2));
        
        // 对每个字符进行处理
        for (int i = 0; i < chars.length; i++) {
            char c = chars[i];
            
            // 使用字符的Unicode值作为特征
            int charValue = (int) c;
            
            // 为每个字符生成一个稳定的伪随机数序列
            Random charRandom = new Random(charValue * 31 + i);
            
            // 计算字符在向量中的基础位置
            // 使用字符的Unicode值和位置信息确定起始位置
            int basePos = (charValue * (i + 1)) % dimension;
            
            // 字符的权重，位置越靠前权重越大
            float weight = 1.0f - (0.1f * i);
            if (weight < 0.3f) weight = 0.3f;
            
            // 将字符的特征分布到向量的多个维度上
            for (int j = 0; j < dimension / chars.length; j++) {
                int pos = (basePos + j) % dimension;
                // 使用字符的Unicode值和位置信息生成向量值
                vector[pos] += weight * (float) charRandom.nextGaussian();
            }
            
            // 考虑字符的上下文信息
            if (chars.length > 1) {
                // 处理相邻字符的关系
                for (int j = 0; j < chars.length; j++) {
                    if (i != j) {
                        // 计算当前字符与其他字符的关系
                        int otherCharValue = (int) chars[j];
                        int relationValue = charValue * 31 + otherCharValue;
                        int relationPos = Math.abs(relationValue) % dimension;
                        
                        // 相邻字符的影响随距离减弱
                        float relationWeight = 0.5f / (1 + Math.abs(i - j));
                        vector[relationPos] += relationWeight;
                    }
                }
            }
        }
        
        // 添加整体文本的特征
        Random textRandom = new Random(textHash);
        
        // 在向量的随机位置添加整体文本特征
        for (int i = 0; i < dimension / 5; i++) {
            int pos = textRandom.nextInt(dimension);
            vector[pos] += (float) textRandom.nextGaussian() * 0.3f;
        }
        
        // 特殊处理：如果文本中只有一个字符被替换，确保向量相似
        if (text.length() > 1) {
            // 为每个可能的单字符替换生成一个特征
            for (int i = 0; i < chars.length; i++) {
                // 计算除了当前字符外的所有字符的哈希值
                StringBuilder sb = new StringBuilder(text);
                sb.deleteCharAt(i);
                long partialHash = computeTextHash(sb.toString());
                
                // 将这个特征添加到向量中
                int partialPos = (int) (partialHash % dimension);
                vector[partialPos] += 0.8f;
            }
        }
        
        // 归一化向量
        normalizeVector(vector);
        
        return vector;
    }
    
    /**
     * 计算文本的哈希值
     */
    private static long computeTextHash(String text) {
        byte[] textBytes = text.getBytes(StandardCharsets.UTF_8);
        return bytesToLong(textBytes);
    }
    
    /**
     * 生成相似的文本向量
     * 
     * @param originalVector 原始向量
     * @param similarityFactor 相似度因子（0-1之间，越大越相似）
     * @param dimension 向量维度
     * @return 相似的向量
     */
    public static float[] generateSimilarVector(float[] originalVector, float similarityFactor, int dimension) {
        if (originalVector == null || originalVector.length != dimension) {
            throw new IllegalArgumentException("原始向量不能为空且维度必须匹配");
        }
        
        if (similarityFactor < 0 || similarityFactor > 1) {
            throw new IllegalArgumentException("相似度因子必须在0-1之间");
        }
        
        float[] similarVector = new float[dimension];
        
        // 生成噪声向量
        float[] noiseVector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            noiseVector[i] = (float) random.nextGaussian();
        }
        normalizeVector(noiseVector);
        
        // 按照相似度因子混合原始向量和噪声向量
        for (int i = 0; i < dimension; i++) {
            similarVector[i] = similarityFactor * originalVector[i] + (1 - similarityFactor) * noiseVector[i];
        }
        
        // 归一化结果向量
        normalizeVector(similarVector);
        
        return similarVector;
    }
    
    /**
     * 计算两个向量之间的余弦相似度
     * 
     * @param vector1 向量1
     * @param vector2 向量2
     * @return 余弦相似度（-1到1之间，越接近1表示越相似）
     */
    public static float cosineSimilarity(float[] vector1, float[] vector2) {
        if (vector1.length != vector2.length) {
            throw new IllegalArgumentException("向量维度不匹配");
        }
        
        float dotProduct = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        
        for (int i = 0; i < vector1.length; i++) {
            dotProduct += vector1[i] * vector2[i];
            norm1 += vector1[i] * vector1[i];
            norm2 += vector2[i] * vector2[i];
        }
        
        // 避免除以零
        if (norm1 == 0 || norm2 == 0) {
            return 0;
        }
        
        return dotProduct / (float) (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
    
    /**
     * 将字节数组转换为长整型，用于随机数种子
     */
    private static long bytesToLong(byte[] bytes) {
        long result = 0;
        for (byte b : bytes) {
            result = result * 31 + (b & 0xFF);
        }
        return result;
    }
    
    /**
     * 归一化向量（使其长度为1）
     */
    private static void normalizeVector(float[] vector) {
        float sum = 0;
        for (float v : vector) {
            sum += v * v;
        }
        
        float norm = (float) Math.sqrt(sum);
        if (norm > 0) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
    }
} 