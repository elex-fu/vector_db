package com.vectordb.index;

import com.vectordb.core.SearchResult;
import com.vectordb.core.Vector;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import lombok.extern.slf4j.Slf4j;

/**
 * 局部敏感哈希（LSH）索引实现
 * 用于高效的近似最近邻搜索
 * 
 * LSH算法特点：
 * 1. 使用多个哈希函数将向量映射到哈希桶中
 * 2. 相似的向量有较高概率被映射到相同的桶中
 * 3. 查询时只需在相同桶中搜索，减少比较次数
 */
@Slf4j
public class LshIndex implements VectorIndex {
    // 索引参数
    private final int dimension;
    private final int maxElements;
    private final int numHashFunctions; // 哈希函数数量
    private final int numHashTables; // 哈希表数量
    private final int bucketWidth; // 桶宽度，影响相似度阈值
    
    // 数据结构
    private final Map<Integer, Vector> vectors; // 存储向量
    private final List<List<float[]>> hashFunctions; // 随机投影向量列表
    private final List<Map<Integer, Set<Integer>>> hashTables; // 哈希表列表
    
    /**
     * 创建LSH索引
     * 
     * @param dimension 向量维度
     * @param maxElements 最大元素数量
     */
    public LshIndex(int dimension, int maxElements) {
        this(dimension, maxElements, 10, 5, 4);
    }
    
    /**
     * 创建LSH索引（带参数）
     * 
     * @param dimension 向量维度
     * @param maxElements 最大元素数量
     * @param numHashFunctions 哈希函数数量
     * @param numHashTables 哈希表数量
     * @param bucketWidth 桶宽度
     */
    public LshIndex(int dimension, int maxElements, int numHashFunctions, int numHashTables, int bucketWidth) {
        this.dimension = dimension;
        this.maxElements = maxElements;
        this.numHashFunctions = numHashFunctions;
        this.numHashTables = numHashTables;
        this.bucketWidth = bucketWidth;
        
        this.vectors = new ConcurrentHashMap<>(maxElements);
        this.hashFunctions = new ArrayList<>(numHashTables);
        this.hashTables = new ArrayList<>(numHashTables);
        
        // 初始化哈希函数和哈希表
        initializeHashFunctions();
    }
    
    /**
     * 初始化哈希函数和哈希表
     */
    private void initializeHashFunctions() {
        Random random = ThreadLocalRandom.current();
        
        // 为每个哈希表创建一组哈希函数
        for (int i = 0; i < numHashTables; i++) {
            List<float[]> tableFunctions = new ArrayList<>(numHashFunctions);
            
            // 为每个哈希函数创建一个随机投影向量
            for (int j = 0; j < numHashFunctions; j++) {
                float[] randomVector = new float[dimension];
                
                // 生成随机单位向量
                for (int k = 0; k < dimension; k++) {
                    randomVector[k] = random.nextFloat() * 2 - 1; // 生成-1到1之间的随机数
                }
                
                // 归一化向量
                float norm = 0;
                for (int k = 0; k < dimension; k++) {
                    norm += randomVector[k] * randomVector[k];
                }
                norm = (float) Math.sqrt(norm);
                
                if (norm > 0) {
                    for (int k = 0; k < dimension; k++) {
                        randomVector[k] /= norm;
                    }
                }
                
                tableFunctions.add(randomVector);
            }
            
            hashFunctions.add(tableFunctions);
            hashTables.add(new HashMap<>());
        }
    }
    
    /**
     * 计算向量的哈希值
     * 
     * @param vector 向量
     * @param tableIndex 哈希表索引
     * @return 哈希值
     */
    private int computeHash(Vector vector, int tableIndex) {
        List<float[]> tableFunctions = hashFunctions.get(tableIndex);
        int[] hashValues = new int[numHashFunctions];
        
        // 计算每个哈希函数的哈希值
        for (int i = 0; i < numHashFunctions; i++) {
            float[] randomVector = tableFunctions.get(i);
            float dotProduct = 0;
            
            // 计算向量与随机向量的点积
            for (int j = 0; j < dimension; j++) {
                dotProduct += vector.getValues()[j] * randomVector[j];
            }
            
            // 将点积映射到整数桶
            hashValues[i] = (int) Math.floor(dotProduct / bucketWidth);
        }
        
        // 组合多个哈希值
        return Arrays.hashCode(hashValues);
    }
    
    /**
     * 添加向量到索引
     */
    @Override
    public synchronized boolean addVector(Vector vector) {
        try {
            int id = vector.getId();
            
            // 检查是否已存在
            if (vectors.containsKey(id)) {
                return false;
            }
            
            // 检查是否超过最大元素数
            if (vectors.size() >= maxElements) {
                return false;
            }
            
            // 存储向量
            vectors.put(id, vector);
            
            // 将向量添加到每个哈希表
            for (int i = 0; i < numHashTables; i++) {
                int hashValue = computeHash(vector, i);
                Map<Integer, Set<Integer>> hashTable = hashTables.get(i);
                
                // 获取或创建桶
                Set<Integer> bucket = hashTable.computeIfAbsent(hashValue, k -> new HashSet<>());
                
                // 将向量ID添加到桶中
                bucket.add(id);
            }
            
            return true;
        } catch (Exception e) {
            log.error("添加向量时发生异常: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 从索引中移除向量
     */
    @Override
    public synchronized boolean removeVector(int id) {
        try {
            // 检查是否存在
            Vector vector = vectors.get(id);
            if (vector == null) {
                return false;
            }
            
            // 从每个哈希表中移除向量
            for (int i = 0; i < numHashTables; i++) {
                int hashValue = computeHash(vector, i);
                Map<Integer, Set<Integer>> hashTable = hashTables.get(i);
                
                // 获取桶
                Set<Integer> bucket = hashTable.get(hashValue);
                if (bucket != null) {
                    // 从桶中移除向量ID
                    bucket.remove(id);
                    
                    // 如果桶为空，移除桶
                    if (bucket.isEmpty()) {
                        hashTable.remove(hashValue);
                    }
                }
            }
            
            // 移除向量
            vectors.remove(id);
            
            return true;
        } catch (Exception e) {
            log.error("移除向量时发生异常: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 搜索最近邻向量
     * 
     * @param queryVector 查询向量
     * @param k 返回结果数量
     * @return 最近邻向量列表
     */
    @Override
    public List<SearchResult> searchNearest(Vector queryVector, int k) {
        try {
            if (vectors.isEmpty()) {
                return new ArrayList<>(); // 如果索引为空，返回空列表
            }
            
            // 如果向量数量小于k，直接计算所有距离
            if (vectors.size() <= k) {
                List<SearchResult> results = new ArrayList<>();
                for (Vector v : vectors.values()) {
                    float distance = queryVector.euclideanDistance(v);
                    results.add(new SearchResult(v.getId(), distance));
                }
                
                // 按距离排序
                results.sort(Comparator.comparing(SearchResult::getDistance));
                return results;
            }
            
            // 使用集合存储候选向量ID，避免重复
            Set<Integer> candidateIds = new HashSet<>();
            
            // 在每个哈希表中搜索
            for (int i = 0; i < numHashTables; i++) {
                int hashValue = computeHash(queryVector, i);
                Map<Integer, Set<Integer>> hashTable = hashTables.get(i);
                
                // 获取桶
                Set<Integer> bucket = hashTable.get(hashValue);
                if (bucket != null) {
                    // 将桶中的向量ID添加到候选集
                    candidateIds.addAll(bucket);
                }
            }
            
            // 如果候选集为空，返回随机k个向量
            if (candidateIds.isEmpty()) {
                List<Integer> allIds = new ArrayList<>(vectors.keySet());
                Collections.shuffle(allIds);
                candidateIds.addAll(allIds.subList(0, Math.min(k * 10, allIds.size())));
            }
            
            // 计算候选向量的距离
            List<SearchResult> results = new ArrayList<>();
            for (int id : candidateIds) {
                Vector v = vectors.get(id);
                if (v != null) {
                    float distance = queryVector.euclideanDistance(v);
                    results.add(new SearchResult(id, distance));
                }
            }
            
            // 按距离排序
            results.sort(Comparator.comparing(SearchResult::getDistance));
            
            // 返回前k个结果
            return results.size() <= k ? results : results.subList(0, k);
        } catch (Exception e) {
            log.error("搜索向量时发生异常: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }
    
    /**
     * 获取索引中的向量数量
     */
    @Override
    public int size() {
        return vectors.size();
    }
    
    /**
     * 重建索引
     * 用于在批量添加或删除向量后，重新构建索引结构以提高搜索效率
     * 
     * @return 是否重建成功
     */
    @Override
    public synchronized boolean buildIndex() {
        try {
            long startTime = System.currentTimeMillis();
            
            // 保存所有向量
            Map<Integer, Vector> savedVectors = new HashMap<>(vectors);
            
            // 清空当前索引结构
            vectors.clear();
            
            // 清空哈希表
            for (Map<Integer, Set<Integer>> hashTable : hashTables) {
                hashTable.clear();
            }
            
            // 重新初始化哈希函数
            hashFunctions.clear();
            initializeHashFunctions();
            
            // 重新添加所有向量
            int successCount = 0;
            for (Vector vector : savedVectors.values()) {
                if (addVector(vector)) {
                    successCount++;
                }
            }
            
            long endTime = System.currentTimeMillis();
            log.info("LSH索引重建完成，包含 {} 个向量，耗时: {} 毫秒", 
                    successCount, (endTime - startTime));
            
            return true;
        } catch (Exception e) {
            log.error("重建索引时发生异常: {}", e.getMessage(), e);
            return false;
        }
    }
} 