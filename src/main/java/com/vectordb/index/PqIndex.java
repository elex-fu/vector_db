package com.vectordb.index;

import com.vectordb.config.CompressionConfig;
import com.vectordb.core.SearchResult;
import com.vectordb.core.Vector;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import lombok.extern.slf4j.Slf4j;

/**
 * 乘积量化（PQ）索引实现
 * 用于高效的近似最近邻搜索和向量压缩
 * 
 * PQ算法特点：
 * 1. 将高维向量分解为多个低维子向量
 * 2. 对每个子空间进行聚类量化
 * 3. 使用量化后的编码代替原始向量，减少存储空间
 * 4. 使用预计算的距离表加速搜索
 */
@Slf4j
public class PqIndex implements VectorIndex {
    // 索引参数
    private final int dimension;
    private final int maxElements;
    private final int numSubvectors; // 子向量数量
    private final int numClusters; // 每个子空间的聚类数量
    
    // 数据结构
    private final Map<Integer, Vector> vectors; // 存储原始向量
    private final Map<Integer, byte[]> codes; // 存储量化后的编码
    private final float[][][] centroids; // 子空间聚类中心 [子空间索引][聚类索引][维度]
    private final int subvectorDim; // 每个子向量的维度
    
    /**
     * 创建PQ索引
     * 
     * @param dimension 向量维度
     * @param maxElements 最大元素数量
     */
    public PqIndex(int dimension, int maxElements) {
        this(dimension, maxElements, 8, 256);
    }
    
    /**
     * 创建PQ索引（带参数）
     *
     * @param dimension 向量维度
     * @param maxElements 最大元素数量
     * @param numSubvectors 子向量数量
     * @param numClusters 每个子空间的聚类数量（通常为256，使用一个字节表示）
     */
    public PqIndex(int dimension, int maxElements, int numSubvectors, int numClusters) {
        this(dimension, maxElements, createCompressionConfig(dimension, numSubvectors, numClusters));
    }

    /**
     * 创建PQ索引（带压缩配置）
     *
     * @param dimension 向量维度
     * @param maxElements 最大元素数量
     * @param compressionConfig 压缩配置
     */
    public PqIndex(int dimension, int maxElements, CompressionConfig compressionConfig) {
        this.dimension = dimension;
        this.maxElements = maxElements;

        // 确保dimension能被pqSubspaces整除
        int pqSubspaces = compressionConfig.getPqSubspaces();
        if (dimension % pqSubspaces != 0) {
            pqSubspaces = findBestSubspaceDivisor(dimension, pqSubspaces);
            log.warn("维度{}不能被PQ子空间数{}整除，自动调整为{}",
                    dimension, compressionConfig.getPqSubspaces(), pqSubspaces);
        }

        this.numSubvectors = pqSubspaces;
        this.numClusters = Math.min(1 << compressionConfig.getPqBits(), 256);

        // 计算每个子向量的维度
        this.subvectorDim = dimension / this.numSubvectors;

        this.vectors = new ConcurrentHashMap<>(maxElements);
        this.codes = new ConcurrentHashMap<>(maxElements);
        this.centroids = new float[this.numSubvectors][this.numClusters][subvectorDim];

        log.info("创建PQ索引: 维度={}, 子空间数={}, 聚类数={}, 压缩比={:.2f}x",
                dimension, numSubvectors, numClusters,
                compressionConfig.getCompressionRatio(dimension));
    }

    /**
     * 从参数创建压缩配置
     */
    private static CompressionConfig createCompressionConfig(int dimension, int numSubvectors, int numClusters) {
        int pqBits = 8;
        while ((1 << pqBits) < numClusters && pqBits < 8) {
            pqBits++;
        }
        return CompressionConfig.builder()
                .enabled(true)
                .type(CompressionConfig.CompressionType.PQ)
                .pqSubspaces(numSubvectors)
                .pqBits(pqBits)
                .pqIterations(25)
                .build();
    }

    /**
     * 查找最佳的子空间除数
     */
    private int findBestSubspaceDivisor(int dimension, int preferredSubspaces) {
        // 首先尝试preferredSubspaces附近能整除的值
        for (int offset = 0; offset < preferredSubspaces; offset++) {
            if (preferredSubspaces - offset > 0 && dimension % (preferredSubspaces - offset) == 0) {
                return preferredSubspaces - offset;
            }
            if (dimension % (preferredSubspaces + offset) == 0) {
                return preferredSubspaces + offset;
            }
        }
        //  fallback到最大公约数
        for (int subspaces = Math.min(dimension, 128); subspaces >= 8; subspaces--) {
            if (dimension % subspaces == 0) {
                return subspaces;
            }
        }
        return dimension;
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
            
            // 存储原始向量
            vectors.put(id, vector);
            
            // 如果聚类中心已初始化，计算量化编码
            if (isCentroidsInitialized()) {
                byte[] code = computeCode(vector);
                codes.put(id, code);
            }
            
            return true;
        } catch (Exception e) {
            log.error("添加向量时发生异常: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 检查聚类中心是否已初始化
     */
    private boolean isCentroidsInitialized() {
        // 检查第一个子空间的第一个聚类中心是否已初始化
        float[] centroid = centroids[0][0];
        for (float value : centroid) {
            if (value != 0) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * 计算向量的PQ编码
     * 
     * @param vector 向量
     * @return PQ编码
     */
    private byte[] computeCode(Vector vector) {
        byte[] code = new byte[numSubvectors];
        float[] values = vector.getValues();
        
        // 对每个子空间计算最近的聚类中心
        for (int i = 0; i < numSubvectors; i++) {
            int startDim = i * subvectorDim;
            int endDim = Math.min((i + 1) * subvectorDim, dimension);
            
            // 找到最近的聚类中心
            int nearestCluster = 0;
            float minDistance = Float.MAX_VALUE;
            
            for (int j = 0; j < numClusters; j++) {
                float distance = 0;
                
                // 计算子向量到聚类中心的距离
                for (int d = startDim, sd = 0; d < endDim; d++, sd++) {
                    float diff = values[d] - centroids[i][j][sd];
                    distance += diff * diff;
                }
                
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestCluster = j;
                }
            }
            
            // 存储聚类索引
            code[i] = (byte) nearestCluster;
        }
        
        return code;
    }
    
    /**
     * 从索引中移除向量
     */
    @Override
    public synchronized boolean removeVector(int id) {
        try {
            // 检查是否存在
            if (!vectors.containsKey(id)) {
                return false;
            }
            
            // 移除向量和编码
            vectors.remove(id);
            codes.remove(id);
            
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
            
            // 如果向量数量小于k或聚类中心未初始化，直接计算所有距离
            if (vectors.size() <= k || !isCentroidsInitialized()) {
                List<SearchResult> results = new ArrayList<>();
                for (Vector v : vectors.values()) {
                    float distance = queryVector.euclideanDistance(v);
                    results.add(new SearchResult(v.getId(), distance));
                }
                
                // 按距离排序
                results.sort(Comparator.comparing(SearchResult::getDistance));
                return results.size() <= k ? results : results.subList(0, k);
            }
            
            // 计算查询向量到每个聚类中心的距离表
            float[][] distanceTables = computeDistanceTables(queryVector);
            
            // 使用距离表计算近似距离
            List<SearchResult> candidates = new ArrayList<>();
            for (Map.Entry<Integer, byte[]> entry : codes.entrySet()) {
                int id = entry.getKey();
                byte[] code = entry.getValue();
                
                // 计算近似距离
                float distance = 0;
                for (int i = 0; i < numSubvectors; i++) {
                    // 将byte转换为无符号整数（0-255）
                    int clusterIndex = code[i] & 0xFF;
                    distance += distanceTables[i][clusterIndex];
                }
                
                candidates.add(new SearchResult(id, distance));
            }
            
            // 按距离排序
            candidates.sort(Comparator.comparing(SearchResult::getDistance));
            
            // 返回前k个结果
            return candidates.size() <= k ? candidates : candidates.subList(0, k);
        } catch (Exception e) {
            log.error("搜索向量时发生异常: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }
    
    /**
     * 计算查询向量到每个聚类中心的距离表
     * 
     * @param queryVector 查询向量
     * @return 距离表 [子空间索引][聚类索引]
     */
    private float[][] computeDistanceTables(Vector queryVector) {
        float[][] distanceTables = new float[numSubvectors][numClusters];
        float[] values = queryVector.getValues();
        
        // 对每个子空间计算距离表
        for (int i = 0; i < numSubvectors; i++) {
            int startDim = i * subvectorDim;
            int endDim = Math.min((i + 1) * subvectorDim, dimension);
            
            // 计算到每个聚类中心的距离
            for (int j = 0; j < numClusters; j++) {
                float distance = 0;
                
                // 计算子向量到聚类中心的距离
                for (int d = startDim, sd = 0; d < endDim; d++, sd++) {
                    float diff = values[d] - centroids[i][j][sd];
                    distance += diff * diff;
                }
                
                distanceTables[i][j] = distance;
            }
        }
        
        return distanceTables;
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
            
            // 如果向量数量太少，不进行量化
            if (vectors.size() < numClusters) {
                log.info("向量数量太少，不进行量化");
                return true;
            }
            
            // 训练PQ聚类中心
            trainCentroids();
            
            // 计算所有向量的PQ编码
            codes.clear();
            for (Map.Entry<Integer, Vector> entry : vectors.entrySet()) {
                int id = entry.getKey();
                Vector vector = entry.getValue();
                byte[] code = computeCode(vector);
                codes.put(id, code);
            }
            
            long endTime = System.currentTimeMillis();
            log.info("PQ索引重建完成，包含 {} 个向量，{} 个子空间，每个子空间 {} 个聚类，耗时: {} 毫秒", 
                    vectors.size(), numSubvectors, numClusters, (endTime - startTime));
            
            return true;
        } catch (Exception e) {
            log.error("重建索引时发生异常: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 训练PQ聚类中心
     * 使用K-means算法
     */
    private void trainCentroids() {
        // 收集所有向量
        List<Vector> allVectors = new ArrayList<>(vectors.values());
        Random random = ThreadLocalRandom.current();
        
        // 对每个子空间训练聚类中心
        for (int subvectorIndex = 0; subvectorIndex < numSubvectors; subvectorIndex++) {
            int startDim = subvectorIndex * subvectorDim;
            int endDim = Math.min((subvectorIndex + 1) * subvectorDim, dimension);
            int subDim = endDim - startDim;
            
            // 提取子向量
            List<float[]> subvectors = new ArrayList<>(allVectors.size());
            for (Vector vector : allVectors) {
                float[] values = vector.getValues();
                float[] subvector = new float[subDim];
                
                for (int d = startDim, sd = 0; d < endDim; d++, sd++) {
                    subvector[sd] = values[d];
                }
                
                subvectors.add(subvector);
            }
            
            // 初始化聚类中心（随机选择）
            Set<Integer> selectedIndices = new HashSet<>();
            for (int i = 0; i < numClusters && i < subvectors.size(); i++) {
                int index;
                do {
                    index = random.nextInt(subvectors.size());
                } while (selectedIndices.contains(index));
                
                selectedIndices.add(index);
                float[] subvector = subvectors.get(index);
                
                for (int d = 0; d < subDim; d++) {
                    centroids[subvectorIndex][i][d] = subvector[d];
                }
            }
            
            // 如果向量数量少于聚类数量，复制已有的向量
            if (subvectors.size() < numClusters) {
                for (int i = subvectors.size(); i < numClusters; i++) {
                    int sourceIndex = i % subvectors.size();
                    for (int d = 0; d < subDim; d++) {
                        centroids[subvectorIndex][i][d] = centroids[subvectorIndex][sourceIndex][d];
                    }
                }
                continue;
            }
            
            // 执行K-means迭代
            int maxIterations = 10;
            for (int iter = 0; iter < maxIterations; iter++) {
                // 分配向量到最近的聚类
                int[] clusterCounts = new int[numClusters];
                float[][] clusterSums = new float[numClusters][subDim];
                
                for (float[] subvector : subvectors) {
                    // 找到最近的聚类
                    int nearestCluster = 0;
                    float minDistance = Float.MAX_VALUE;
                    
                    for (int j = 0; j < numClusters; j++) {
                        float distance = 0;
                        for (int d = 0; d < subDim; d++) {
                            float diff = subvector[d] - centroids[subvectorIndex][j][d];
                            distance += diff * diff;
                        }
                        
                        if (distance < minDistance) {
                            minDistance = distance;
                            nearestCluster = j;
                        }
                    }
                    
                    // 累加向量到聚类和
                    clusterCounts[nearestCluster]++;
                    for (int d = 0; d < subDim; d++) {
                        clusterSums[nearestCluster][d] += subvector[d];
                    }
                }
                
                // 更新聚类中心
                boolean changed = false;
                for (int j = 0; j < numClusters; j++) {
                    if (clusterCounts[j] > 0) {
                        for (int d = 0; d < subDim; d++) {
                            float newValue = clusterSums[j][d] / clusterCounts[j];
                            if (Math.abs(newValue - centroids[subvectorIndex][j][d]) > 1e-6) {
                                changed = true;
                                centroids[subvectorIndex][j][d] = newValue;
                            }
                        }
                    }
                }
                
                // 如果聚类中心没有变化，提前结束
                if (!changed) {
                    break;
                }
            }
        }
    }
} 