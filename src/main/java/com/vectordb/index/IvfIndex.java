package com.vectordb.index;

import com.vectordb.core.SearchResult;
import com.vectordb.core.Vector;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;

/**
 * 倒排文件索引（IVF）实现
 * 用于高效的近似最近邻搜索
 * 
 * IVF算法特点：
 * 1. 使用聚类将向量空间分割成多个单元
 * 2. 每个向量被分配到最近的聚类中心
 * 3. 查询时只需在最近的几个聚类中搜索
 */
@Slf4j
public class IvfIndex implements VectorIndex {
    // 索引参数
    private final int dimension;
    private final int maxElements;
    private final int numClusters; // 聚类数量
    private final int numProbes; // 查询时探测的聚类数量
    
    // 数据结构
    private final Map<Integer, Vector> vectors; // 存储向量
    private final List<Vector> centroids; // 聚类中心
    private final List<Map<Integer, Float>> clusters; // 聚类->向量ID->距离
    private final Map<Integer, Integer> vectorToCluster; // 向量ID到聚类的映射
    
    /**
     * 创建IVF索引
     * 
     * @param dimension 向量维度
     * @param maxElements 最大元素数量
     */
    public IvfIndex(int dimension, int maxElements) {
        this(dimension, maxElements, 100, 10);
    }
    
    /**
     * 创建IVF索引（带参数）
     * 
     * @param dimension 向量维度
     * @param maxElements 最大元素数量
     * @param numClusters 聚类数量
     * @param numProbes 查询时探测的聚类数量
     */
    public IvfIndex(int dimension, int maxElements, int numClusters, int numProbes) {
        this.dimension = dimension;
        this.maxElements = maxElements;
        this.numClusters = Math.min(numClusters, maxElements / 10); // 确保每个聚类至少有10个元素
        this.numProbes = Math.min(numProbes, this.numClusters);
        
        this.vectors = new ConcurrentHashMap<>(maxElements);
        this.centroids = new ArrayList<>(this.numClusters);
        this.clusters = new ArrayList<>(this.numClusters);
        this.vectorToCluster = new ConcurrentHashMap<>(maxElements);
        
        // 初始化聚类
        for (int i = 0; i < this.numClusters; i++) {
            clusters.add(new HashMap<>());
        }
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
            
            // 如果是前numClusters个向量，直接用作聚类中心
            if (centroids.size() < numClusters && vectors.size() <= numClusters) {
                centroids.add(vector);
                int clusterIndex = centroids.size() - 1;
                clusters.get(clusterIndex).put(id, 0.0f); // 距离为0
                vectorToCluster.put(id, clusterIndex);
                return true;
            }
            
            // 如果聚类中心还未初始化完成，延迟分配
            if (centroids.size() < numClusters) {
                return true;
            }
            
            // 找到最近的聚类中心
            int nearestCluster = findNearestCluster(vector);
            float distance = vector.euclideanDistance(centroids.get(nearestCluster));
            
            // 将向量添加到聚类
            clusters.get(nearestCluster).put(id, distance);
            vectorToCluster.put(id, nearestCluster);
            
            return true;
        } catch (Exception e) {
            log.error("添加向量时发生异常: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 找到最近的聚类中心
     * 
     * @param vector 向量
     * @return 最近聚类的索引
     */
    private int findNearestCluster(Vector vector) {
        int nearestIndex = 0;
        float minDistance = Float.MAX_VALUE;
        
        for (int i = 0; i < centroids.size(); i++) {
            float distance = vector.euclideanDistance(centroids.get(i));
            if (distance < minDistance) {
                minDistance = distance;
                nearestIndex = i;
            }
        }
        
        return nearestIndex;
    }
    
    /**
     * 找到最近的n个聚类中心
     * 
     * @param vector 向量
     * @param n 返回的聚类数量
     * @return 最近聚类的索引列表
     */
    private List<Integer> findNearestClusters(Vector vector, int n) {
        // 计算到每个聚类中心的距离
        List<Map.Entry<Integer, Float>> distances = new ArrayList<>();
        for (int i = 0; i < centroids.size(); i++) {
            float distance = vector.euclideanDistance(centroids.get(i));
            distances.add(new AbstractMap.SimpleEntry<>(i, distance));
        }
        
        // 按距离排序
        distances.sort(Comparator.comparing(Map.Entry::getValue));
        
        // 返回前n个聚类索引
        return distances.stream()
                .limit(n)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
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
            
            // 获取向量所在的聚类
            Integer clusterIndex = vectorToCluster.get(id);
            if (clusterIndex != null) {
                // 从聚类中移除向量
                clusters.get(clusterIndex).remove(id);
            }
            
            // 移除向量和映射
            vectors.remove(id);
            vectorToCluster.remove(id);
            
            // 如果移除的是聚类中心，需要重建索引
            if (centroids.stream().anyMatch(c -> c.getId() == id)) {
                return buildIndex();
            }
            
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
            if (vectors.size() <= k || centroids.isEmpty()) {
                List<SearchResult> results = new ArrayList<>();
                for (Vector v : vectors.values()) {
                    float distance = queryVector.euclideanDistance(v);
                    results.add(new SearchResult(v.getId(), distance));
                }
                
                // 按距离排序
                results.sort(Comparator.comparing(SearchResult::getDistance));
                return results.size() <= k ? results : results.subList(0, k);
            }
            
            // 找到最近的numProbes个聚类
            List<Integer> nearestClusters = findNearestClusters(queryVector, numProbes);
            
            // 收集候选向量
            List<SearchResult> candidates = new ArrayList<>();
            for (int clusterIndex : nearestClusters) {
                Map<Integer, Float> cluster = clusters.get(clusterIndex);
                
                // 计算到每个向量的距离
                for (int vectorId : cluster.keySet()) {
                    Vector v = vectors.get(vectorId);
                    if (v != null) {
                        float distance = queryVector.euclideanDistance(v);
                        candidates.add(new SearchResult(vectorId, distance));
                    }
                }
            }
            
            // 如果候选集为空，返回随机k个向量
            if (candidates.isEmpty()) {
                List<Integer> allIds = new ArrayList<>(vectors.keySet());
                Collections.shuffle(allIds);
                for (int i = 0; i < Math.min(k, allIds.size()); i++) {
                    int id = allIds.get(i);
                    Vector v = vectors.get(id);
                    if (v != null) {
                        float distance = queryVector.euclideanDistance(v);
                        candidates.add(new SearchResult(id, distance));
                    }
                }
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
            centroids.clear();
            for (Map<Integer, Float> cluster : clusters) {
                cluster.clear();
            }
            vectorToCluster.clear();
            
            // 如果向量数量为0，直接返回
            if (savedVectors.isEmpty()) {
                return true;
            }
            
            // 使用K-means++初始化聚类中心
            initializeCentroids(savedVectors.values());
            
            // 重新添加所有向量
            int successCount = 0;
            for (Vector vector : savedVectors.values()) {
                if (addVector(vector)) {
                    successCount++;
                }
            }
            
            // 优化聚类中心
            optimizeCentroids();
            
            long endTime = System.currentTimeMillis();
            log.info("IVF索引重建完成，包含 {} 个向量，{} 个聚类中心，耗时: {} 毫秒", 
                    successCount, numClusters, (endTime - startTime));
            
            return true;
        } catch (Exception e) {
            log.error("重建索引时发生异常: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 使用K-means++初始化聚类中心
     * 
     * @param vectors 向量集合
     */
    private void initializeCentroids(Collection<Vector> vectors) {
        List<Vector> vectorList = new ArrayList<>(vectors);
        Random random = ThreadLocalRandom.current();
        
        // 随机选择第一个聚类中心
        if (!vectorList.isEmpty()) {
            Vector firstCentroid = vectorList.get(random.nextInt(vectorList.size()));
            centroids.add(firstCentroid);
        }
        
        // 使用K-means++选择剩余的聚类中心
        while (centroids.size() < numClusters && centroids.size() < vectorList.size()) {
            // 计算每个向量到最近聚类中心的距离
            double[] distances = new double[vectorList.size()];
            double totalDistance = 0;
            
            for (int i = 0; i < vectorList.size(); i++) {
                Vector vector = vectorList.get(i);
                double minDistance = Double.MAX_VALUE;
                
                // 找到最近的聚类中心
                for (Vector centroid : centroids) {
                    double distance = vector.euclideanDistance(centroid);
                    minDistance = Math.min(minDistance, distance);
                }
                
                distances[i] = minDistance * minDistance; // 平方距离
                totalDistance += distances[i];
            }
            
            // 如果总距离为0，随机选择
            if (totalDistance == 0) {
                while (centroids.size() < numClusters && centroids.size() < vectorList.size()) {
                    int index;
                    do {
                        index = random.nextInt(vectorList.size());
                    } while (centroids.contains(vectorList.get(index)));
                    
                    centroids.add(vectorList.get(index));
                }
                break;
            }
            
            // 按距离的平方加权随机选择下一个聚类中心
            double threshold = random.nextDouble() * totalDistance;
            double cumulativeDistance = 0;
            int selectedIndex = -1;
            
            for (int i = 0; i < distances.length; i++) {
                cumulativeDistance += distances[i];
                if (cumulativeDistance >= threshold) {
                    selectedIndex = i;
                    break;
                }
            }
            
            // 如果没有选中，选择最后一个
            if (selectedIndex == -1) {
                selectedIndex = distances.length - 1;
            }
            
            // 添加新的聚类中心
            centroids.add(vectorList.get(selectedIndex));
        }
    }
    
    /**
     * 优化聚类中心
     * 使用一次K-means迭代
     */
    private void optimizeCentroids() {
        // 如果向量数量太少，不进行优化
        if (vectors.size() < numClusters * 2) {
            return;
        }
        
        // 重新分配向量到最近的聚类
        for (Map.Entry<Integer, Vector> entry : vectors.entrySet()) {
            int id = entry.getKey();
            Vector vector = entry.getValue();
            
            // 找到最近的聚类中心
            int nearestCluster = findNearestCluster(vector);
            int currentCluster = vectorToCluster.getOrDefault(id, -1);
            
            // 如果聚类发生变化，更新分配
            if (nearestCluster != currentCluster) {
                // 从当前聚类中移除
                if (currentCluster >= 0) {
                    clusters.get(currentCluster).remove(id);
                }
                
                // 添加到新聚类
                float distance = vector.euclideanDistance(centroids.get(nearestCluster));
                clusters.get(nearestCluster).put(id, distance);
                vectorToCluster.put(id, nearestCluster);
            }
        }
        
        // 更新聚类中心
        for (int i = 0; i < centroids.size(); i++) {
            Map<Integer, Float> cluster = clusters.get(i);
            
            // 如果聚类为空，保持原样
            if (cluster.isEmpty()) {
                continue;
            }
            
            // 计算聚类中所有向量的平均值
            float[] sum = new float[dimension];
            for (int id : cluster.keySet()) {
                Vector vector = vectors.get(id);
                if (vector != null) {
                    float[] values = vector.getValues();
                    for (int j = 0; j < dimension; j++) {
                        sum[j] += values[j];
                    }
                }
            }
            
            // 计算新的聚类中心
            for (int j = 0; j < dimension; j++) {
                sum[j] /= cluster.size();
            }
            
            // 找到最接近中心的向量作为新的聚类中心
            Vector newCentroid = null;
            float minDistance = Float.MAX_VALUE;
            
            for (int id : cluster.keySet()) {
                Vector vector = vectors.get(id);
                if (vector != null) {
                    float distance = 0;
                    float[] values = vector.getValues();
                    
                    for (int j = 0; j < dimension; j++) {
                        float diff = values[j] - sum[j];
                        distance += diff * diff;
                    }
                    
                    if (distance < minDistance) {
                        minDistance = distance;
                        newCentroid = vector;
                    }
                }
            }
            
            // 更新聚类中心
            if (newCentroid != null) {
                centroids.set(i, newCentroid);
            }
        }
    }
} 