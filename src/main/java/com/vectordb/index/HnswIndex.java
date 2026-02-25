package com.vectordb.index;

import com.vectordb.core.SearchResult;
import com.vectordb.core.Vector;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;

/**
 * HNSW（Hierarchical Navigable Small World）索引实现
 * 用于高效的近似最近邻搜索
 */
@Slf4j
public class HnswIndex implements VectorIndex {
    // 索引参数
    private final int dimension;
    private final int maxElements;
    private int m = 32; // 每个节点的最大连接数，默认值增加到32
    private int efConstruction = 400; // 构建时的搜索宽度，默认值增加到400
    private int ef = 400; // 搜索时的候选集大小，默认值增加到400
    private int maxLevel; // 最大层数，根据M值动态计算
    private boolean useCosineSimilarity = true; // 是否使用余弦相似度
    private boolean normalizeVectors = true; // 是否归一化向量
    
    // 数据结构
    private final Map<Integer, Vector> vectors; // 存储向量
    private final Map<Integer, Integer> idToLevel; // 向量ID到层级的映射
    private final List<Map<Integer, List<Integer>>> graph; // 多层图结构
    
    // 入口点
    private volatile int entryPoint = -1;
    
    /**
     * 创建HNSW索引
     * 
     * @param dimension 向量维度
     * @param maxElements 最大元素数量
     */
    public HnswIndex(int dimension, int maxElements) {
        this.dimension = dimension;
        this.maxElements = maxElements;
        this.vectors = new ConcurrentHashMap<>(maxElements);
        this.idToLevel = new ConcurrentHashMap<>(maxElements);
        
        // 根据M值计算最大层级
        this.maxLevel = (int)(Math.log(maxElements) / Math.log(m)) + 1;
        
        this.graph = new ArrayList<>();
        
        // 初始化多层图结构
        for (int i = 0; i < maxLevel; i++) {
            graph.add(new ConcurrentHashMap<>());
        }
    }
    
    /**
     * 创建HNSW索引，带自定义参数
     * 
     * @param dimension 向量维度
     * @param maxElements 最大元素数量
     * @param m 每个节点的最大连接数
     * @param efConstruction 构建时的搜索宽度
     * @param ef 搜索时的候选集大小
     */
    public HnswIndex(int dimension, int maxElements, int m, int efConstruction, int ef) {
        this.dimension = dimension;
        this.maxElements = maxElements;
        this.m = m;
        this.efConstruction = efConstruction;
        this.ef = ef;
        this.vectors = new ConcurrentHashMap<>(maxElements);
        this.idToLevel = new ConcurrentHashMap<>(maxElements);
        
        // 根据M值计算最大层级
        this.maxLevel = (int)(Math.log(maxElements) / Math.log(m)) + 1;
        
        this.graph = new ArrayList<>();
        
        // 初始化多层图结构
        for (int i = 0; i < maxLevel; i++) {
            graph.add(new ConcurrentHashMap<>());
        }
    }
    
    /**
     * 设置索引参数
     * 
     * @param m 每个节点的最大连接数
     * @param efConstruction 构建时的搜索宽度
     * @param ef 搜索时的候选集大小
     * @param useCosineSimilarity 是否使用余弦相似度
     * @param normalizeVectors 是否归一化向量
     */
    public void setIndexParameters(int m, int efConstruction, int ef, boolean useCosineSimilarity, boolean normalizeVectors) {
        this.m = m;
        this.efConstruction = efConstruction;
        this.ef = ef;
        this.useCosineSimilarity = useCosineSimilarity;
        this.normalizeVectors = normalizeVectors;
        
        // 重新计算最大层级
        this.maxLevel = (int)(Math.log(maxElements) / Math.log(m)) + 1;
        
        // 如果图结构大小需要调整
        while (graph.size() < maxLevel) {
            graph.add(new ConcurrentHashMap<>());
        }
    }
    
    /**
     * 归一化向量
     */
    private Vector normalizeVector(Vector vector) {
        if (normalizeVectors) {
            return vector.normalize();
        }
        return vector;
    }
    
    /**
     * 计算两个向量之间的距离
     */
    private float calculateDistance(Vector v1, Vector v2) {
        if (useCosineSimilarity) {
            // 余弦相似度转换为距离（1-相似度）
            return 1.0f - v1.cosineSimilarity(v2);
        } else {
            return v1.euclideanDistance(v2);
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
            
            // 归一化向量
            Vector normalizedVector = normalizeVector(vector);
            
            // 存储向量
            vectors.put(id, normalizedVector);
            
            // 随机分配层级（按照HNSW论文中的概率分布）
            int level = assignLevel();
            idToLevel.put(id, level);
            
            // 如果是第一个元素，设为入口点
            if (entryPoint == -1) {
                entryPoint = id;
                
                // 为每一层创建空邻居列表
                for (int i = 0; i <= level; i++) {
                    graph.get(i).put(id, new ArrayList<>());
                }
                
                return true;
            }
            
            // 从最高层开始插入
            int currentEntryPoint = entryPoint;
            
            for (int currentLevel = maxLevel - 1; currentLevel > level; currentLevel--) {
                // 在当前层搜索最近邻
                currentEntryPoint = searchLayerClosest(normalizedVector, currentEntryPoint, currentLevel);
            }
            
            // 对每一层执行插入
            for (int currentLevel = Math.min(level, maxLevel - 1); currentLevel >= 0; currentLevel--) {
                // 在当前层搜索efConstruction个最近邻
                List<SearchResult> neighbors = searchLayer(normalizedVector, currentEntryPoint, efConstruction, currentLevel);
                
                // 确保neighbors不为null
                if (neighbors == null) {
                    neighbors = new ArrayList<>();
                }
                
                // 获取最近的M个邻居
                List<Integer> selectedNeighbors = selectNeighbors(normalizedVector, neighbors, m);
                
                // 创建当前节点的邻居列表
                graph.get(currentLevel).put(id, new ArrayList<>(selectedNeighbors));
                
                // 为每个选中的邻居添加反向连接
                for (int neighborId : selectedNeighbors) {
                    List<Integer> neighborConnections = graph.get(currentLevel).get(neighborId);
                    
                    // 确保neighborConnections不为null
                    if (neighborConnections == null) {
                        neighborConnections = new ArrayList<>();
                        graph.get(currentLevel).put(neighborId, neighborConnections);
                    }
                    
                    // 添加反向连接
                    neighborConnections.add(id);
                    
                    // 如果邻居连接数超过M，进行修剪
                    if (neighborConnections.size() > m) {
                        Vector neighborVector = vectors.get(neighborId);
                        List<SearchResult> neighborNeighbors = new ArrayList<>();
                        
                        for (int nnId : neighborConnections) {
                            Vector nnVector = vectors.get(nnId);
                            if (nnVector != null) {
                                float distance = calculateDistance(neighborVector, nnVector);
                                neighborNeighbors.add(new SearchResult(nnId, distance));
                            }
                        }
                        
                        List<Integer> prunedNeighbors = selectNeighbors(neighborVector, neighborNeighbors, m);
                        graph.get(currentLevel).put(neighborId, new ArrayList<>(prunedNeighbors));
                    }
                }
                
                // 使用找到的最近邻居作为下一层的入口点（如果存在）
                if (!selectedNeighbors.isEmpty()) {
                    currentEntryPoint = selectedNeighbors.get(0);
                } else {
                    currentEntryPoint = id;
                }
            }

            // 如果新节点的层级高于当前入口点，更新入口点
            if (level > idToLevel.getOrDefault(entryPoint, -1)) {
                entryPoint = id;
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
            if (!vectors.containsKey(id)) {
                return false;
            }
            
            // 获取向量层级
            int level = idToLevel.getOrDefault(id, -1);
            if (level == -1) {
                // 如果没有层级信息，只移除向量
                vectors.remove(id);
                return true;
            }
            
            // 从每一层移除连接
            for (int currentLevel = 0; currentLevel <= level; currentLevel++) {
                Map<Integer, List<Integer>> currentGraph = graph.get(currentLevel);
                if (currentGraph == null) continue;
                
                // 获取当前节点的邻居
                List<Integer> neighbors = currentGraph.get(id);
                if (neighbors == null) continue;
                
                // 创建邻居列表的副本，避免并发修改异常
                List<Integer> neighborsCopy = new ArrayList<>(neighbors);
                
                // 从每个邻居的连接中移除当前节点
                for (int neighborId : neighborsCopy) {
                    List<Integer> neighborConnections = currentGraph.get(neighborId);
                    if (neighborConnections != null) {
                        neighborConnections.remove(Integer.valueOf(id));
                    }
                }
                
                // 从图中移除当前节点
                currentGraph.remove(id);
            }
            
            // 如果删除的是入口点，需要更新入口点
            if (id == entryPoint) {
                // 如果没有其他节点，设置入口点为-1
                if (vectors.size() == 1) {
                    entryPoint = -1;
                } else {
                    // 否则选择一个新的入口点（简单地选择第一个找到的节点）
                    for (int candidateId : vectors.keySet()) {
                        if (candidateId != id) {
                            entryPoint = candidateId;
                            break;
                        }
                    }
                }
            }
            
            // 移除向量和层级信息
            vectors.remove(id);
            idToLevel.remove(id);
            
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
            
            if (entryPoint == -1) {
                // 如果没有入口点，随机选择一个向量作为入口点
                if (!vectors.isEmpty()) {
                    entryPoint = vectors.keySet().iterator().next();
                } else {
                    return new ArrayList<>(); // 如果没有向量，返回空列表
                }
            }
            
            // 归一化查询向量
            Vector normalizedQuery = normalizeVector(queryVector);
            
            // 增加搜索ef值，确保有足够的候选
            int searchEf = Math.max(ef, k * 4); // 增加到k的4倍，提高准确率
            
            // 从最高层开始搜索
            int currentEntryPoint = entryPoint;
            
            for (int currentLevel = maxLevel - 1; currentLevel > 0; currentLevel--) {
                currentEntryPoint = searchLayerClosest(normalizedQuery, currentEntryPoint, currentLevel);
            }
            
            // 在最底层搜索k个最近邻
            List<SearchResult> results = searchLayer(normalizedQuery, currentEntryPoint, searchEf, 0);
            
            // 只返回前k个结果
            return results.size() <= k ? results : results.subList(0, k);
        } catch (Exception e) {
            log.error("搜索最近邻时发生异常: {}", e.getMessage(), e);
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
     * 针对高维向量优化索引参数
     * 特别适用于1536维以上的向量
     * 
     * @param dimension 向量维度
     * @return 是否成功优化
     */
    public boolean optimizeForHighDimension(int dimension) {
        try {
            log.info("正在为{}维向量优化HNSW索引参数...", dimension);
            
            // 根据维度自动调整参数
            if (dimension >= 1000) {
                // 高维向量（如OpenAI的1536维向量）
                int newM = 32;
                int newEfConstruction = 400;
                int newEf = 400;
                
                // 对于超高维向量，进一步增加参数
                if (dimension >= 1500) {
                    newM = 48;
                    newEfConstruction = 600;
                    newEf = 600;
                }
                
                // 设置参数
                setIndexParameters(newM, newEfConstruction, newEf, true, true);
                
                log.info("高维向量优化完成，参数设置为: M={}, efConstruction={}, ef={}, 使用余弦相似度={}, 向量归一化={}", 
                        m, efConstruction, ef, useCosineSimilarity, normalizeVectors);
                
                // 如果索引中已有向量，建议重建索引
                if (!vectors.isEmpty()) {
                    log.info("索引中已有{}个向量，建议调用buildIndex()重建索引以应用新参数", vectors.size());
                }
                
                return true;
            } else {
                // 低维向量，使用默认参数即可
                log.info("维度较低，保持默认参数: M={}, efConstruction={}, ef={}", m, efConstruction, ef);
                return false;
            }
        } catch (Exception e) {
            log.error("优化高维向量参数时发生异常: {}", e.getMessage(), e);
            return false;
        }
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
            log.info("开始重建HNSW索引，当前参数: M={}, efConstruction={}, ef={}, 使用余弦相似度={}, 向量归一化={}", 
                    m, efConstruction, ef, useCosineSimilarity, normalizeVectors);
            
            // 保存所有向量的副本
            Map<Integer, Vector> savedVectors = new HashMap<>(vectors);
            
            // 清空当前索引结构
            vectors.clear();
            idToLevel.clear();
            
            // 清空图结构
            for (int i = 0; i < maxLevel; i++) {
                graph.get(i).clear();
            }
            
            // 重置入口点
            entryPoint = -1;
            
            // 重新添加所有向量
            List<Vector> vectorList = new ArrayList<>(savedVectors.values());
            
            // 按照ID排序，确保添加顺序一致
            vectorList.sort(Comparator.comparing(Vector::getId));
            
            int successCount = 0;
            for (Vector vector : vectorList) {
                if (addVector(vector)) {
                    successCount++;
                }
            }
            
            long endTime = System.currentTimeMillis();
            log.info("HNSW索引重建完成，包含 {} 个向量，耗时: {} 毫秒", 
                    successCount, (endTime - startTime));
            
            return true;
        } catch (Exception e) {
            log.error("重建索引时发生异常: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 获取当前索引的性能统计信息
     * 
     * @return 包含索引性能信息的字符串
     */
    public String getIndexStats() {
        StringBuilder stats = new StringBuilder();
        stats.append("HNSW索引统计信息:\n");
        stats.append("- 向量数量: ").append(vectors.size()).append("\n");
        stats.append("- 向量维度: ").append(dimension).append("\n");
        stats.append("- 最大元素数: ").append(maxElements).append("\n");
        stats.append("- M值(每节点最大连接数): ").append(m).append("\n");
        stats.append("- efConstruction(构建搜索宽度): ").append(efConstruction).append("\n");
        stats.append("- ef(搜索候选集大小): ").append(ef).append("\n");
        stats.append("- 最大层级: ").append(maxLevel).append("\n");
        stats.append("- 使用余弦相似度: ").append(useCosineSimilarity).append("\n");
        stats.append("- 向量归一化: ").append(normalizeVectors).append("\n");
        
        // 计算每层的节点数
        Map<Integer, Integer> levelCounts = new HashMap<>();
        for (int level : idToLevel.values()) {
            levelCounts.put(level, levelCounts.getOrDefault(level, 0) + 1);
        }
        
        stats.append("- 层级分布:\n");
        for (int i = 0; i < maxLevel; i++) {
            stats.append("  - 层级 ").append(i).append(": ")
                 .append(levelCounts.getOrDefault(i, 0)).append(" 节点\n");
        }
        
        return stats.toString();
    }
    
    /**
     * 在指定层查找最近的节点
     */
    private int searchLayerClosest(Vector queryVector, int entryPointId, int level) {
        if (!vectors.containsKey(entryPointId)) {
            return entryPointId; // 如果入口点不存在，直接返回
        }
        
        int currentId = entryPointId;
        float currentDistance = calculateDistance(vectors.get(currentId), queryVector);
        
        boolean changed;
        // 使用贪婪搜索找到最近的节点
        do {
            changed = false;
            
            // 获取当前节点在当前层的邻居
            Map<Integer, List<Integer>> layerGraph = graph.get(level);
            if (layerGraph == null) {
                return currentId;
            }
            
            List<Integer> neighbors = layerGraph.get(currentId);
            if (neighbors == null || neighbors.isEmpty()) {
                return currentId;
            }
            
            // 检查是否有更近的邻居
            for (int neighborId : neighbors) {
                if (!vectors.containsKey(neighborId)) {
                    continue; // 跳过已删除的向量
                }
                
                float distance = calculateDistance(vectors.get(neighborId), queryVector);
                if (distance < currentDistance) {
                    currentDistance = distance;
                    currentId = neighborId;
                    changed = true;
                    break; // 找到更近的邻居后立即跳出，加速搜索
                }
            }
        } while (changed);
        
        return currentId;
    }
    
    /**
     * 在指定层搜索最近的k个节点
     */
    private List<SearchResult> searchLayer(Vector queryVector, int entryPointId, int ef, int level) {
        try {
            if (!vectors.containsKey(entryPointId)) {
                return new ArrayList<>(); // 如果入口点不存在，返回空列表
            }
            
            // 初始化结果集和候选集
            PriorityQueue<SearchResult> resultSet = new PriorityQueue<>(
                    Comparator.comparing(SearchResult::getDistance).reversed());
            
            PriorityQueue<SearchResult> candidateSet = new PriorityQueue<>(
                    Comparator.comparing(SearchResult::getDistance));
            
            // 添加入口点
            float distance = calculateDistance(vectors.get(entryPointId), queryVector);
            SearchResult entryResult = new SearchResult(entryPointId, distance);
            resultSet.add(entryResult);
            candidateSet.add(entryResult);
            
            // 已访问节点集合
            Set<Integer> visited = new HashSet<>();
            visited.add(entryPointId);
            
            float furthestDistance = distance; // 跟踪结果集中最远的距离
            
            // 当候选集不为空时
            while (!candidateSet.isEmpty()) {
                SearchResult current = candidateSet.poll();
                if (current == null) continue;
                
                // 如果当前候选的距离大于结果集中最远的距离，则停止搜索
                if (current.getDistance() > furthestDistance) {
                    break;
                }
                
                // 获取当前节点的邻居
                Map<Integer, List<Integer>> layerGraph = graph.get(level);
                if (layerGraph == null) {
                    continue;
                }
                
                List<Integer> neighbors = layerGraph.get(current.getId());
                if (neighbors == null || neighbors.isEmpty()) {
                    continue;
                }
                
                // 检查每个邻居
                for (int neighborId : neighbors) {
                    if (visited.contains(neighborId) || !vectors.containsKey(neighborId)) {
                        continue;
                    }
                    
                    visited.add(neighborId);
                    
                    Vector neighborVector = vectors.get(neighborId);
                    if (neighborVector == null) continue;
                    
                    float neighborDistance = calculateDistance(neighborVector, queryVector);
                    SearchResult neighborResult = new SearchResult(neighborId, neighborDistance);
                    
                    // 如果结果集大小小于ef或者新节点比结果集中最远的节点更近
                    if (resultSet.size() < ef || neighborDistance < furthestDistance) {
                        // 添加到候选集
                        candidateSet.add(neighborResult);
                        
                        // 添加到结果集
                        resultSet.add(neighborResult);
                        
                        // 如果结果集超过ef，移除最远的元素
                        if (resultSet.size() > ef) {
                            resultSet.poll();
                            // 更新最远距离
                            furthestDistance = resultSet.peek().getDistance();
                        }
                    }
                }
            }
            
            // 转换为列表并排序
            List<SearchResult> results = new ArrayList<>(resultSet);
            results.sort(Comparator.comparing(SearchResult::getDistance));
            
            return results;
        } catch (Exception e) {
            log.error("搜索层时发生异常: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }
    
    /**
     * 从候选邻居中选择最佳的M个邻居
     */
    private List<Integer> selectNeighbors(Vector vector, List<SearchResult> candidates, int m) {
        try {
            // 如果候选数量小于等于M，直接返回所有候选
            if (candidates == null || candidates.isEmpty()) {
                return new ArrayList<>();
            }
            
            if (candidates.size() <= m) {
                return candidates.stream()
                        .map(SearchResult::getId)
                        .collect(Collectors.toList());
            }
            
            // 否则选择最近的M个邻居
            return candidates.stream()
                    .sorted()
                    .limit(m)
                    .map(SearchResult::getId)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            log.error("选择邻居时发生异常: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }
    
    /**
     * 随机分配层级
     * 使用与HNSW论文相同的概率分布
     */
    private int assignLevel() {
        // 基础层级为0
        int level = 0;
        
        // 以1/e的概率增加层级，最大不超过maxLevel-1
        double probability = 1.0 / Math.E;
        
        while (ThreadLocalRandom.current().nextDouble() < probability && level < maxLevel - 1) {
            level++;
        }
        
        return level;
    }
} 