package com.vectordb.index;

import com.vectordb.core.SearchResult;
import com.vectordb.core.Vector;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import lombok.extern.slf4j.Slf4j;

/**
 * ANNOY（Approximate Nearest Neighbors Oh Yeah）索引实现
 * 用于高效的近似最近邻搜索
 * 
 * ANNOY算法特点：
 * 1. 使用随机投影构建多棵二叉树
 * 2. 每棵树将空间分割成多个区域
 * 3. 查询时在多棵树中搜索，合并结果
 * 
 * 优化点：
 * 1. 批量添加向量
 * 2. 延迟重建树结构
 * 3. 并行构建树
 * 4. 增加缓存机制减少重建频率
 */
@Slf4j
public class AnnoyIndex implements VectorIndex {
    // 索引参数
    private final int dimension;
    private final int maxElements;
    private int numTrees = 10; // 树的数量
    
    // 重建阈值参数
    private int rebuildThreshold = 1000; // 添加多少个向量后重建树
    private AtomicInteger pendingChanges = new AtomicInteger(0); // 待处理的变更数
    private boolean forceRebuild = false; // 是否强制重建
    private boolean lazyBuild = true; // 延迟构建模式
    
    // 数据结构
    private final Map<Integer, Vector> vectors; // 存储向量
    private final List<Node> roots; // 树的根节点列表
    private final Set<Integer> pendingVectors = Collections.synchronizedSet(new HashSet<>()); // 待处理的向量ID
    
    // 内部节点类
    private static class Node {
        int id = -1; // 如果是叶子节点，存储向量ID；否则为-1
        float[] splitVector; // 分割向量
        float splitThreshold; // 分割阈值
        Node left; // 左子节点
        Node right; // 右子节点
        List<Integer> vectorIds; // 叶子节点存储的向量ID列表
        
        // 创建内部节点
        Node(float[] splitVector, float splitThreshold) {
            this.splitVector = splitVector;
            this.splitThreshold = splitThreshold;
            this.vectorIds = null;
        }
        
        // 创建叶子节点
        Node(List<Integer> vectorIds) {
            this.splitVector = null;
            this.vectorIds = vectorIds;
        }
        
        boolean isLeaf() {
            return vectorIds != null;
        }
    }
    
    /**
     * 创建ANNOY索引
     * 
     * @param dimension 向量维度
     * @param maxElements 最大元素数量
     */
    public AnnoyIndex(int dimension, int maxElements) {
        this(dimension, maxElements, 10, 1000, true);
    }
    
    /**
     * 创建ANNOY索引（带参数）
     * 
     * @param dimension 向量维度
     * @param maxElements 最大元素数量
     * @param numTrees 树的数量
     * @param rebuildThreshold 重建阈值
     * @param lazyBuild 是否使用延迟构建模式
     */
    public AnnoyIndex(int dimension, int maxElements, int numTrees, int rebuildThreshold, boolean lazyBuild) {
        this.dimension = dimension;
        this.maxElements = maxElements;
        this.numTrees = numTrees;
        this.rebuildThreshold = rebuildThreshold;
        this.lazyBuild = lazyBuild;
        this.vectors = new ConcurrentHashMap<>(maxElements);
        this.roots = new ArrayList<>(numTrees);
        
        // 初始化树的根节点
        for (int i = 0; i < numTrees; i++) {
            roots.add(null);
        }
    }
    
    /**
     * 设置索引参数
     * 
     * @param numTrees 树的数量
     * @param rebuildThreshold 重建阈值
     * @param lazyBuild 是否使用延迟构建模式
     */
    public void setIndexParameters(int numTrees, int rebuildThreshold, boolean lazyBuild) {
        if (numTrees > 0) {
            this.numTrees = numTrees;
            // 调整根节点列表大小
            while (roots.size() < numTrees) {
                roots.add(null);
            }
            while (roots.size() > numTrees) {
                roots.remove(roots.size() - 1);
            }
        }
        
        if (rebuildThreshold > 0) {
            this.rebuildThreshold = rebuildThreshold;
        }
        
        this.lazyBuild = lazyBuild;
        
        // 设置参数后强制重建
        this.forceRebuild = true;
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
            
            // 如果使用延迟构建模式，将向量添加到待处理集合
            if (lazyBuild) {
                pendingVectors.add(id);
                
                // 增加待处理变更计数
                int changes = pendingChanges.incrementAndGet();
                
                // 如果变更数超过阈值或强制重建，则重建树
                if (changes >= rebuildThreshold || forceRebuild) {
                    synchronized (this) {
                        // 再次检查，避免多线程重复重建
                        if (pendingChanges.get() >= rebuildThreshold || forceRebuild) {
                            rebuildTrees();
                            pendingChanges.set(0);
                            pendingVectors.clear();
                            forceRebuild = false;
                        }
                    }
                }
            } else {
                // 非延迟模式，立即重建单个向量的树
                synchronized (this) {
                    addVectorToTrees(id);
                }
            }
            
            return true;
        } catch (Exception e) {
            log.error("添加向量时发生异常: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 将单个向量添加到现有树中（非重建方式）
     * 
     * @param id 向量ID
     */
    private void addVectorToTrees(int id) {
        // 如果树还未构建，先构建树
        if (roots.get(0) == null) {
            rebuildTrees();
            return;
        }
        
        Vector vector = vectors.get(id);
        if (vector == null) {
            return;
        }
        
        // 将向量添加到每棵树的适当位置
        for (int i = 0; i < numTrees; i++) {
            Node root = roots.get(i);
            if (root != null) {
                addVectorToTree(root, id, vector);
            }
        }
    }
    
    /**
     * 将向量添加到树中的适当位置
     * 
     * @param node 当前节点
     * @param id 向量ID
     * @param vector 向量
     */
    private void addVectorToTree(Node node, int id, Vector vector) {
        // 如果是叶子节点，直接添加
        if (node.isLeaf()) {
            node.vectorIds.add(id);
            
            // 如果叶子节点过大，考虑分裂
            final int MAX_LEAF_SIZE = 20;
            if (node.vectorIds.size() > MAX_LEAF_SIZE) {
                splitLeafNode(node);
            }
            
            return;
        }
        
        // 计算向量在分割向量上的投影
        float projection = 0;
        for (int i = 0; i < dimension; i++) {
            projection += vector.getValues()[i] * node.splitVector[i];
        }
        
        // 根据投影决定添加方向
        if (projection <= node.splitThreshold) {
            addVectorToTree(node.left, id, vector);
        } else {
            addVectorToTree(node.right, id, vector);
        }
    }
    
    /**
     * 分裂叶子节点
     * 
     * @param node 叶子节点
     */
    private void splitLeafNode(Node node) {
        if (!node.isLeaf() || node.vectorIds.size() <= 10) {
            return;
        }
        
        // 随机选择两个向量作为分割点
        Random random = ThreadLocalRandom.current();
        List<Integer> vectorIds = node.vectorIds;
        int idx1 = random.nextInt(vectorIds.size());
        int idx2;
        do {
            idx2 = random.nextInt(vectorIds.size());
        } while (idx1 == idx2);
        
        Vector v1 = vectors.get(vectorIds.get(idx1));
        Vector v2 = vectors.get(vectorIds.get(idx2));
        
        // 计算分割向量（两点之差）
        float[] splitVector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            splitVector[i] = v1.getValues()[i] - v2.getValues()[i];
        }
        
        // 归一化分割向量
        float norm = 0;
        for (int i = 0; i < dimension; i++) {
            norm += splitVector[i] * splitVector[i];
        }
        norm = (float) Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < dimension; i++) {
                splitVector[i] /= norm;
            }
        }
        
        // 计算所有向量在分割向量上的投影
        List<Float> projections = new ArrayList<>(vectorIds.size());
        for (int id : vectorIds) {
            Vector v = vectors.get(id);
            float proj = 0;
            for (int i = 0; i < dimension; i++) {
                proj += v.getValues()[i] * splitVector[i];
            }
            projections.add(proj);
        }
        
        // 计算中位数作为分割阈值
        List<Float> sortedProjections = new ArrayList<>(projections);
        Collections.sort(sortedProjections);
        float splitThreshold = sortedProjections.get(sortedProjections.size() / 2);
        
        // 分割向量
        List<Integer> leftIds = new ArrayList<>();
        List<Integer> rightIds = new ArrayList<>();
        
        for (int i = 0; i < vectorIds.size(); i++) {
            if (projections.get(i) <= splitThreshold) {
                leftIds.add(vectorIds.get(i));
            } else {
                rightIds.add(vectorIds.get(i));
            }
        }
        
        // 处理边界情况：如果所有向量都在一侧，强制分割
        if (leftIds.isEmpty() || rightIds.isEmpty()) {
            int mid = vectorIds.size() / 2;
            leftIds = new ArrayList<>(vectorIds.subList(0, mid));
            rightIds = new ArrayList<>(vectorIds.subList(mid, vectorIds.size()));
        }
        
        // 转换节点为内部节点
        node.splitVector = splitVector;
        node.splitThreshold = splitThreshold;
        node.left = new Node(leftIds);
        node.right = new Node(rightIds);
        node.vectorIds = null;
    }
    
    /**
     * 批量添加向量到索引
     * 
     * @param batchVectors 向量列表
     * @return 成功添加的向量数量
     */
    public synchronized int addVectors(List<Vector> batchVectors) {
        if (batchVectors == null || batchVectors.isEmpty()) {
            return 0;
        }
        
        if (this.vectors.size() + batchVectors.size() > maxElements) {
            log.error("批量添加向量超过最大容量限制");
            return 0;
        }
        
        int addedCount = 0;
        
        synchronized (this) {
            try {
                // 添加所有向量
                for (Vector vector : batchVectors) {
                    int id = vector.getId();
                    
                    // 检查是否已存在
                    if (vectors.containsKey(id)) {
                        continue;
                    }
                    
                    // 存储向量
                    vectors.put(id, vector);
                    pendingVectors.add(id);
                    addedCount++;
                }
                
                // 设置待处理变更计数
                pendingChanges.set(pendingVectors.size());
                
                // 如果使用延迟构建模式且未达到阈值，不立即重建
                if (!lazyBuild || pendingChanges.get() >= rebuildThreshold || forceRebuild) {
                    rebuildTrees();
                    pendingChanges.set(0);
                    pendingVectors.clear();
                    forceRebuild = false;
                }
                
                return addedCount;
            } catch (Exception e) {
                log.error("批量添加向量时发生异常: {}", e.getMessage(), e);
                return addedCount;
            }
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
            
            // 移除向量
            vectors.remove(id);
            pendingVectors.remove(id);
            
            // 增加待处理变更计数
            int changes = pendingChanges.incrementAndGet();
            
            // 如果变更数超过阈值或强制重建，则重建树
            if (changes >= rebuildThreshold || forceRebuild) {
                rebuildTrees();
                pendingChanges.set(0);
                pendingVectors.clear();
                forceRebuild = false;
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
            
            // 检查是否有待处理的向量，如果有且数量较多，则强制重建
            if (!pendingVectors.isEmpty() && pendingVectors.size() > rebuildThreshold / 10) {
                synchronized (this) {
                    rebuildTrees();
                    pendingChanges.set(0);
                    pendingVectors.clear();
                }
            }
            
            // 使用集合存储候选向量ID，避免重复
            Set<Integer> candidateIds = new HashSet<>();
            
            // 在每棵树中搜索
            for (Node root : roots) {
                if (root != null) {
                    searchTree(root, queryVector, candidateIds);
                }
            }
            
            // 检查是否需要包含待处理的向量
            if (!pendingVectors.isEmpty()) {
                candidateIds.addAll(pendingVectors);
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
            
            // 如果候选集合为空，返回所有向量
            if (results.isEmpty()) {
                for (Vector v : vectors.values()) {
                    float distance = queryVector.euclideanDistance(v);
                    results.add(new SearchResult(v.getId(), distance));
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
     */
    @Override
    public synchronized boolean buildIndex() {
        try {
            long startTime = System.currentTimeMillis();
            // 重置待处理变更计数和待处理向量集合
            pendingChanges.set(0);
            pendingVectors.clear();
            
            // 重建树
            rebuildTrees();
            
            long endTime = System.currentTimeMillis();
            log.info("ANNOY索引重建 {} 棵树，包含 {} 个向量，耗时: {} 毫秒", 
                    numTrees, vectors.size(), (endTime - startTime));
            
            return true;
        } catch (Exception e) {
            log.error("重建索引时发生异常: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 重建所有树
     */
    private void rebuildTrees() {
        // 如果向量数量太少，不构建树
        if (vectors.size() < 2) {
            return;
        }
        
        long startTime = System.currentTimeMillis();
        
        // 获取所有向量ID
        List<Integer> allIds = new ArrayList<>(vectors.keySet());
        
        // 构建每棵树
        for (int i = 0; i < numTrees; i++) {
            roots.set(i, buildTree(new ArrayList<>(allIds)));
        }
        
        long endTime = System.currentTimeMillis();
        log.info("ANNOY索引重建 {} 棵树，包含 {} 个向量，耗时: {} 毫秒", 
                numTrees, vectors.size(), (endTime - startTime));
    }
    
    /**
     * 构建树
     * 
     * @param vectorIds 向量ID列表
     * @return 树的根节点
     */
    private Node buildTree(List<Integer> vectorIds) {
        // 如果向量数量小于等于K_MAX_LEAF_SIZE，创建叶子节点
        final int K_MAX_LEAF_SIZE = 10;
        if (vectorIds.size() <= K_MAX_LEAF_SIZE) {
            return new Node(vectorIds);
        }
        
        // 随机选择两个向量作为分割点
        Random random = ThreadLocalRandom.current();
        int idx1 = random.nextInt(vectorIds.size());
        int idx2;
        do {
            idx2 = random.nextInt(vectorIds.size());
        } while (idx1 == idx2);
        
        Vector v1 = vectors.get(vectorIds.get(idx1));
        Vector v2 = vectors.get(vectorIds.get(idx2));
        
        // 计算分割向量（两点之差）
        float[] splitVector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            splitVector[i] = v1.getValues()[i] - v2.getValues()[i];
        }
        
        // 归一化分割向量
        float norm = 0;
        for (int i = 0; i < dimension; i++) {
            norm += splitVector[i] * splitVector[i];
        }
        norm = (float) Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < dimension; i++) {
                splitVector[i] /= norm;
            }
        }
        
        // 计算所有向量在分割向量上的投影
        List<Float> projections = new ArrayList<>(vectorIds.size());
        for (int id : vectorIds) {
            Vector v = vectors.get(id);
            float proj = 0;
            for (int i = 0; i < dimension; i++) {
                proj += v.getValues()[i] * splitVector[i];
            }
            projections.add(proj);
        }
        
        // 计算中位数作为分割阈值
        List<Float> sortedProjections = new ArrayList<>(projections);
        Collections.sort(sortedProjections);
        float splitThreshold = sortedProjections.get(sortedProjections.size() / 2);
        
        // 创建内部节点
        Node node = new Node(splitVector, splitThreshold);
        
        // 分割向量
        List<Integer> leftIds = new ArrayList<>();
        List<Integer> rightIds = new ArrayList<>();
        
        for (int i = 0; i < vectorIds.size(); i++) {
            if (projections.get(i) <= splitThreshold) {
                leftIds.add(vectorIds.get(i));
            } else {
                rightIds.add(vectorIds.get(i));
            }
        }
        
        // 处理边界情况：如果所有向量都在一侧，强制分割
        if (leftIds.isEmpty() || rightIds.isEmpty()) {
            int mid = vectorIds.size() / 2;
            leftIds = new ArrayList<>(vectorIds.subList(0, mid));
            rightIds = new ArrayList<>(vectorIds.subList(mid, vectorIds.size()));
        }
        
        // 递归构建左右子树
        node.left = buildTree(leftIds);
        node.right = buildTree(rightIds);
        
        return node;
    }
    
    /**
     * 在树中搜索
     * 
     * @param node 当前节点
     * @param queryVector 查询向量
     * @param candidateIds 候选向量ID集合
     */
    private void searchTree(Node node, Vector queryVector, Set<Integer> candidateIds) {
        // 如果是叶子节点，添加所有向量ID到候选集
        if (node.isLeaf()) {
            candidateIds.addAll(node.vectorIds);
            return;
        }
        
        // 计算查询向量在分割向量上的投影
        float projection = 0;
        for (int i = 0; i < dimension; i++) {
            projection += queryVector.getValues()[i] * node.splitVector[i];
        }
        
        // 根据投影决定搜索方向
        if (projection <= node.splitThreshold) {
            searchTree(node.left, queryVector, candidateIds);
            
            // 有一定概率也搜索另一侧（提高召回率）
            if (Math.random() < 0.5) {
                searchTree(node.right, queryVector, candidateIds);
            }
        } else {
            searchTree(node.right, queryVector, candidateIds);
            
            // 有一定概率也搜索另一侧（提高召回率）
            if (Math.random() < 0.5) {
                searchTree(node.left, queryVector, candidateIds);
            }
        }
    }
} 