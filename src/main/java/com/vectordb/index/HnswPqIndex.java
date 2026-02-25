package com.vectordb.index;

import com.vectordb.config.CompressionConfig;
import com.vectordb.core.SearchResult;
import com.vectordb.core.Vector;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

/**
 * HNSW + Product Quantization 混合索引实现
 * 结合HNSW的快速搜索和PQ的内存压缩能力
 * 提供高召回率的同时大幅降低内存占用
 */
@Slf4j
public class HnswPqIndex implements VectorIndex {

    // 基础参数
    private final int dimension;
    private final int maxElements;
    private final CompressionConfig compressionConfig;

    // HNSW参数
    private int m = 32;
    private int efConstruction = 64;
    private int ef = 64;
    private int maxLevel;
    private boolean useCosineSimilarity = true;
    private boolean normalizeVectors = true;

    // PQ参数
    private int pqSubspaces;
    private int pqBits;
    private int pqIterations;
    private int subDim;
    private int nCentroids;
    private boolean trained = false;

    // 数据结构
    private final Map<Integer, Vector> vectors;
    private final Map<Integer, Integer> idToLevel;
    private final Map<Integer, Integer> idToIndex;
    private final List<Map<Integer, List<Integer>>> graph;

    // PQ数据结构
    private float[][][] codebooks;  // [subspace][centroid][subDim]
    private byte[][] codes;         // [vectorIndex][subspace]
    private int currentSize = 0;

    // 入口点
    private volatile int entryPoint = -1;

    // 训练样本缓存
    private List<Vector> trainingSamples;
    private static final int TRAINING_SAMPLE_SIZE = 10000;

    /**
     * 创建HNSW+PQ混合索引
     *
     * @param dimension 向量维度
     * @param maxElements 最大元素数量
     * @param compressionConfig 压缩配置
     */
    public HnswPqIndex(int dimension, int maxElements, CompressionConfig compressionConfig) {
        this.dimension = dimension;
        this.maxElements = maxElements;
        this.compressionConfig = compressionConfig;

        // 初始化PQ参数
        this.pqSubspaces = compressionConfig.getPqSubspaces();
        this.pqBits = compressionConfig.getPqBits();
        this.pqIterations = compressionConfig.getPqIterations();

        // 确保dimension能被pqSubspaces整除
        if (dimension % pqSubspaces != 0) {
            // 自动调整pqSubspaces
            this.pqSubspaces = findBestSubspaceDivisor(dimension);
            log.warn("维度{}不能被PQ子空间数{}整除，自动调整为{}" ,
                    dimension, compressionConfig.getPqSubspaces(), this.pqSubspaces);
        }

        this.subDim = dimension / pqSubspaces;
        this.nCentroids = 1 << pqBits;

        // 初始化HNSW参数
        this.maxLevel = (int) (Math.log(maxElements) / Math.log(m)) + 1;

        // 初始化数据结构
        this.vectors = new ConcurrentHashMap<>(maxElements);
        this.idToLevel = new ConcurrentHashMap<>(maxElements);
        this.idToIndex = new ConcurrentHashMap<>(maxElements);
        this.graph = new ArrayList<>();
        this.trainingSamples = new ArrayList<>();

        for (int i = 0; i < maxLevel; i++) {
            graph.add(new ConcurrentHashMap<>());
        }

        // 预分配PQ数据结构
        this.codebooks = new float[pqSubspaces][nCentroids][subDim];
        this.codes = new byte[maxElements][pqSubspaces];

        log.info("创建HNSW+PQ索引: 维度={}, PQ子空间={}, 聚类中心={}, 压缩比={:.2f}x",
                dimension, pqSubspaces, nCentroids, getCompressionRatio());
    }

    /**
     * 查找最佳的子空间除数
     */
    private int findBestSubspaceDivisor(int dimension) {
        // 寻找能使每个子空间维度在2-4之间的除数
        for (int subspaces = dimension / 2; subspaces >= 4; subspaces--) {
            if (dimension % subspaces == 0) {
                int subDim = dimension / subspaces;
                if (subDim >= 2 && subDim <= 4) {
                    return subspaces;
                }
            }
        }
        // 如果找不到理想的，返回最大公约数
        for (int subspaces = Math.min(dimension, 128); subspaces >= 8; subspaces--) {
            if (dimension % subspaces == 0) {
                return subspaces;
            }
        }
        return dimension; // 最坏情况，每个维度一个子空间
    }

    /**
     * 获取压缩比
     */
    public double getCompressionRatio() {
        return compressionConfig.getCompressionRatio(dimension);
    }

    /**
     * 获取内存节省百分比
     */
    public double getMemorySavings() {
        return compressionConfig.getMemorySavings(dimension);
    }

    /**
     * 检查是否已训练
     */
    public boolean isTrained() {
        return trained;
    }

    @Override
    public boolean addVector(Vector vector) {
        try {
            int id = vector.getId();

            if (vectors.containsKey(id)) {
                return false;
            }

            if (vectors.size() >= maxElements) {
                return false;
            }

            // 归一化向量
            Vector normalizedVector = normalizeVector(vector);

            // 收集训练样本
            if (!trained && trainingSamples.size() < TRAINING_SAMPLE_SIZE) {
                trainingSamples.add(normalizedVector);
                if (trainingSamples.size() >= Math.min(TRAINING_SAMPLE_SIZE, maxElements / 10)) {
                    train();
                }
            }

            // 如果没有训练，使用原始向量存储
            if (!trained) {
                return addVectorRaw(normalizedVector);
            }

            // 使用PQ编码并添加
            return addVectorCompressed(normalizedVector);

        } catch (Exception e) {
            log.error("添加向量时发生异常: {}", e.getMessage(), e);
            return false;
        }
    }

    /**
     * 训练PQ码本
     */
    private synchronized void train() {
        if (trained || trainingSamples.isEmpty()) {
            return;
        }

        long startTime = System.currentTimeMillis();
        log.info("开始训练PQ码本，样本数: {}", trainingSamples.size());

        try {
            // 对每个子空间进行KMeans聚类
            for (int m = 0; m < pqSubspaces; m++) {
                trainSubspace(m);
            }

            trained = true;
            long endTime = System.currentTimeMillis();
            log.info("PQ码本训练完成，耗时: {} ms", (endTime - startTime));

        } catch (Exception e) {
            log.error("训练PQ码本时发生异常: {}", e.getMessage(), e);
        }
    }

    /**
     * 训练单个子空间
     */
    private void trainSubspace(int subspaceIdx) {
        // 提取子空间数据
        float[][] subVectors = new float[trainingSamples.size()][subDim];
        for (int i = 0; i < trainingSamples.size(); i++) {
            float[] vec = trainingSamples.get(i).getValues();
            System.arraycopy(vec, subspaceIdx * subDim, subVectors[i], 0, subDim);
        }

        // KMeans++初始化
        initializeCentroidsKMeansPlusPlus(subspaceIdx, subVectors);

        // KMeans迭代
        for (int iter = 0; iter < pqIterations; iter++) {
            // E步: 分配样本到最近的聚类中心
            int[] assignments = new int[trainingSamples.size()];
            boolean changed = false;

            for (int i = 0; i < trainingSamples.size(); i++) {
                int nearest = findNearestCentroid(subspaceIdx, subVectors[i]);
                if (assignments[i] != nearest) {
                    assignments[i] = nearest;
                    changed = true;
                }
            }

            if (!changed && iter > 5) break;  // 提前收敛

            // M步: 更新聚类中心
            updateCentroids(subspaceIdx, subVectors, assignments);
        }
    }

    /**
     * KMeans++初始化
     */
    private void initializeCentroidsKMeansPlusPlus(int subspaceIdx, float[][] subVectors) {
        Random random = new Random(42 + subspaceIdx);

        // 第一个中心随机选择
        int firstIdx = random.nextInt(subVectors.length);
        System.arraycopy(subVectors[firstIdx], 0, codebooks[subspaceIdx][0], 0, subDim);

        float[] minDistances = new float[subVectors.length];
        Arrays.fill(minDistances, Float.MAX_VALUE);

        // 选择剩余中心
        for (int c = 1; c < nCentroids && c < subVectors.length; c++) {
            float totalDist = 0;

            // 更新最小距离
            for (int i = 0; i < subVectors.length; i++) {
                float dist = computeSubspaceDistance(subVectors[i], codebooks[subspaceIdx][c - 1]);
                if (dist < minDistances[i]) {
                    minDistances[i] = dist;
                }
                totalDist += minDistances[i];
            }

            // 按概率选择下一个中心
            float target = random.nextFloat() * totalDist;
            float cumsum = 0;
            int selected = 0;

            for (int i = 0; i < subVectors.length; i++) {
                cumsum += minDistances[i];
                if (cumsum >= target) {
                    selected = i;
                    break;
                }
            }

            System.arraycopy(subVectors[selected], 0, codebooks[subspaceIdx][c], 0, subDim);
        }
    }

    /**
     * 更新聚类中心
     */
    private void updateCentroids(int subspaceIdx, float[][] subVectors, int[] assignments) {
        int[] counts = new int[nCentroids];
        float[][] newCentroids = new float[nCentroids][subDim];

        // 累加
        for (int i = 0; i < subVectors.length; i++) {
            int cluster = assignments[i];
            counts[cluster]++;
            for (int d = 0; d < subDim; d++) {
                newCentroids[cluster][d] += subVectors[i][d];
            }
        }

        // 平均
        for (int c = 0; c < nCentroids; c++) {
            if (counts[c] > 0) {
                float invCount = 1.0f / counts[c];
                for (int d = 0; d < subDim; d++) {
                    codebooks[subspaceIdx][c][d] = newCentroids[c][d] * invCount;
                }
            }
        }
    }

    /**
     * 查找最近的聚类中心
     */
    private int findNearestCentroid(int subspaceIdx, float[] subVector) {
        int nearest = 0;
        float minDist = Float.MAX_VALUE;

        for (int i = 0; i < nCentroids; i++) {
            float dist = computeSubspaceDistance(subVector, codebooks[subspaceIdx][i]);
            if (dist < minDist) {
                minDist = dist;
                nearest = i;
            }
        }

        return nearest;
    }

    /**
     * 计算子空间距离
     */
    private float computeSubspaceDistance(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < subDim; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    /**
     * PQ编码向量
     */
    private byte[] encodeVector(Vector vector) {
        byte[] code = new byte[pqSubspaces];
        float[] values = vector.getValues();

        for (int m = 0; m < pqSubspaces; m++) {
            float[] subVector = new float[subDim];
            System.arraycopy(values, m * subDim, subVector, 0, subDim);
            code[m] = (byte) findNearestCentroid(m, subVector);
        }

        return code;
    }

    /**
     * 计算PQ距离（ADC: Asymmetric Distance Computation）
     */
    private float computePQDistance(Vector query, int vectorIdx) {
        float[] queryValues = query.getValues();
        float distance = 0;

        for (int m = 0; m < pqSubspaces; m++) {
            int centroidIdx = codes[vectorIdx][m] & 0xFF;
            float[] centroid = codebooks[m][centroidIdx];

            for (int d = 0; d < subDim; d++) {
                float diff = queryValues[m * subDim + d] - centroid[d];
                distance += diff * diff;
            }
        }

        return distance;
    }

    /**
     * 添加原始向量（未压缩）
     */
    private boolean addVectorRaw(Vector vector) {
        int id = vector.getId();
        int index = currentSize++;

        vectors.put(id, vector);
        idToIndex.put(id, index);

        int level = assignLevel();
        idToLevel.put(id, level);

        if (entryPoint == -1) {
            entryPoint = id;
            for (int i = 0; i <= level; i++) {
                graph.get(i).put(id, new ArrayList<>());
            }
            return true;
        }

        // HNSW插入逻辑
        int currentEntryPoint = entryPoint;

        for (int currentLevel = maxLevel - 1; currentLevel > level; currentLevel--) {
            currentEntryPoint = searchLayerClosest(vector, currentEntryPoint, currentLevel);
        }

        for (int currentLevel = Math.min(level, maxLevel - 1); currentLevel >= 0; currentLevel--) {
            List<SearchResult> neighbors = searchLayer(vector, currentEntryPoint, efConstruction, currentLevel);
            List<Integer> selectedNeighbors = selectNeighbors(vector, neighbors, m);

            graph.get(currentLevel).put(id, new ArrayList<>(selectedNeighbors));

            for (int neighborId : selectedNeighbors) {
                List<Integer> neighborConnections = graph.get(currentLevel).get(neighborId);
                if (neighborConnections == null) {
                    neighborConnections = new ArrayList<>();
                    graph.get(currentLevel).put(neighborId, neighborConnections);
                }
                neighborConnections.add(id);

                if (neighborConnections.size() > m) {
                    Vector neighborVector = vectors.get(neighborId);
                    List<SearchResult> neighborNeighbors = new ArrayList<>();

                    for (int nnId : neighborConnections) {
                        Vector nnVector = vectors.get(nnId);
                        if (nnVector != null) {
                            float dist = calculateDistance(neighborVector, nnVector);
                            neighborNeighbors.add(new SearchResult(nnId, dist));
                        }
                    }

                    List<Integer> prunedNeighbors = selectNeighbors(neighborVector, neighborNeighbors, m);
                    graph.get(currentLevel).put(neighborId, new ArrayList<>(prunedNeighbors));
                }
            }

            currentEntryPoint = id;
        }

        if (level > idToLevel.getOrDefault(entryPoint, -1)) {
            entryPoint = id;
        }

        return true;
    }

    /**
     * 添加压缩向量
     */
    private boolean addVectorCompressed(Vector vector) {
        int id = vector.getId();
        int index = currentSize++;

        vectors.put(id, vector);
        idToIndex.put(id, index);

        // PQ编码
        byte[] code = encodeVector(vector);
        System.arraycopy(code, 0, codes[index], 0, pqSubspaces);

        int level = assignLevel();
        idToLevel.put(id, level);

        if (entryPoint == -1) {
            entryPoint = id;
            for (int i = 0; i <= level; i++) {
                graph.get(i).put(id, new ArrayList<>());
            }
            return true;
        }

        // HNSW插入逻辑
        int currentEntryPoint = entryPoint;

        for (int currentLevel = maxLevel - 1; currentLevel > level; currentLevel--) {
            currentEntryPoint = searchLayerClosestCompressed(vector, currentEntryPoint, currentLevel);
        }

        for (int currentLevel = Math.min(level, maxLevel - 1); currentLevel >= 0; currentLevel--) {
            List<SearchResult> neighbors = searchLayerCompressed(vector, currentEntryPoint, efConstruction, currentLevel);
            List<Integer> selectedNeighbors = selectNeighbors(vector, neighbors, m);

            graph.get(currentLevel).put(id, new ArrayList<>(selectedNeighbors));

            for (int neighborId : selectedNeighbors) {
                List<Integer> neighborConnections = graph.get(currentLevel).get(neighborId);
                if (neighborConnections == null) {
                    neighborConnections = new ArrayList<>();
                    graph.get(currentLevel).put(neighborId, neighborConnections);
                }
                neighborConnections.add(id);

                if (neighborConnections.size() > m) {
                    Vector neighborVector = vectors.get(neighborId);
                    List<SearchResult> neighborNeighbors = new ArrayList<>();

                    for (int nnId : neighborConnections) {
                        float dist = calculateDistance(neighborVector, vectors.get(nnId));
                        neighborNeighbors.add(new SearchResult(nnId, dist));
                    }

                    List<Integer> prunedNeighbors = selectNeighbors(neighborVector, neighborNeighbors, m);
                    graph.get(currentLevel).put(neighborId, new ArrayList<>(prunedNeighbors));
                }
            }

            currentEntryPoint = id;
        }

        if (level > idToLevel.getOrDefault(entryPoint, -1)) {
            entryPoint = id;
        }

        return true;
    }

    @Override
    public boolean removeVector(int id) {
        try {
            if (!vectors.containsKey(id)) {
                return false;
            }

            int level = idToLevel.getOrDefault(id, -1);
            if (level == -1) {
                vectors.remove(id);
                return true;
            }

            for (int currentLevel = 0; currentLevel <= level; currentLevel++) {
                Map<Integer, List<Integer>> currentGraph = graph.get(currentLevel);
                if (currentGraph == null) continue;

                List<Integer> neighbors = currentGraph.get(id);
                if (neighbors == null) continue;

                List<Integer> neighborsCopy = new ArrayList<>(neighbors);

                for (int neighborId : neighborsCopy) {
                    List<Integer> neighborConnections = currentGraph.get(neighborId);
                    if (neighborConnections != null) {
                        neighborConnections.remove(Integer.valueOf(id));
                    }
                }

                currentGraph.remove(id);
            }

            if (id == entryPoint) {
                if (vectors.size() == 1) {
                    entryPoint = -1;
                } else {
                    for (int candidateId : vectors.keySet()) {
                        if (candidateId != id) {
                            entryPoint = candidateId;
                            break;
                        }
                    }
                }
            }

            vectors.remove(id);
            idToLevel.remove(id);
            idToIndex.remove(id);
            currentSize--;

            return true;
        } catch (Exception e) {
            log.error("移除向量时发生异常: {}", e.getMessage(), e);
            return false;
        }
    }

    @Override
    public List<SearchResult> searchNearest(Vector queryVector, int k) {
        try {
            if (vectors.isEmpty()) {
                return new ArrayList<>();
            }

            if (entryPoint == -1) {
                if (!vectors.isEmpty()) {
                    entryPoint = vectors.keySet().iterator().next();
                } else {
                    return new ArrayList<>();
                }
            }

            Vector normalizedQuery = normalizeVector(queryVector);
            int searchEf = Math.max(ef, k * 4);

            int currentEntryPoint = entryPoint;

            // 使用PQ距离进行上层搜索
            if (trained) {
                for (int currentLevel = maxLevel - 1; currentLevel > 0; currentLevel--) {
                    currentEntryPoint = searchLayerClosestCompressed(normalizedQuery, currentEntryPoint, currentLevel);
                }

                // 底层使用精确距离
                List<SearchResult> results = searchLayer(normalizedQuery, currentEntryPoint, searchEf, 0);
                return results.size() <= k ? results : results.subList(0, k);
            } else {
                for (int currentLevel = maxLevel - 1; currentLevel > 0; currentLevel--) {
                    currentEntryPoint = searchLayerClosest(normalizedQuery, currentEntryPoint, currentLevel);
                }

                List<SearchResult> results = searchLayer(normalizedQuery, currentEntryPoint, searchEf, 0);
                return results.size() <= k ? results : results.subList(0, k);
            }
        } catch (Exception e) {
            log.error("搜索最近邻时发生异常: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }

    @Override
    public int size() {
        return vectors.size();
    }

    @Override
    public boolean buildIndex() {
        try {
            long startTime = System.currentTimeMillis();
            log.info("开始重建HNSW+PQ索引...");

            // 如果还没训练，先训练
            if (!trained && !trainingSamples.isEmpty()) {
                train();
            }

            Map<Integer, Vector> savedVectors = new HashMap<>(vectors);

            vectors.clear();
            idToLevel.clear();
            idToIndex.clear();
            currentSize = 0;

            for (int i = 0; i < maxLevel; i++) {
                graph.get(i).clear();
            }

            entryPoint = -1;

            List<Vector> vectorList = new ArrayList<>(savedVectors.values());
            vectorList.sort(Comparator.comparing(Vector::getId));

            int successCount = 0;
            for (Vector vector : vectorList) {
                if (addVector(vector)) {
                    successCount++;
                }
            }

            long endTime = System.currentTimeMillis();
            log.info("HNSW+PQ索引重建完成，包含 {} 个向量，耗时: {} ms", successCount, (endTime - startTime));

            return true;
        } catch (Exception e) {
            log.error("重建索引时发生异常: {}", e.getMessage(), e);
            return false;
        }
    }

    // ========== 辅助方法 ==========

    private Vector normalizeVector(Vector vector) {
        if (normalizeVectors) {
            return vector.normalize();
        }
        return vector;
    }

    private float calculateDistance(Vector v1, Vector v2) {
        if (useCosineSimilarity) {
            return 1.0f - v1.cosineSimilarity(v2);
        } else {
            return v1.euclideanDistance(v2);
        }
    }

    private int assignLevel() {
        int level = 0;
        double probability = 1.0 / Math.E;
        while (ThreadLocalRandom.current().nextDouble() < probability && level < maxLevel - 1) {
            level++;
        }
        return level;
    }

    private int searchLayerClosest(Vector queryVector, int entryPointId, int level) {
        if (!vectors.containsKey(entryPointId)) {
            return entryPointId;
        }

        int currentId = entryPointId;
        float currentDistance = calculateDistance(vectors.get(currentId), queryVector);

        boolean changed;
        do {
            changed = false;
            Map<Integer, List<Integer>> layerGraph = graph.get(level);
            if (layerGraph == null) return currentId;

            List<Integer> neighbors = layerGraph.get(currentId);
            if (neighbors == null || neighbors.isEmpty()) return currentId;

            for (int neighborId : neighbors) {
                if (!vectors.containsKey(neighborId)) continue;

                float distance = calculateDistance(vectors.get(neighborId), queryVector);
                if (distance < currentDistance) {
                    currentDistance = distance;
                    currentId = neighborId;
                    changed = true;
                    break;
                }
            }
        } while (changed);

        return currentId;
    }

    private int searchLayerClosestCompressed(Vector queryVector, int entryPointId, int level) {
        if (!vectors.containsKey(entryPointId)) {
            return entryPointId;
        }

        int currentId = entryPointId;
        int currentIdx = idToIndex.get(currentId);
        float currentDistance = computePQDistance(queryVector, currentIdx);

        boolean changed;
        do {
            changed = false;
            Map<Integer, List<Integer>> layerGraph = graph.get(level);
            if (layerGraph == null) return currentId;

            List<Integer> neighbors = layerGraph.get(currentId);
            if (neighbors == null || neighbors.isEmpty()) return currentId;

            for (int neighborId : neighbors) {
                if (!vectors.containsKey(neighborId)) continue;

                int neighborIdx = idToIndex.get(neighborId);
                float distance = computePQDistance(queryVector, neighborIdx);
                if (distance < currentDistance) {
                    currentDistance = distance;
                    currentId = neighborId;
                    changed = true;
                    break;
                }
            }
        } while (changed);

        return currentId;
    }

    private List<SearchResult> searchLayer(Vector queryVector, int entryPointId, int ef, int level) {
        try {
            if (!vectors.containsKey(entryPointId)) {
                return new ArrayList<>();
            }

            PriorityQueue<SearchResult> resultSet = new PriorityQueue<>(
                    Comparator.comparing(SearchResult::getDistance).reversed());
            PriorityQueue<SearchResult> candidateSet = new PriorityQueue<>(
                    Comparator.comparing(SearchResult::getDistance));

            float distance = calculateDistance(vectors.get(entryPointId), queryVector);
            SearchResult entryResult = new SearchResult(entryPointId, distance);
            resultSet.add(entryResult);
            candidateSet.add(entryResult);

            Set<Integer> visited = new HashSet<>();
            visited.add(entryPointId);

            float furthestDistance = distance;

            while (!candidateSet.isEmpty()) {
                SearchResult current = candidateSet.poll();
                if (current == null) continue;

                if (current.getDistance() > furthestDistance) {
                    break;
                }

                Map<Integer, List<Integer>> layerGraph = graph.get(level);
                if (layerGraph == null) continue;

                List<Integer> neighbors = layerGraph.get(current.getId());
                if (neighbors == null || neighbors.isEmpty()) continue;

                for (int neighborId : neighbors) {
                    if (visited.contains(neighborId) || !vectors.containsKey(neighborId)) continue;

                    visited.add(neighborId);

                    Vector neighborVector = vectors.get(neighborId);
                    if (neighborVector == null) continue;

                    float neighborDistance = calculateDistance(neighborVector, queryVector);
                    SearchResult neighborResult = new SearchResult(neighborId, neighborDistance);

                    if (resultSet.size() < ef || neighborDistance < furthestDistance) {
                        candidateSet.add(neighborResult);
                        resultSet.add(neighborResult);

                        if (resultSet.size() > ef) {
                            resultSet.poll();
                            furthestDistance = resultSet.peek().getDistance();
                        }
                    }
                }
            }

            List<SearchResult> results = new ArrayList<>(resultSet);
            results.sort(Comparator.comparing(SearchResult::getDistance));
            return results;
        } catch (Exception e) {
            log.error("搜索层时发生异常: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }

    private List<SearchResult> searchLayerCompressed(Vector queryVector, int entryPointId, int ef, int level) {
        try {
            if (!vectors.containsKey(entryPointId)) {
                return new ArrayList<>();
            }

            PriorityQueue<SearchResult> resultSet = new PriorityQueue<>(
                    Comparator.comparing(SearchResult::getDistance).reversed());
            PriorityQueue<SearchResult> candidateSet = new PriorityQueue<>(
                    Comparator.comparing(SearchResult::getDistance));

            int entryIdx = idToIndex.get(entryPointId);
            float distance = computePQDistance(queryVector, entryIdx);
            SearchResult entryResult = new SearchResult(entryPointId, distance);
            resultSet.add(entryResult);
            candidateSet.add(entryResult);

            Set<Integer> visited = new HashSet<>();
            visited.add(entryPointId);

            float furthestDistance = distance;

            while (!candidateSet.isEmpty()) {
                SearchResult current = candidateSet.poll();
                if (current == null) continue;

                if (current.getDistance() > furthestDistance) {
                    break;
                }

                Map<Integer, List<Integer>> layerGraph = graph.get(level);
                if (layerGraph == null) continue;

                List<Integer> neighbors = layerGraph.get(current.getId());
                if (neighbors == null || neighbors.isEmpty()) continue;

                for (int neighborId : neighbors) {
                    if (visited.contains(neighborId) || !vectors.containsKey(neighborId)) continue;

                    visited.add(neighborId);

                    int neighborIdx = idToIndex.get(neighborId);
                    float neighborDistance = computePQDistance(queryVector, neighborIdx);
                    SearchResult neighborResult = new SearchResult(neighborId, neighborDistance);

                    if (resultSet.size() < ef || neighborDistance < furthestDistance) {
                        candidateSet.add(neighborResult);
                        resultSet.add(neighborResult);

                        if (resultSet.size() > ef) {
                            resultSet.poll();
                            furthestDistance = resultSet.peek().getDistance();
                        }
                    }
                }
            }

            List<SearchResult> results = new ArrayList<>(resultSet);
            results.sort(Comparator.comparing(SearchResult::getDistance));
            return results;
        } catch (Exception e) {
            log.error("搜索层(压缩)时发生异常: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }

    private List<Integer> selectNeighbors(Vector vector, List<SearchResult> candidates, int m) {
        try {
            if (candidates == null || candidates.isEmpty()) {
                return new ArrayList<>();
            }

            if (candidates.size() <= m) {
                return candidates.stream()
                        .map(SearchResult::getId)
                        .collect(Collectors.toList());
            }

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
     * 获取索引统计信息
     */
    public String getIndexStats() {
        StringBuilder stats = new StringBuilder();
        stats.append("HNSW+PQ索引统计信息:\n");
        stats.append("- 向量数量: ").append(vectors.size()).append("\n");
        stats.append("- 向量维度: ").append(dimension).append("\n");
        stats.append("- PQ子空间数: ").append(pqSubspaces).append("\n");
        stats.append("- 子空间维度: ").append(subDim).append("\n");
        stats.append("- 聚类中心数: ").append(nCentroids).append("\n");
        stats.append("- 是否已训练: ").append(trained).append("\n");
        stats.append("- 压缩比: ").append(String.format("%.2fx", getCompressionRatio())).append("\n");
        stats.append("- 内存节省: ").append(String.format("%.1f%%", getMemorySavings())).append("\n");
        stats.append("- M值: ").append(m).append("\n");
        stats.append("- 最大层级: ").append(maxLevel).append("\n");
        return stats.toString();
    }
}
