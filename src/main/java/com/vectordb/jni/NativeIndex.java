package com.vectordb.jni;

import com.vectordb.core.SearchResult;
import com.vectordb.core.Vector;
import com.vectordb.index.VectorIndex;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Native索引基类
 * 提供JNI调用的统一接口
 */
public abstract class NativeIndex implements VectorIndex {
    protected final long nativeHandle;
    protected final int dimension;

    static {
        if (!NativeLoader.load()) {
            throw new RuntimeException("Failed to load native library");
        }
    }

    protected NativeIndex(int dimension, long nativeHandle) {
        this.dimension = dimension;
        this.nativeHandle = nativeHandle;
    }

    @Override
    public boolean addVector(Vector vector) {
        if (vector == null) {
            return false;
        }
        float[] data = vector.getValues();
        if (data.length != dimension) {
            throw new IllegalArgumentException(
                "Vector dimension mismatch: expected " + dimension + ", got " + data.length);
        }
        nativeAdd(nativeHandle, vector.getId(), data);
        return true;
    }

    @Override
    public boolean removeVector(int id) {
        // Native implementation may not support direct removal
        return false;
    }

    @Override
    public List<SearchResult> searchNearest(Vector queryVector, int k) {
        if (queryVector == null) {
            return new ArrayList<>();
        }
        float[] data = queryVector.getValues();
        if (data.length != dimension) {
            throw new IllegalArgumentException(
                "Query dimension mismatch: expected " + dimension + ", got " + data.length);
        }
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }

        int[] ids = new int[k];
        float[] distances = new float[k];

        int count = nativeSearch(nativeHandle, data, k, ids, distances);

        List<SearchResult> results = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            results.add(new SearchResult(ids[i], distances[i]));
        }
        return results;
    }

    @Override
    public int size() {
        return -1;
    }

    @Override
    public boolean buildIndex() {
        return true;
    }

    /**
     * 添加向量（原始数组接口）
     * @param id 向量ID
     * @param vector 向量数据
     */
    public void addVector(int id, float[] vector) {
        if (vector.length != dimension) {
            throw new IllegalArgumentException(
                "Vector dimension mismatch: expected " + dimension + ", got " + vector.length);
        }
        nativeAdd(nativeHandle, id, vector);
    }

    /**
     * 搜索（原始数组接口）
     * @param query 查询向量
     * @param k 返回结果数量
     * @return 搜索结果列表
     */
    public List<SearchResult> search(float[] query, int k) {
        if (query.length != dimension) {
            throw new IllegalArgumentException(
                "Query dimension mismatch: expected " + dimension + ", got " + query.length);
        }
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }

        int[] ids = new int[k];
        float[] distances = new float[k];

        int count = nativeSearch(nativeHandle, query, k, ids, distances);

        List<SearchResult> results = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            results.add(new SearchResult(ids[i], distances[i]));
        }
        return results;
    }

    /**
     * 关闭索引，释放Native资源
     */
    public void close() {
        nativeDestroy(nativeHandle);
    }

    // Native方法声明
    protected native void nativeAdd(long handle, int id, float[] vector);
    protected native int nativeSearch(long handle, float[] query, int k, int[] resultIds, float[] resultDistances);
    protected native void nativeDestroy(long handle);

    // 批量操作（使用DirectByteBuffer实现零拷贝）
    protected native void nativeAddBatch(long handle, ByteBuffer ids, ByteBuffer vectors,
                                         int count, int dimension);
    protected native int nativeSearchBatch(long handle, ByteBuffer queries, int nQueries,
                                           int k, int dimension, ByteBuffer resultIds,
                                           ByteBuffer resultDistances);

    /**
     * 批量添加向量（零拷贝优化）
     * @param ids 向量ID数组
     * @param vectors 向量数据数组 [count][dimension]
     */
    public void addBatch(int[] ids, float[][] vectors) {
        if (ids.length != vectors.length) {
            throw new IllegalArgumentException("ids and vectors must have same length");
        }

        int count = ids.length;

        ByteBuffer idsBuffer = ByteBuffer.allocateDirect(count * 4)
            .order(ByteOrder.nativeOrder());
        ByteBuffer vectorsBuffer = ByteBuffer.allocateDirect(count * dimension * 4)
            .order(ByteOrder.nativeOrder());

        IntBuffer intBuffer = idsBuffer.asIntBuffer();
        FloatBuffer floatBuffer = vectorsBuffer.asFloatBuffer();

        for (int i = 0; i < count; i++) {
            if (vectors[i].length != dimension) {
                throw new IllegalArgumentException(
                    "Vector " + i + " dimension mismatch: expected " + dimension +
                    ", got " + vectors[i].length);
            }
            intBuffer.put(ids[i]);
            floatBuffer.put(vectors[i]);
        }

        nativeAddBatch(nativeHandle, idsBuffer, vectorsBuffer, count, dimension);
    }

    /**
     * 批量搜索（零拷贝优化）
     * @param queries 查询向量数组 [nQueries][dimension]
     * @param k 每个查询返回的结果数
     * @return 每个查询的结果列表
     */
    public List<List<SearchResult>> searchBatch(float[][] queries, int k) {
        int nQueries = queries.length;

        for (int i = 0; i < nQueries; i++) {
            if (queries[i].length != dimension) {
                throw new IllegalArgumentException(
                    "Query " + i + " dimension mismatch: expected " + dimension +
                    ", got " + queries[i].length);
            }
        }

        ByteBuffer queriesBuffer = ByteBuffer.allocateDirect(nQueries * dimension * 4)
            .order(ByteOrder.nativeOrder());
        ByteBuffer resultIdsBuffer = ByteBuffer.allocateDirect(nQueries * k * 4)
            .order(ByteOrder.nativeOrder());
        ByteBuffer resultDistsBuffer = ByteBuffer.allocateDirect(nQueries * k * 4)
            .order(ByteOrder.nativeOrder());

        FloatBuffer floatBuffer = queriesBuffer.asFloatBuffer();
        for (float[] query : queries) {
            floatBuffer.put(query);
        }

        nativeSearchBatch(nativeHandle, queriesBuffer, nQueries, k, dimension,
                         resultIdsBuffer, resultDistsBuffer);

        List<List<SearchResult>> results = new ArrayList<>(nQueries);
        IntBuffer idsBuffer = resultIdsBuffer.asIntBuffer();
        FloatBuffer distsBuffer = resultDistsBuffer.asFloatBuffer();

        for (int i = 0; i < nQueries; i++) {
            List<SearchResult> queryResults = new ArrayList<>(k);
            for (int j = 0; j < k; j++) {
                int id = idsBuffer.get(i * k + j);
                float dist = distsBuffer.get(i * k + j);
                if (id >= 0) {
                    queryResults.add(new SearchResult(id, dist));
                }
            }
            results.add(queryResults);
        }

        return results;
    }
}
