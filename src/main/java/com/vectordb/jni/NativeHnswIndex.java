package com.vectordb.jni;

/**
 * HNSW索引的Native实现
 */
public class NativeHnswIndex extends NativeIndex {

    /**
     * 创建HNSW索引
     */
    public NativeHnswIndex(int dimension, int maxElements, int M, int efConstruction, int ef) {
        super(dimension, nativeCreateHNSW(dimension, maxElements, M, efConstruction, ef));
    }

    /**
     * 使用默认参数创建HNSW索引
     */
    public NativeHnswIndex(int dimension, int maxElements) {
        this(dimension, maxElements, 32, 64, 64);
    }

    // Native方法
    private static native long nativeCreateHNSW(int dimension, int maxElements, int M, int efConstruction, int ef);
}
