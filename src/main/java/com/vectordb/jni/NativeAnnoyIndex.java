package com.vectordb.jni;

/**
 * Annoy索引的Native实现
 */
public class NativeAnnoyIndex extends NativeIndex {

    public NativeAnnoyIndex(int dimension, int maxElements, int numTrees) {
        super(dimension, nativeCreateAnnoy(dimension, maxElements, numTrees));
    }

    public NativeAnnoyIndex(int dimension, int maxElements) {
        this(dimension, maxElements, 10);
    }

    /**
     * 构建索引
     */
    public void build() {
        nativeBuild(nativeHandle);
    }

    // Native方法
    private static native long nativeCreateAnnoy(int dimension, int maxElements, int numTrees);
    private native void nativeBuild(long handle);
}
