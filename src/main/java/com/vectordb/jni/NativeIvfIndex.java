package com.vectordb.jni;

/**
 * IVF索引的Native实现
 */
public class NativeIvfIndex extends NativeIndex {

    public NativeIvfIndex(int dimension, int maxElements, int nLists, int nProbes) {
        super(dimension, nativeCreateIVF(dimension, maxElements, nLists, nProbes));
    }

    public NativeIvfIndex(int dimension, int maxElements) {
        this(dimension, maxElements, 100, 10);
    }

    /**
     * 训练聚类中心
     */
    public void train(int nSamples, float[] samples) {
        nativeTrain(nativeHandle, nSamples, samples);
    }

    // Native方法
    private static native long nativeCreateIVF(int dimension, int maxElements, int nLists, int nProbes);
    private native void nativeTrain(long handle, int nSamples, float[] samples);
}
