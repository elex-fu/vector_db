package com.vectordb.jni;

/**
 * PQ索引的Native实现
 */
public class NativePqIndex extends NativeIndex {

    public NativePqIndex(int dimension, int maxElements, int M, int nBits) {
        super(dimension, nativeCreatePQ(dimension, maxElements, M, nBits));
    }

    public NativePqIndex(int dimension, int maxElements) {
        this(dimension, maxElements, 8, 8);
    }

    /**
     * 训练码本
     */
    public void train(int nSamples, float[] samples) {
        nativeTrain(nativeHandle, nSamples, samples);
    }

    // Native方法
    private static native long nativeCreatePQ(int dimension, int maxElements, int M, int nBits);
    private native void nativeTrain(long handle, int nSamples, float[] samples);
}
