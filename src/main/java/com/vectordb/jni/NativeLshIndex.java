package com.vectordb.jni;

/**
 * LSH索引的Native实现
 */
public class NativeLshIndex extends NativeIndex {

    public NativeLshIndex(int dimension, int maxElements, int numHashTables, int numHashFunctions) {
        super(dimension, nativeCreateLSH(dimension, maxElements, numHashTables, numHashFunctions));
    }

    public NativeLshIndex(int dimension, int maxElements) {
        this(dimension, maxElements, 10, 20);
    }

    // Native方法
    private static native long nativeCreateLSH(int dimension, int maxElements, int numHashTables, int numHashFunctions);
}
