#include <jni.h>
#include "index/HNSWIndex.h"
#include "index/PQIndex.h"
#include "index/IVFIndex.h"
#include "index/LSHIndex.h"
#include "index/AnnoyIndex.h"
#include <unordered_map>
#include <memory>
#include <mutex>

using namespace vectordb;

static std::unordered_map<jlong, std::shared_ptr<VectorIndex>> g_indices;
static std::mutex g_mutex;
static jlong g_nextHandle = 1;

static jlong registerIndex(std::shared_ptr<VectorIndex> index) {
    std::lock_guard<std::mutex> lock(g_mutex);
    jlong handle = g_nextHandle++;
    g_indices[handle] = index;
    return handle;
}

static std::shared_ptr<VectorIndex> getIndex(jlong handle) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_indices.find(handle);
    if (it != g_indices.end()) {
        return it->second;
    }
    return nullptr;
}

static void unregisterIndex(jlong handle) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_indices.erase(handle);
}

// HNSW Index
JNIEXPORT jlong JNICALL Java_com_vectordb_jni_NativeHnswIndex_nativeCreateHNSW
  (JNIEnv *env, jclass clazz, jint dimension, jint maxElements, jint M, jint efConstruction, jint ef) {
    try {
        HNSWConfig config;
        config.M = M;
        config.efConstruction = efConstruction;
        config.efSearch = ef;
        auto index = std::make_shared<HNSWIndex>(dimension, maxElements, config);
        return registerIndex(index);
    } catch (...) {
        return 0;
    }
}

// PQ Index
JNIEXPORT jlong JNICALL Java_com_vectordb_jni_NativePqIndex_nativeCreatePQ
  (JNIEnv *env, jclass clazz, jint dimension, jint maxElements, jint M, jint nBits) {
    try {
        PQConfig config;
        config.M = M;
        config.nBits = nBits;
        auto index = std::make_shared<PQIndex>(dimension, maxElements, config);
        return registerIndex(index);
    } catch (...) {
        return 0;
    }
}

JNIEXPORT void JNICALL Java_com_vectordb_jni_NativePqIndex_nativeTrain
  (JNIEnv *env, jobject obj, jlong handle, jint nSamples, jfloatArray samples) {
    auto index = std::dynamic_pointer_cast<PQIndex>(getIndex(handle));
    if (index) {
        jfloat* samplesData = env->GetFloatArrayElements(samples, nullptr);
        index->train(nSamples, samplesData);
        env->ReleaseFloatArrayElements(samples, samplesData, JNI_ABORT);
    }
}

// IVF Index
JNIEXPORT jlong JNICALL Java_com_vectordb_jni_NativeIvfIndex_nativeCreateIVF
  (JNIEnv *env, jclass clazz, jint dimension, jint maxElements, jint nLists, jint nProbes) {
    try {
        IVFConfig config;
        config.nLists = nLists;
        config.nProbes = nProbes;
        auto index = std::make_shared<IVFIndex>(dimension, maxElements, config);
        return registerIndex(index);
    } catch (...) {
        return 0;
    }
}

JNIEXPORT void JNICALL Java_com_vectordb_jni_NativeIvfIndex_nativeTrain
  (JNIEnv *env, jobject obj, jlong handle, jint nSamples, jfloatArray samples) {
    auto index = std::dynamic_pointer_cast<IVFIndex>(getIndex(handle));
    if (index) {
        jfloat* samplesData = env->GetFloatArrayElements(samples, nullptr);
        index->train(nSamples, samplesData);
        env->ReleaseFloatArrayElements(samples, samplesData, JNI_ABORT);
    }
}

// LSH Index
JNIEXPORT jlong JNICALL Java_com_vectordb_jni_NativeLshIndex_nativeCreateLSH
  (JNIEnv *env, jclass clazz, jint dimension, jint maxElements, jint numHashTables, jint numHashFunctions) {
    try {
        auto index = std::make_shared<LSHIndex>(dimension, maxElements, numHashTables, numHashFunctions);
        return registerIndex(index);
    } catch (...) {
        return 0;
    }
}

// Annoy Index
JNIEXPORT jlong JNICALL Java_com_vectordb_jni_NativeAnnoyIndex_nativeCreateAnnoy
  (JNIEnv *env, jclass clazz, jint dimension, jint maxElements, jint numTrees) {
    try {
        auto index = std::make_shared<AnnoyIndex>(dimension, maxElements, numTrees);
        return registerIndex(index);
    } catch (...) {
        return 0;
    }
}

JNIEXPORT void JNICALL Java_com_vectordb_jni_NativeAnnoyIndex_nativeBuild
  (JNIEnv *env, jobject obj, jlong handle) {
    auto index = std::dynamic_pointer_cast<AnnoyIndex>(getIndex(handle));
    if (index) {
        index->build();
    }
}

// Common methods
JNIEXPORT void JNICALL Java_com_vectordb_jni_NativeIndex_nativeAdd
  (JNIEnv *env, jobject obj, jlong handle, jint id, jfloatArray vector) {
    auto index = getIndex(handle);
    if (index) {
        jfloat* vecData = env->GetFloatArrayElements(vector, nullptr);
        index->add(id, vecData);
        env->ReleaseFloatArrayElements(vector, vecData, JNI_ABORT);
    }
}

JNIEXPORT jint JNICALL Java_com_vectordb_jni_NativeIndex_nativeSearch
  (JNIEnv *env, jobject obj, jlong handle, jfloatArray query, jint k, jintArray resultIds, jfloatArray resultDistances) {
    auto index = getIndex(handle);
    if (!index) return 0;

    jfloat* queryData = env->GetFloatArrayElements(query, nullptr);
    jint* idsData = env->GetIntArrayElements(resultIds, nullptr);
    jfloat* distsData = env->GetFloatArrayElements(resultDistances, nullptr);

    int count = 0;
    index->search(queryData, k, idsData, distsData, &count);

    env->ReleaseFloatArrayElements(query, queryData, JNI_ABORT);
    env->ReleaseIntArrayElements(resultIds, idsData, 0);
    env->ReleaseFloatArrayElements(resultDistances, distsData, 0);

    return count;
}

JNIEXPORT void JNICALL Java_com_vectordb_jni_NativeIndex_nativeDestroy
  (JNIEnv *env, jobject obj, jlong handle) {
    unregisterIndex(handle);
}

// Batch operations
JNIEXPORT void JNICALL Java_com_vectordb_jni_NativeIndex_nativeAddBatch
  (JNIEnv *env, jobject obj, jlong handle, jobject idsBuffer, jobject vectorsBuffer, jint count, jint dimension) {
    auto index = getIndex(handle);
    if (!index) return;

    jint* idsData = static_cast<jint*>(env->GetDirectBufferAddress(idsBuffer));
    jfloat* vectorsData = static_cast<jfloat*>(env->GetDirectBufferAddress(vectorsBuffer));

    if (auto hnsw = std::dynamic_pointer_cast<HNSWIndex>(index)) {
        hnsw->addBatch(vectorsData, idsData, count);
    } else if (auto pq = std::dynamic_pointer_cast<PQIndex>(index)) {
        pq->addBatch(vectorsData, idsData, count);
    } else if (auto ivf = std::dynamic_pointer_cast<IVFIndex>(index)) {
        ivf->addBatch(vectorsData, idsData, count);
    }
}

JNIEXPORT jint JNICALL Java_com_vectordb_jni_NativeIndex_nativeSearchBatch
  (JNIEnv *env, jobject obj, jlong handle, jobject queriesBuffer, jint nQueries, jint k, jint dimension, jobject resultIdsBuffer, jobject resultDistancesBuffer) {
    auto index = getIndex(handle);
    if (!index) return 0;

    jfloat* queriesData = static_cast<jfloat*>(env->GetDirectBufferAddress(queriesBuffer));
    jint* resultIdsData = static_cast<jint*>(env->GetDirectBufferAddress(resultIdsBuffer));
    jfloat* resultDistsData = static_cast<jfloat*>(env->GetDirectBufferAddress(resultDistancesBuffer));

    if (auto hnsw = std::dynamic_pointer_cast<HNSWIndex>(index)) {
        hnsw->searchBatch(queriesData, nQueries, k, resultIdsData, resultDistsData);
    } else if (auto pq = std::dynamic_pointer_cast<PQIndex>(index)) {
        pq->searchBatch(queriesData, nQueries, k, resultIdsData, resultDistsData);
    }

    return nQueries;
}
