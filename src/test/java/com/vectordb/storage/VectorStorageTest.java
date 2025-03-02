package com.vectordb.storage;

import com.vectordb.core.Vector;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.util.Optional;

/**
 * VectorStorage类的单元测试
 */
public class VectorStorageTest {
    
    private VectorStorage storage;
    private static final String TEST_STORAGE_PATH = "test_vector_storage";
    private static final int DIMENSION = 10;
    private static final int MAX_ELEMENTS=10_0000;
    
    @Before
    public void setUp() throws IOException {
        // 创建测试存储目录
        File storageDir = new File(TEST_STORAGE_PATH);
        if (!storageDir.exists()) {
            storageDir.mkdirs();
        }
        
        // 初始化存储
        storage = new VectorStorage(TEST_STORAGE_PATH,DIMENSION,MAX_ELEMENTS);
    }
    
    @After
    public void tearDown() throws IOException {
        // 关闭存储
        if (storage != null) {
            storage.close();
        }
        
        // 清理测试存储目录
        deleteDirectory(new File(TEST_STORAGE_PATH));
    }
    
    /**
     * 测试保存和获取向量
     */
    @Test
    public void testSaveAndGetVector() throws IOException {
        // 创建测试向量
        float[] values = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            values[i] = i * 0.1f;
        }
        Vector vector = new Vector(1, values);
        
        // 保存向量
        assertTrue(storage.saveVector(vector));
        
        // 获取向量
        Optional<Vector> retrieved = storage.getVector(1);
        
        // 验证向量
        assertNotNull(retrieved);
        assertEquals(vector.getId(), retrieved.get().getId());
        assertEquals(vector.getDimension(), retrieved.get().getDimension());
        
        // 验证向量值
        for (int i = 0; i < DIMENSION; i++) {
            assertEquals(vector.getValues()[i], retrieved.get().getValues()[i], 0.0001f);
        }
    }
    
    /**
     * 测试删除向量
     */
    @Test
    public void testDeleteVector() throws IOException {
        // 创建测试向量
        float[] values = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            values[i] = i * 0.1f;
        }
        Vector vector = new Vector(1, values);
        
        // 保存向量
        assertTrue(storage.saveVector(vector));
        
        // 验证向量存在
        assertNotNull(storage.getVector(1));
        
        // 删除向量
        assertTrue(storage.deleteVector(1));
        
        // 验证向量不存在
        assertNull(storage.getVector(1));
        
        // 删除不存在的向量应该返回false
        assertFalse(storage.deleteVector(2));
    }
    
    /**
     * 测试加载所有向量
     */
    @Test
    public void testLoadAllVectors() throws IOException {
        // 创建多个测试向量
        int numVectors = 10;
        for (int id = 1; id <= numVectors; id++) {
            float[] values = new float[DIMENSION];
            for (int i = 0; i < DIMENSION; i++) {
                values[i] = id * i * 0.1f;
            }
            Vector vector = new Vector(id, values);
            storage.saveVector(vector);
        }
        
        // 加载所有向量
        List<Vector> vectors = storage.loadVectors();
        
        // 验证向量数量
        assertEquals(numVectors, vectors.size());
        
        // 验证向量ID
        List<Integer> ids = new ArrayList<>();
        for (Vector v : vectors) {
            ids.add(v.getId());
        }
        
        for (int id = 1; id <= numVectors; id++) {
            assertTrue(ids.contains(id));
        }
    }
    
    /**
     * 测试持久化和重新加载
     */
    @Test
    public void testPersistenceAndReload() throws IOException {
        // 创建测试向量
        float[] values = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            values[i] = i * 0.1f;
        }
        Vector vector = new Vector(1, values);
        
        // 保存向量
        assertTrue(storage.saveVector(vector));
        
        // 关闭存储
        storage.close();
        
        // 重新打开存储
        storage = new VectorStorage(TEST_STORAGE_PATH,DIMENSION,MAX_ELEMENTS);
        
        // 获取向量
        Optional<Vector> retrieved = storage.getVector(1);
        
        // 验证向量
        assertNotNull(retrieved.get());
        assertEquals(vector.getId(), retrieved.get().getId());
        assertEquals(vector.getDimension(), retrieved.get().getDimension());
        
        // 验证向量值
        for (int i = 0; i < DIMENSION; i++) {
            assertEquals(vector.getValues()[i], retrieved.get().getValues()[i], 0.0001f);
        }
    }
    
    /**
     * 递归删除目录
     */
    private void deleteDirectory(File directory) {
        if (directory.exists()) {
            File[] files = directory.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isDirectory()) {
                        deleteDirectory(file);
                    } else {
                        file.delete();
                    }
                }
            }
            directory.delete();
        }
    }
} 