package com.vectordb.storage;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.vectordb.core.Vector;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import lombok.extern.slf4j.Slf4j;

/**
 * 向量存储类，负责向量的持久化存储
 * 所有向量存储在一个文件中，避免小文件过多
 */
@Slf4j
public class VectorStorage implements Closeable {
    private final String storagePath;
    private final int dimension;
    private final int maxElements;
    private final Map<Integer, Vector> cache; // 内存缓存
    private final ObjectMapper objectMapper;
    private final ReadWriteLock lock; // 用于同步文件访问
    private final Path vectorsFilePath; // 存储所有向量的文件路径
    private boolean isDirty = false; // 标记是否有未保存的更改
    
    /**
     * 创建向量存储
     * 
     * @param storagePath 存储路径
     * @param dimension 向量维度
     * @param maxElements 最大元素数量
     */
    public VectorStorage(String storagePath, int dimension, int maxElements) {
        this.storagePath = storagePath;
        this.dimension = dimension;
        this.maxElements = maxElements;
        this.cache = new ConcurrentHashMap<>(maxElements);
        this.objectMapper = new ObjectMapper();
        this.lock = new ReentrantReadWriteLock();
        this.vectorsFilePath = Paths.get(storagePath, "vectors.json");
        
        // 创建存储目录
        try {
            Path path = Paths.get(storagePath);
            Files.createDirectories(path);
        } catch (IOException e) {
            throw new RuntimeException("创建存储目录失败: " + storagePath, e);
        }
    }
    
    /**
     * 保存向量
     * 
     * @param vector 要保存的向量
     * @return 是否保存成功
     */
    public boolean saveVector(Vector vector) throws IOException {
        int id = vector.getId();
        
        // 检查维度
        if (vector.getDimension() != dimension) {
            throw new IllegalArgumentException("向量维度不匹配，期望: " + dimension + ", 实际: " + vector.getDimension());
        }
        
        // 保存到缓存
        cache.put(id, vector);
        
        // 标记为脏，需要保存到文件
        isDirty = true;
        
        // 定期保存到文件（可以根据需要调整保存策略）
        if (cache.size() % 1000 == 0) {
            saveToFile();
        }
        
        return true;
    }
    
    /**
     * 获取向量
     * 
     * @param id 向量ID
     * @return 向量（如果存在）
     */
    public Optional<Vector> getVector(int id) throws IOException {
        // 先从缓存获取
        if (cache.containsKey(id)) {
            return Optional.of(cache.get(id));
        }
        
        // 如果缓存中没有，尝试从文件加载所有向量
        if (cache.isEmpty()) {
            loadVectors();
            
            // 再次检查缓存
            if (cache.containsKey(id)) {
                return Optional.of(cache.get(id));
            }
        }
        
        return Optional.empty();
    }
    
    /**
     * 删除向量
     * 
     * @param id 向量ID
     * @return 是否删除成功
     */
    public boolean deleteVector(int id) throws IOException {
        // 从缓存删除
        Vector removed = cache.remove(id);
        
        if (removed != null) {
            // 标记为脏，需要保存到文件
            isDirty = true;
            
            // 定期保存到文件
            if (cache.size() % 1000 == 0) {
                saveToFile();
            }
            
            return true;
        }
        
        return false;
    }
    
    /**
     * 加载所有向量
     * 
     * @return 向量列表
     */
    public List<Vector> loadVectors() throws IOException {
        lock.readLock().lock();
        try {
            // 如果文件不存在，返回空列表
            if (!Files.exists(vectorsFilePath)) {
                return new ArrayList<>();
            }
            
            // 从文件加载所有向量
            try {
                Map<Integer, Vector> loadedVectors = objectMapper.readValue(
                    vectorsFilePath.toFile(), 
                    new TypeReference<Map<Integer, Vector>>() {}
                );
                
                // 更新缓存
                cache.clear();
                cache.putAll(loadedVectors);
                
                return new ArrayList<>(loadedVectors.values());
            } catch (IOException e) {
                log.error("读取向量文件失败: {}, 错误: {}", vectorsFilePath, e.getMessage(), e);
                // 如果文件损坏或格式不正确，返回空列表
                return new ArrayList<>();
            }
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * 将缓存中的所有向量保存到文件
     */
    private void saveToFile() throws IOException {
        // 如果没有更改，不需要保存
        if (!isDirty) {
            return;
        }
        
        lock.writeLock().lock();
        try {
            // 将缓存中的所有向量写入文件
            objectMapper.writeValue(vectorsFilePath.toFile(), cache);
            
            // 重置脏标记
            isDirty = false;
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * 关闭存储，释放资源
     */
    @Override
    public void close() throws IOException {
        // 保存所有未保存的更改
        if (isDirty) {
            saveToFile();
        }
        
        // 清空缓存
        cache.clear();
    }
} 