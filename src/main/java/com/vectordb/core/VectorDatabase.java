package com.vectordb.core;

import com.vectordb.index.AnnoyIndex;
import com.vectordb.index.HnswIndex;
import com.vectordb.index.IvfIndex;
import com.vectordb.index.LshIndex;
import com.vectordb.index.PqIndex;
import com.vectordb.index.VectorIndex;
import com.vectordb.storage.VectorStorage;

import java.io.Closeable;
import java.io.IOException;
import java.util.List;
import java.util.Optional;
import lombok.extern.slf4j.Slf4j;

/**
 * 向量数据库接口，定义基本操作
 */
@Slf4j
public class VectorDatabase implements Closeable {
    // 索引类型枚举
    public enum IndexType {
        HNSW,   // 层次可导航小世界图索引
        ANNOY,  // 近似最近邻哦耶索引
        LSH,    // 局部敏感哈希索引
        IVF,    // 倒排文件索引
        PQ      // 乘积量化索引
    }
    
    private final int dimension;
    private final int maxElements;
    private final String storagePath;
    private final VectorIndex index;
    private final VectorStorage storage;
    
    /**
     * 使用Builder模式创建VectorDatabase实例
     */
    private VectorDatabase(Builder builder) {
        this.dimension = builder.dimension;
        this.maxElements = builder.maxElements;
        this.storagePath = builder.storagePath;
        
        // 初始化存储
        this.storage = builder.storage != null ? 
                builder.storage : 
                new VectorStorage(storagePath, dimension, maxElements);
        
        // 根据索引类型初始化索引
        if (builder.index != null) {
            this.index = builder.index;
        } else {
            switch (builder.indexType) {
                case ANNOY:
                    this.index = new AnnoyIndex(dimension, maxElements);
                    break;
                case LSH:
                    this.index = new LshIndex(dimension, maxElements);
                    break;
                case IVF:
                    this.index = new IvfIndex(dimension, maxElements);
                    break;
                case PQ:
                    this.index = new PqIndex(dimension, maxElements);
                    break;
                case HNSW:
                default:
                    this.index = new HnswIndex(dimension, maxElements);
                    break;
            }
        }
        
        // 从存储加载数据到索引
        loadFromStorage();
    }
    
    /**
     * 直接使用预先创建的存储和索引创建VectorDatabase实例
     * 
     * @param dimension 向量维度
     * @param maxElements 最大元素数量
     * @param storagePath 存储路径
     * @param storage 向量存储实例
     * @param index 向量索引实例
     */
    public VectorDatabase(int dimension, int maxElements, String storagePath, 
                          VectorStorage storage, VectorIndex index) {
        this.dimension = dimension;
        this.maxElements = maxElements;
        this.storagePath = storagePath;
        this.storage = storage;
        this.index = index;
        
        // 从存储加载数据到索引
        loadFromStorage();
    }
    
    /**
     * 添加向量到数据库
     * 
     * @param id 向量ID
     * @param values 向量值
     * @return 是否添加成功
     */
    public boolean addVector(int id, float[] values) {
        if (values.length != dimension) {
            throw new IllegalArgumentException("向量维度不匹配，期望: " + dimension + ", 实际: " + values.length);
        }
        
        Vector vector = new Vector(id, values);
        
        try {
            // 先保存到存储
            boolean stored = storage.saveVector(vector);
            if (!stored) {
                return false;
            }
            
            // 再添加到索引
            return index.addVector(vector);
        } catch (IOException e) {
            throw new RuntimeException("保存向量失败", e);
        }
    }
    
    /**
     * 根据ID获取向量
     * 
     * @param id 向量ID
     * @return 向量（如果存在）
     */
    public Optional<Vector> getVector(int id) {
        try {
            return storage.getVector(id);
        } catch (IOException e) {
            throw new RuntimeException("获取向量失败", e);
        }
    }
    
    /**
     * 删除向量
     * 
     * @param id 向量ID
     * @return 是否删除成功
     */
    public boolean deleteVector(int id) {
        try {
            // 先从存储中删除
            boolean removed = storage.deleteVector(id);
            if (!removed) {
                return false;
            }
            
            // 再从索引中删除
            return index.removeVector(id);
        } catch (IOException e) {
            throw new RuntimeException("删除向量失败", e);
        }
    }
    
    /**
     * 搜索最近邻向量
     * 
     * @param queryVector 查询向量
     * @param k 返回结果数量
     * @return 最近邻向量列表
     */
    public List<SearchResult> search(float[] queryVector, int k) {
        if (queryVector.length != dimension) {
            throw new IllegalArgumentException("查询向量维度不匹配，期望: " + dimension + ", 实际: " + queryVector.length);
        }
        
        Vector query = new Vector(-1, queryVector); // 使用-1作为临时ID
        return index.searchNearest(query, k);
    }
    
    /**
     * 获取数据库中的向量数量
     */
    public int size() {
        return index.size();
    }
    
    /**
     * 获取当前使用的索引实例
     * 
     * @return 索引实例
     */
    public VectorIndex getIndex() {
        return index;
    }
    
    /**
     * 获取当前使用的索引类型
     * 
     * @return 索引类型
     */
    public String getIndexType() {
        return index.getClass().getSimpleName();
    }
    
    /**
     * 重建索引
     * 用于在批量添加或删除向量后，重新构建索引结构以提高搜索效率
     * 
     * @return 是否重建成功
     */
    public boolean rebuildIndex() {
        return index.buildIndex();
    }
    
    /**
     * 从存储中加载向量数据
     */
    public boolean loadFromStorage() {
        try {
            List<Vector> loadedVectors = storage.loadVectors();
            
            if (loadedVectors.isEmpty()) {
                return false;
            }
            
            for (Vector vector : loadedVectors) {
                try {
                    index.addVector(vector);
                } catch (Exception e) {
                    log.error("将向量添加到索引时出错 (ID: {}): {}", vector.getId(), e.getMessage(), e);
                }
            }
            
            log.info("成功从存储加载了 {} 个向量", loadedVectors.size());
            return true;
        } catch (Exception e) {
            log.error("加载向量数据失败: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * 关闭数据库，释放资源
     */
    @Override
    public void close() throws IOException {
        storage.close();
    }
    
    /**
     * 向量数据库构建器
     */
    public static class Builder {
        private int dimension = 1000; // 默认维度
        private int maxElements = 100000; // 默认最大元素数
        private String storagePath = "./vector_data"; // 默认存储路径
        private IndexType indexType = IndexType.HNSW; // 默认使用HNSW索引
        private VectorStorage storage;
        private VectorIndex index;
        
        public Builder withDimension(int dimension) {
            this.dimension = dimension;
            return this;
        }
        
        public Builder withMaxElements(int maxElements) {
            this.maxElements = maxElements;
            return this;
        }
        
        public Builder withStoragePath(String storagePath) {
            this.storagePath = storagePath;
            return this;
        }
        
        /**
         * 设置索引类型
         * 
         * @param indexType 索引类型
         * @return Builder实例
         */
        public Builder withIndexType(IndexType indexType) {
            this.indexType = indexType;
            return this;
        }
        
        public Builder withStorage(VectorStorage storage) {
            this.storage = storage;
            return this;
        }
        
        public Builder withIndex(VectorIndex index) {
            this.index = index;
            return this;
        }
        
        public VectorDatabase build() {
            return new VectorDatabase(this);
        }
    }
} 