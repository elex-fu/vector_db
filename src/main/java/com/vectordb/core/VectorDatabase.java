package com.vectordb.core;

import com.vectordb.config.CompressionConfig;
import com.vectordb.index.AnnoyIndex;
import com.vectordb.index.HnswIndex;
import com.vectordb.index.HnswPqIndex;
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
        HNSW,     // 层次可导航小世界图索引
        ANNOY,    // 近似最近邻哦耶索引
        LSH,      // 局部敏感哈希索引
        IVF,      // 倒排文件索引
        PQ,       // 乘积量化索引
        HNSWPQ    // HNSW + PQ 混合索引 (支持压缩)
    }
    
    private final int dimension;
    private final int maxElements;
    private final String storagePath;
    private final VectorIndex index;
    private final VectorStorage storage;
    private final CompressionConfig compressionConfig;
    
    /**
     * 使用Builder模式创建VectorDatabase实例
     */
    private VectorDatabase(Builder builder) {
        this.dimension = builder.dimension;
        this.maxElements = builder.maxElements;
        this.storagePath = builder.storagePath;
        this.compressionConfig = builder.compressionConfig != null ?
                builder.compressionConfig :
                CompressionConfig.defaultConfig();

        // 初始化存储
        this.storage = builder.storage != null ?
                builder.storage :
                new VectorStorage(storagePath, dimension, maxElements);

        // 根据索引类型初始化索引
        if (builder.index != null) {
            this.index = builder.index;
        } else {
            this.index = createIndex(builder.indexType, this.compressionConfig);
        }

        // 从存储加载数据到索引
        loadFromStorage();
    }

    /**
     * 根据索引类型和压缩配置创建索引实例
     */
    private VectorIndex createIndex(IndexType indexType, CompressionConfig compression) {
        // 如果启用了压缩且使用HNSWPQ类型，创建HNSW+PQ混合索引
        if (compression.isEnabled() && compression.getType() == CompressionConfig.CompressionType.HNSWPQ) {
            log.info("创建HNSW+PQ混合索引，PQ子空间数: {}, 位数: {}",
                    compression.getPqSubspaces(), compression.getPqBits());
            return new HnswPqIndex(dimension, maxElements, compression);
        }

        // 如果启用了压缩且使用PQ类型，创建纯PQ索引
        if (compression.isEnabled() && compression.getType() == CompressionConfig.CompressionType.PQ) {
            log.info("创建PQ索引，PQ子空间数: {}, 位数: {}",
                    compression.getPqSubspaces(), compression.getPqBits());
            return new PqIndex(dimension, maxElements, compression);
        }

        // 默认根据indexType创建索引
        switch (indexType) {
            case ANNOY:
                return new AnnoyIndex(dimension, maxElements);
            case LSH:
                return new LshIndex(dimension, maxElements);
            case IVF:
                return new IvfIndex(dimension, maxElements);
            case PQ:
                return new PqIndex(dimension, maxElements);
            case HNSWPQ:
                // 即使压缩未启用，也允许显式使用HNSWPQ索引（使用默认PQ配置）
                return new HnswPqIndex(dimension, maxElements, CompressionConfig.recommendedConfig(dimension));
            case HNSW:
            default:
                return new HnswIndex(dimension, maxElements);
        }
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
        this(dimension, maxElements, storagePath, storage, index, CompressionConfig.defaultConfig());
    }

    /**
     * 直接使用预先创建的存储和索引创建VectorDatabase实例，带压缩配置
     *
     * @param dimension 向量维度
     * @param maxElements 最大元素数量
     * @param storagePath 存储路径
     * @param storage 向量存储实例
     * @param index 向量索引实例
     * @param compressionConfig 压缩配置
     */
    public VectorDatabase(int dimension, int maxElements, String storagePath,
                          VectorStorage storage, VectorIndex index,
                          CompressionConfig compressionConfig) {
        this.dimension = dimension;
        this.maxElements = maxElements;
        this.storagePath = storagePath;
        this.storage = storage;
        this.index = index;
        this.compressionConfig = compressionConfig != null ?
                compressionConfig : CompressionConfig.defaultConfig();

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
     * 获取压缩配置
     *
     * @return 压缩配置
     */
    public CompressionConfig getCompressionConfig() {
        return compressionConfig;
    }

    /**
     * 检查是否启用了压缩
     *
     * @return 是否启用压缩
     */
    public boolean isCompressionEnabled() {
        return compressionConfig != null && compressionConfig.isEnabled();
    }

    /**
     * 获取当前压缩比
     *
     * @return 压缩比，如果未启用压缩则返回1.0
     */
    public double getCompressionRatio() {
        if (compressionConfig == null) {
            return 1.0;
        }
        return compressionConfig.getCompressionRatio(dimension);
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
        private CompressionConfig compressionConfig;

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

        /**
         * 设置压缩配置
         *
         * @param config 压缩配置
         * @return Builder实例
         */
        public Builder withCompression(CompressionConfig config) {
            this.compressionConfig = config;
            return this;
        }

        /**
         * 启用压缩（使用推荐配置）
         *
         * @return Builder实例
         */
        public Builder withCompressionEnabled() {
            this.compressionConfig = CompressionConfig.recommendedConfig(this.dimension);
            return this;
        }

        /**
         * 设置是否启用压缩
         *
         * @param enabled 是否启用
         * @return Builder实例
         */
        public Builder withCompressionEnabled(boolean enabled) {
            if (enabled) {
                this.compressionConfig = CompressionConfig.recommendedConfig(this.dimension);
            } else {
                this.compressionConfig = CompressionConfig.defaultConfig();
            }
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