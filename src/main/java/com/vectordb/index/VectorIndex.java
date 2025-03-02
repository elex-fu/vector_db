package com.vectordb.index;

import com.vectordb.core.SearchResult;
import com.vectordb.core.Vector;

import java.util.List;

/**
 * 向量索引接口，定义索引的基本操作
 */
public interface VectorIndex {
    
    /**
     * 添加向量到索引
     * 
     * @param vector 要添加的向量
     * @return 是否添加成功
     */
    boolean addVector(Vector vector);
    
    /**
     * 从索引中移除向量
     * 
     * @param id 要移除的向量ID
     * @return 是否移除成功
     */
    boolean removeVector(int id);
    
    /**
     * 搜索最近邻向量
     * 
     * @param queryVector 查询向量
     * @param k 返回结果数量
     * @return 最近邻向量列表
     */
    List<SearchResult> searchNearest(Vector queryVector, int k);
    
    /**
     * 获取索引中的向量数量
     * 
     * @return 向量数量
     */
    int size();
    
    /**
     * 重建索引
     * 用于在批量添加或删除向量后，重新构建索引结构以提高搜索效率
     * 
     * @return 是否重建成功
     */
    boolean buildIndex();
} 