package com.vectordb.core;

/**
 * 表示向量搜索的结果
 */
public class SearchResult implements Comparable<SearchResult> {
    private final int id;
    private final float distance;
    private final float similarity; // 相似度，范围0~1

    // 相似度计算的缩放因子，用于调整相似度范围
    private static final float SIMILARITY_SCALE_FACTOR = 0.5f;

    /**
     * 创建一个搜索结果
     * 
     * @param id 向量ID
     * @param distance 与查询向量的距离
     */
    public SearchResult(int id, float distance) {
        this.id = id;
        this.distance = distance;
        // 计算相似度
        this.similarity = calculateSimilarity(distance);
    }

    /**
     * 计算相似度
     * 使用缩放后的公式计算相似度，增加区分度
     * 这样当距离为0时，相似度为1；距离越大，相似度越接近0
     * 最后将结果四舍五入到四位小数
     */
    private float calculateSimilarity(float distance) {
        // 使用缩放因子调整相似度计算
        float scaledDistance = distance * SIMILARITY_SCALE_FACTOR;
        float sim = 1.0f / (1.0f + scaledDistance);
        
        // 四舍五入到四位小数
        return Math.round(sim * 10000) / 10000.0f;
    }

    /**
     * 获取向量ID
     */
    public int getId() {
        return id;
    }

    /**
     * 获取与查询向量的距离
     */
    public float getDistance() {
        return distance;
    }

    /**
     * 获取相似度（0~1之间，四位小数）
     */
    public float getSimilarity() {
        return similarity;
    }

    /**
     * 比较两个搜索结果的距离
     * 用于按距离排序（升序）
     */
    @Override
    public int compareTo(SearchResult other) {
        return Float.compare(this.distance, other.distance);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SearchResult that = (SearchResult) o;
        return id == that.id && Float.compare(that.distance, distance) == 0;
    }

    @Override
    public int hashCode() {
        int result = id;
        result = 31 * result + (distance != +0.0f ? Float.floatToIntBits(distance) : 0);
        return result;
    }

    @Override
    public String toString() {
        return "SearchResult{" +
                "id=" + id +
                ", distance=" + distance +
                ", similarity=" + similarity +
                '}';
    }
} 