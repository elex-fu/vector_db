package com.vectordb.core;

import org.junit.Test;
import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * SearchResult类的单元测试
 */
public class SearchResultTest {

    /**
     * 测试基本属性
     */
    @Test
    public void testBasicProperties() {
        int id = 42;
        float distance = 0.75f;
        SearchResult result = new SearchResult(id, distance);
        
        assertEquals(id, result.getId());
        assertEquals(distance, result.getDistance(), 0.0001f);
    }
    
    /**
     * 测试比较功能（用于排序）
     */
    @Test
    public void testComparison() {
        SearchResult r1 = new SearchResult(1, 0.5f);
        SearchResult r2 = new SearchResult(2, 0.3f);
        SearchResult r3 = new SearchResult(3, 0.7f);
        
        // r2 < r1 < r3 (按距离排序)
        assertTrue(r1.compareTo(r2) > 0);
        assertTrue(r1.compareTo(r3) < 0);
        assertTrue(r2.compareTo(r3) < 0);
        
        // 相等的情况
        assertEquals(0, r1.compareTo(new SearchResult(4, 0.5f)));
    }
    
    /**
     * 测试排序功能
     */
    @Test
    public void testSorting() {
        List<SearchResult> results = new ArrayList<>();
        results.add(new SearchResult(1, 0.5f));
        results.add(new SearchResult(2, 0.3f));
        results.add(new SearchResult(3, 0.7f));
        results.add(new SearchResult(4, 0.1f));
        
        Collections.sort(results);
        
        // 排序后应该是按距离升序
        assertEquals(4, results.get(0).getId()); // 距离0.1
        assertEquals(2, results.get(1).getId()); // 距离0.3
        assertEquals(1, results.get(2).getId()); // 距离0.5
        assertEquals(3, results.get(3).getId()); // 距离0.7
    }
    
    /**
     * 测试equals和hashCode方法
     */
    @Test
    public void testEqualsAndHashCode() {
        SearchResult r1 = new SearchResult(1, 0.5f);
        SearchResult r2 = new SearchResult(1, 0.5f);
        SearchResult r3 = new SearchResult(2, 0.5f);
        SearchResult r4 = new SearchResult(1, 0.6f);
        
        // 相同ID和距离的结果应该相等
        assertEquals(r1, r2);
        assertEquals(r1.hashCode(), r2.hashCode());
        
        // 不同ID的结果不应该相等
        assertNotEquals(r1, r3);
        
        // 不同距离的结果不应该相等
        assertNotEquals(r1, r4);
        
        // 与null比较
        assertNotEquals(r1, null);
        
        // 与其他类型比较
        assertNotEquals(r1, "not a search result");
    }
    
    /**
     * 测试toString方法
     */
    @Test
    public void testToString() {
        SearchResult result = new SearchResult(42, 0.75f);
        String str = result.toString();
        
        // toString应该包含ID和距离信息
        assertTrue(str.contains("42"));
        assertTrue(str.contains("0.75"));
    }
} 