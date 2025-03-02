package com.vectordb;

import com.vectordb.core.VectorTest;
import com.vectordb.core.SearchResultTest;
import com.vectordb.index.HnswIndexTest;
import com.vectordb.storage.VectorStorageTest;
import com.vectordb.util.VectorUtilsTest;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/**
 * 向量数据库测试套件
 * 运行所有单元测试
 */
@RunWith(Suite.class)
@SuiteClasses({
    // 核心类测试
    VectorTest.class,
    SearchResultTest.class,
    
    // 索引测试
    HnswIndexTest.class,
    
    // 存储测试
    VectorStorageTest.class,
    
    // 工具类测试
    VectorUtilsTest.class,
    
    // 集成测试
    VectorDatabaseIntegrationTest.class
})
public class VectorDatabaseTestSuite {
    // 测试套件不需要任何代码
    // JUnit会运行@SuiteClasses中列出的所有测试
} 