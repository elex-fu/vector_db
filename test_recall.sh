#!/bin/bash
# 召回率优化测试脚本

echo "========== VectorDB 召回率优化测试 =========="
echo ""
echo "修复内容:"
echo "  Fix #1: PQ参数优化 (256→64子空间)"
echo "  Fix #2: efSearch调整 (1%→15%数据访问)"
echo "  Fix #3: 双层重排序 (20x→100x候选)"
echo "  Fix #4: 精确距离建图"
echo ""
echo "目标: Recall 8.56% → 90%+"
echo ""

cd /Users/lex/Documents/product/vector/vectordb

# 运行测试
mvn test -Dtest=RecallOptimizationTest#testFix1_PQParameterOptimization -q 2>&1 | grep -E "(测试|验证|✓|✗|通过|失败|PQ|子空间|压缩)" && echo ""
mvn test -Dtest=RecallOptimizationTest#testQuickRecallValidation -q 2>&1 | tail -30

echo ""
echo "========== 测试完成 =========="
