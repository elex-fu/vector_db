package com.vectordb;

import com.vectordb.core.SearchResult;
import com.vectordb.core.Vector;
import com.vectordb.core.VectorDatabase;
import com.vectordb.core.VectorDatabase.IndexType;
import com.vectordb.index.AnnoyIndex;
import com.vectordb.index.VectorIndex;
import com.vectordb.util.TextVectorizer;

import java.io.File;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * 文字向量搜索示例
 * 测试将文字转换为向量并进行相似度搜索
 * 对比不同索引类型的性能和效果
 * 使用1536维向量进行测试，模拟大型语言模型的嵌入向量
 * 使用10,000个向量进行大规模测试
 */
public class TextSearchExample {
    // 向量维度 - 调整为1536维，模拟大型语言模型的嵌入向量维度
    private static final int VECTOR_DIMENSION = 1536;
    
    // 测试数据集大小 - 增加到10,000个
    private static final int DATASET_SIZE = 1000;
    
    // 最大显示数量
    private static final int MAX_DISPLAY_COUNT = 10;
    
    // 查询测试次数
    private static final int NUM_QUERIES = 100;

    
    // 常用汉字集合（包含一些常见汉字）
    private static final String COMMON_CHINESE_CHARS = 
            "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严";
    
    public static void main(String[] args) {
        System.out.println("开始向量索引对比测试...");
        System.out.println("向量维度: " + VECTOR_DIMENSION + ", 模拟大型语言模型的嵌入向量");
        
        // 测试所有索引类型
        testAllIndexTypes();
    }
    
    /**
     * 测试所有索引类型
     */
    private static void testAllIndexTypes() {
        // 定义要测试的索引类型
        IndexType[] indexTypes = {
            IndexType.HNSW,
            IndexType.ANNOY,
            IndexType.LSH,
            IndexType.IVF,
            IndexType.PQ
        };
        
        // 生成测试数据 - 增加到10000个词汇
        int wordCount = DATASET_SIZE;
        System.out.println("\n生成" + wordCount + "个随机汉字词汇...");
        List<String> chineseWords = generateRandomChineseWords(wordCount, 2, 4);
        System.out.println("生成完成，显示前20个:");
        for (int i = 0; i < Math.min(20, chineseWords.size()); i++) {
            System.out.println((i + 1) + ". " + chineseWords.get(i));
        }
        System.out.println("... 共" + wordCount + "个词汇");
        
        // 生成测试查询
        List<TestQuery> testQueries = new ArrayList<>();
        for (int i = 0; i < NUM_QUERIES; i++) {  // 使用NUM_QUERIES常量

            int targetIndex = i % chineseWords.size();
            String targetWord = chineseWords.get(targetIndex);
            String similarWord = generateSimilarWord(targetWord);
            float[] queryVector = TextVectorizer.textToVector(similarWord, VECTOR_DIMENSION);
            
            TestQuery query = new TestQuery();
            query.targetIndex = targetIndex;
            query.targetWord = targetWord;
            query.similarWord = similarWord;
            query.queryVector = queryVector;
            testQueries.add(query);
        }
        
        // 存储每种索引类型的测试结果
        Map<IndexType, IndexTestResult> results = new HashMap<>();
        
        // 测试每种索引类型
        for (IndexType indexType : indexTypes) {
            System.out.println("\n" + generateRepeatedString("=", 80));
            System.out.println("测试索引类型: " + indexType);
            System.out.println(generateRepeatedString("=", 80));
            
            IndexTestResult result = testIndexType(indexType, chineseWords, testQueries);
            results.put(indexType, result);
            
            // 清理临时文件
            File dbDir = new File("temp_db_" + indexType.name().toLowerCase());
            deleteDirectory(dbDir);
        }
        
        // 输出对比结果
        System.out.println("\n" + generateRepeatedString("=", 80));
        System.out.println("索引类型性能对比");
        System.out.println(generateRepeatedString("=", 80));
        
        System.out.printf("%-10s %-15s %-15s %-15s %-15s %-15s %-15s\n", 
                "索引类型", "平均添加时间(ms)", "平均搜索时间(ms)", "准确率(%)", 
                "Top-3准确率(%)", "Top-5准确率(%)", "内存占用(KB)");
        System.out.println(generateRepeatedString("-", 100));
        
        for (IndexType indexType : indexTypes) {
            IndexTestResult result = results.get(indexType);
            System.out.printf("%-10s %-15.2f %-15.2f %-15.2f %-15.2f %-15.2f %-15.2f\n", 
                    indexType, 
                    result.avgAddTime, 
                    result.avgSearchTime, 
                    result.accuracy * 100, 
                    result.top3Accuracy * 100,
                    result.top5Accuracy * 100,
                    result.memoryUsage / 1024.0);
        }
    }
    
    /**
     * 测试特定索引类型
     * 
     * @param indexType 索引类型
     * @param chineseWords 中文词汇列表
     * @param testQueries 测试查询列表
     * @return 测试结果
     */
    private static IndexTestResult testIndexType(IndexType indexType, List<String> chineseWords, List<TestQuery> testQueries) {
        IndexTestResult result = new IndexTestResult();
        
        // 创建临时目录
        File dbDir = new File("temp_db_" + indexType.name().toLowerCase());
        if (dbDir.exists()) {
            deleteDirectory(dbDir);
        }
        dbDir.mkdirs();
        
        // 设置ANNOY索引参数（如果是ANNOY索引）
        int rebuildThreshold = 1000;
        boolean lazyBuild = true;
        
        try (VectorDatabase db = new VectorDatabase.Builder()
                .withStoragePath(dbDir.getPath())
                .withDimension(VECTOR_DIMENSION)
                .withMaxElements(20000)  // 增加最大元素数量
                .withIndexType(indexType)
                .build()) {
            
            // 如果是ANNOY索引，设置特殊参数
            if (indexType == IndexType.ANNOY) {
                // 获取ANNOY索引实例
                VectorIndex vectorIndex = db.getIndex();
                if (vectorIndex instanceof com.vectordb.index.AnnoyIndex) {
                    com.vectordb.index.AnnoyIndex annoyIndex = (com.vectordb.index.AnnoyIndex) vectorIndex;
                    annoyIndex.setIndexParameters(10, rebuildThreshold, lazyBuild);
                }
            }
            
            System.out.println("\n将词汇转换为向量并添加到数据库...");
            
            // 准备批量添加的向量
            List<com.vectordb.core.Vector> batchVectors = new ArrayList<>();
            long totalAddTime = 0;
            
            // 将词汇转换为向量
            for (int i = 0; i < chineseWords.size(); i++) {
                String word = chineseWords.get(i);
                float[] values = TextVectorizer.textToVector(word, VECTOR_DIMENSION);
                
                // 每1000个向量批量添加一次
                if (batchVectors.size() >= 1000 || i == chineseWords.size() - 1) {
                    long startTime = System.currentTimeMillis();
                    
                    // 批量添加向量
                    if (indexType == IndexType.ANNOY) {
                        // 使用ANNOY索引的批量添加方法
                        VectorIndex vectorIndex = db.getIndex();
                        if (vectorIndex instanceof com.vectordb.index.AnnoyIndex) {
                            com.vectordb.index.AnnoyIndex annoyIndex = (com.vectordb.index.AnnoyIndex) vectorIndex;
                            
                            // 转换为Vector对象
                            List<Vector> vectors = new ArrayList<>();
                            for (int j = 0; j < batchVectors.size(); j++) {
                                int id = batchVectors.get(j).getId();
                                float[] vectorValues = batchVectors.get(j).getValues();
                                vectors.add(new Vector(id, vectorValues));
                            }
                            
                            annoyIndex.addVectors(vectors);
                        } else {
                            // 逐个添加
                            for (int j = 0; j < batchVectors.size(); j++) {
                                int id = batchVectors.get(j).getId();
                                float[] vectorValues = batchVectors.get(j).getValues();
                                db.addVector(id, vectorValues);
                            }
                        }
                    } else {
                        // 其他索引类型逐个添加
                        for (int j = 0; j < batchVectors.size(); j++) {
                            int id = batchVectors.get(j).getId();
                            float[] vectorValues = batchVectors.get(j).getValues();
                            db.addVector(id, vectorValues);
                        }
                    }
                    
                    long endTime = System.currentTimeMillis();
                    long batchTime = endTime - startTime;
                    totalAddTime += batchTime;
                    
                    System.out.println("批量添加 " + batchVectors.size() + " 个向量，耗时: " + batchTime + " 毫秒");
                    batchVectors.clear();
                }
                
                // 创建向量并添加到批处理列表
                com.vectordb.core.Vector vector = new com.vectordb.core.Vector(i + 1, values);
                batchVectors.add(vector);
            }
            
            // 计算平均添加时间
            result.avgAddTime = (double) totalAddTime / chineseWords.size();
            System.out.println("平均每个向量添加时间: " + result.avgAddTime + " 毫秒");
            
            // 重建索引并计时
            System.out.println("\n重建索引...");
            long startRebuild = System.currentTimeMillis();
            db.rebuildIndex();
            long endRebuild = System.currentTimeMillis();
            result.rebuildTime = endRebuild - startRebuild;
            System.out.println("索引重建完成，耗时: " + result.rebuildTime + " 毫秒");
            
            // 测试内存占用
            Runtime runtime = Runtime.getRuntime();
            runtime.gc();
            long usedMemoryBefore = runtime.totalMemory() - runtime.freeMemory();
            
            // 执行一些操作，确保内存使用稳定
            db.size();
            
            runtime.gc();
            long usedMemoryAfter = runtime.totalMemory() - runtime.freeMemory();
            result.memoryUsage = usedMemoryAfter - usedMemoryBefore;
            System.out.println("估计内存占用: " + (result.memoryUsage / 1024) + " KB");
            
            // 测试搜索
            System.out.println("\n测试搜索...");
            int successCount = 0;
            
            for (TestQuery query : testQueries) {
                System.out.println("\n查询: " + query.similarWord + " (目标词汇: " + query.targetWord + ")");
                
                // 搜索最相似的向量
                long startSearch = System.currentTimeMillis();
                List<SearchResult> searchResults = db.search(query.queryVector, 10);
                long endSearch = System.currentTimeMillis();
                
                // 记录查询结果
                QueryResult queryResult = new QueryResult();
                queryResult.searchTime = endSearch - startSearch;
                System.out.println("搜索耗时: " + queryResult.searchTime + " 毫秒");
                
                // 验证最相似的向量是否是目标词汇
                if (!searchResults.isEmpty()) {
                    SearchResult topResult = searchResults.get(0);
                    int topResultId = topResult.getId();
                    queryResult.foundTarget = (topResultId == query.targetIndex + 1);
                    queryResult.similarity = topResult.getSimilarity();
                    
                    // 记录Top-K结果ID
                    for (SearchResult sr : searchResults) {
                        int resultId = sr.getId();
                        // 如果是目标ID，存储为-1作为标记
                        if (resultId == query.targetIndex + 1) {
                            queryResult.topKResults.add(-1);
                        } else {
                            queryResult.topKResults.add(resultId);
                        }
                    }
                    
                    System.out.println("\n验证结果:");
                    System.out.println("最相似向量ID: " + topResultId + 
                            ", 文字: " + chineseWords.get(topResultId - 1));
                    System.out.println("目标词汇ID: " + (query.targetIndex + 1) + 
                            ", 文字: " + query.targetWord);
                    System.out.println("是否找到目标词汇: " + (queryResult.foundTarget ? "是" : "否"));
                    
                    if (queryResult.foundTarget) {
                        System.out.println("测试成功: 成功找到最相似的词汇！");
                        successCount++;
                    } else {
                        System.out.println("测试失败: 未能找到最相似的词汇。");
                    }
                }
                
                result.queryResults.add(queryResult);
            }
            
            result.accuracy = (double) successCount / testQueries.size();
            System.out.println("\n总体准确率: " + (result.accuracy * 100) + "%");
            
            // 计算Top-K准确率
            result.calculateTopKAccuracy(testQueries);
            System.out.println("Top-3准确率: " + (result.top3Accuracy * 100) + "%");
            System.out.println("Top-5准确率: " + (result.top5Accuracy * 100) + "%");
            
            // 计算平均搜索时间
            result.avgSearchTime = result.queryResults.stream()
                    .mapToDouble(qr -> qr.searchTime)
                    .average()
                    .orElse(0);
            System.out.println("平均搜索时间: " + result.avgSearchTime + "ms");
            
        } catch (Exception e) {
            System.err.println("测试索引类型 " + indexType + " 时发生异常: " + e.getMessage());
            e.printStackTrace();
        }
        
        return result;
    }
    
    /**
     * 获取当前内存使用量
     */
    private static long getMemoryUsage() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }
    
    /**
     * 测试查询类
     */
    private static class TestQuery {
        int targetIndex;
        String targetWord;
        String similarWord;
        float[] queryVector;
    }
    
    /**
     * 查询结果类
     */
    private static class QueryResult {
        boolean foundTarget;
        double similarity;
        double searchTime;
        List<Integer> topKResults = new ArrayList<>();  // 新增：存储Top-K结果ID
        
        // 检查目标是否在Top-K结果中
        public boolean foundTargetInTopK(int k) {
            if (topKResults.size() < k) {
                return foundTarget;
            }
            
            for (int i = 0; i < Math.min(k, topKResults.size()); i++) {
                if (topKResults.get(i) == -1) {  // -1表示目标ID
                    return true;
                }
            }
            
            return false;
        }
    }
    
    /**
     * 索引测试结果类
     */
    private static class IndexTestResult {
        double avgAddTime;
        long totalAddTime;
        long rebuildTime;
        double avgSearchTime;
        double accuracy;
        long memoryUsage;
        double top3Accuracy;  // 新增：Top-3准确率
        double top5Accuracy;  // 新增：Top-5准确率
        List<QueryResult> queryResults = new ArrayList<>();
        
        // 计算Top-K准确率
        public void calculateTopKAccuracy(List<TestQuery> testQueries) {
            int top3Count = 0;
            int top5Count = 0;
            
            for (QueryResult result : queryResults) {
                if (result.foundTargetInTopK(3)) {
                    top3Count++;
                }
                if (result.foundTargetInTopK(5)) {
                    top5Count++;
                }
            }
            
            this.top3Accuracy = (double) top3Count / testQueries.size();
            this.top5Accuracy = (double) top5Count / testQueries.size();
        }
    }
    
    /**
     * 生成重复字符串
     * 
     * @param str 要重复的字符串
     * @param count 重复次数
     * @return 重复后的字符串
     */
    private static String generateRepeatedString(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }
    
    /**
     * 根据余弦相似度找到最相似的向量
     * 
     * @param similarityMap ID到相似度的映射
     * @return 最相似的向量ID
     */
    private static int findMostSimilarVectorByCosine(Map<Integer, Float> similarityMap) {
        return similarityMap.entrySet().stream()
                .max(Comparator.comparing(Map.Entry::getValue))
                .map(Map.Entry::getKey)
                .orElse(-1);
    }
    
    /**
     * 生成随机汉字词汇列表
     * 
     * @param count 词汇数量
     * @param minLength 最小词长
     * @param maxLength 最大词长
     * @return 随机汉字词汇列表
     */
    private static List<String> generateRandomChineseWords(int count, int minLength, int maxLength) {
        List<String> words = new ArrayList<>();
        Random random = new Random();
        
        for (int i = 0; i < count; i++) {
            int length = random.nextInt(maxLength - minLength + 1) + minLength;
            StringBuilder word = new StringBuilder();
            
            for (int j = 0; j < length; j++) {
                int charIndex = random.nextInt(COMMON_CHINESE_CHARS.length());
                word.append(COMMON_CHINESE_CHARS.charAt(charIndex));
            }
            
            words.add(word.toString());
        }
        
        return words;
    }
    
    /**
     * 生成相似的词汇（替换或修改一个字符）
     * 
     * @param originalWord 原始词汇
     * @return 相似的词汇
     */
    private static String generateSimilarWord(String originalWord) {
        if (originalWord == null || originalWord.isEmpty()) {
            return "";
        }
        
        Random random = new Random();
        StringBuilder similarWord = new StringBuilder(originalWord);
        
        // 随机选择一个位置
        int position = random.nextInt(originalWord.length());
        
        // 随机选择一个操作：替换字符
        int charIndex = random.nextInt(COMMON_CHINESE_CHARS.length());
        char newChar = COMMON_CHINESE_CHARS.charAt(charIndex);
        
        // 确保新字符与原字符不同
        while (newChar == originalWord.charAt(position)) {
            charIndex = random.nextInt(COMMON_CHINESE_CHARS.length());
            newChar = COMMON_CHINESE_CHARS.charAt(charIndex);
        }
        
        // 执行替换操作
        similarWord.setCharAt(position, newChar);
        
        return similarWord.toString();
    }
    
    /**
     * 删除目录及其所有内容
     * 
     * @param directory 要删除的目录
     */
    private static void deleteDirectory(File directory) {
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