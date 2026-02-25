package com.vectordb.jni;

import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.nio.file.*;

/**
 * Native库加载器
 * 负责从JAR包或文件系统加载本地库
 */
@Slf4j
public class NativeLoader {

    private static final String LIBRARY_NAME = "vectordb";
    private static boolean loaded = false;

    /**
     * 加载本地库
     * @return 是否成功加载
     */
    public static synchronized boolean load() {
        if (loaded) {
            return true;
        }

        try {
            // 尝试从系统属性或环境变量指定的路径加载
            String libPath = System.getProperty("vectordb.native.path");
            if (libPath != null) {
                System.load(libPath);
                loaded = true;
                log.info("Loaded native library from: {}", libPath);
                return true;
            }

            // 尝试从Java库路径加载
            try {
                System.loadLibrary(LIBRARY_NAME);
                loaded = true;
                log.info("Loaded native library from java.library.path");
                return true;
            } catch (UnsatisfiedLinkError e) {
                log.debug("Could not load from java.library.path, trying embedded library");
            }

            // 从JAR包中解压并加载
            loadFromJar();
            loaded = true;
            return true;

        } catch (Exception e) {
            log.error("Failed to load native library", e);
            return false;
        }
    }

    /**
     * 从JAR包中加载本地库
     */
    private static void loadFromJar() throws IOException {
        String libName = System.mapLibraryName(LIBRARY_NAME);
        String resourcePath = "/native/" + libName;

        // 创建临时目录
        Path tempDir = Files.createTempDirectory("vectordb-native-");
        tempDir.toFile().deleteOnExit();

        Path libFile = tempDir.resolve(libName);

        // 从JAR中复制库文件
        try (InputStream is = NativeLoader.class.getResourceAsStream(resourcePath)) {
            if (is == null) {
                throw new FileNotFoundException("Native library not found in JAR: " + resourcePath);
            }
            Files.copy(is, libFile, StandardCopyOption.REPLACE_EXISTING);
        }

        // 设置可执行权限（Unix系统）
        libFile.toFile().setExecutable(true);
        libFile.toFile().deleteOnExit();

        // 加载库
        System.load(libFile.toAbsolutePath().toString());
        log.info("Loaded native library from JAR: {}", libFile);
    }

    /**
     * 检查本地库是否已加载
     */
    public static boolean isLoaded() {
        return loaded;
    }
}
