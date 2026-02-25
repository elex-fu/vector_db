package com.vectordb.jni;

import java.io.*;
import java.nio.file.*;
import java.util.Locale;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Native库加载器
 * 负责从JAR包或文件系统加载本地库，支持多平台多架构
 */
public class NativeLoader {

    private static final Logger LOGGER = Logger.getLogger(NativeLoader.class.getName());
    private static final String LIBRARY_NAME = "vectordb";
    private static volatile boolean loaded = false;

    /**
     * 平台架构枚举
     */
    public enum Platform {
        LINUX_X86_64("linux-x86_64"),
        LINUX_ARM64("linux-arm64"),
        MACOS_X86_64("macos-x86_64"),
        MACOS_ARM64("macos-arm64"),
        WINDOWS_X86_64("windows-x86_64"),
        UNKNOWN("unknown");

        private final String name;

        Platform(String name) {
            this.name = name;
        }

        public String getName() {
            return name;
        }
    }

    /**
     * 加载本地库（自动检测平台）
     * @return 是否成功加载
     */
    public static synchronized boolean load() {
        if (loaded) {
            return true;
        }

        try {
            // 1. 尝试从系统属性指定的路径加载
            String libPath = System.getProperty("vectordb.native.path");
            if (libPath != null && !libPath.isEmpty()) {
                System.load(libPath);
                loaded = true;
                LOGGER.info("Loaded native library from system property: " + libPath);
                return true;
            }

            // 2. 尝试从环境变量指定的路径加载
            String envPath = System.getenv("VECTORDB_NATIVE_PATH");
            if (envPath != null && !envPath.isEmpty()) {
                System.load(envPath);
                loaded = true;
                LOGGER.info("Loaded native library from environment variable: " + envPath);
                return true;
            }

            // 3. 尝试从Java库路径加载
            try {
                System.loadLibrary(LIBRARY_NAME);
                loaded = true;
                LOGGER.info("Loaded native library from java.library.path");
                return true;
            } catch (UnsatisfiedLinkError e) {
                LOGGER.fine("Could not load from java.library.path, trying embedded library");
            }

            // 4. 从JAR包中自动检测并加载
            Platform platform = detectPlatform();
            if (platform == Platform.UNKNOWN) {
                throw new RuntimeException("Unsupported platform/architecture: " + System.getProperty("os.name") + " / " + System.getProperty("os.arch"));
            }

            loadFromJar(platform);
            loaded = true;
            LOGGER.info("Loaded native library from JAR for platform: " + platform.getName());
            return true;

        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Failed to load native library: " + e.getMessage(), e);
            return false;
        }
    }

    /**
     * 检测当前平台架构
     */
    public static Platform detectPlatform() {
        String osName = System.getProperty("os.name").toLowerCase(Locale.ROOT);
        String osArch = System.getProperty("os.arch").toLowerCase(Locale.ROOT);

        boolean isLinux = osName.contains("linux");
        boolean isMac = osName.contains("mac") || osName.contains("darwin");
        boolean isWindows = osName.contains("windows");

        boolean isX86_64 = osArch.equals("amd64") || osArch.equals("x86_64");
        boolean isArm64 = osArch.equals("aarch64") || osArch.equals("arm64");

        if (isLinux && isX86_64) {
            return Platform.LINUX_X86_64;
        } else if (isLinux && isArm64) {
            return Platform.LINUX_ARM64;
        } else if (isMac && isX86_64) {
            return Platform.MACOS_X86_64;
        } else if (isMac && isArm64) {
            return Platform.MACOS_ARM64;
        } else if (isWindows && isX86_64) {
            return Platform.WINDOWS_X86_64;
        }

        return Platform.UNKNOWN;
    }

    /**
     * 从JAR包中加载特定平台的本地库
     */
    private static void loadFromJar(Platform platform) throws IOException {
        String libName = System.mapLibraryName(LIBRARY_NAME);
        String platformDir = platform.getName();
        String resourcePath = "/native/" + platformDir + "/" + libName;

        LOGGER.fine("Attempting to load native library from: " + resourcePath);

        // 创建临时目录
        Path tempDir = Files.createTempDirectory("vectordb-native-" + platformDir + "-");
        tempDir.toFile().deleteOnExit();

        Path libFile = tempDir.resolve(libName);

        // 从JAR中复制库文件
        try (InputStream is = NativeLoader.class.getResourceAsStream(resourcePath)) {
            if (is == null) {
                // 尝试不带平台目录的旧路径
                String fallbackPath = "/native/" + libName;
                try (InputStream fallbackIs = NativeLoader.class.getResourceAsStream(fallbackPath)) {
                    if (fallbackIs == null) {
                        throw new FileNotFoundException(
                            "Native library not found in JAR. Tried paths:\n" +
                            "  - " + resourcePath + "\n" +
                            "  - " + fallbackPath + "\n" +
                            "Detected platform: " + platform.getName()
                        );
                    }
                    Files.copy(fallbackIs, libFile, StandardCopyOption.REPLACE_EXISTING);
                }
            } else {
                Files.copy(is, libFile, StandardCopyOption.REPLACE_EXISTING);
            }
        }

        // 设置可执行权限（Unix系统）
        File libFileObj = libFile.toFile();
        libFileObj.setExecutable(true, false);
        libFileObj.deleteOnExit();

        // 在 macOS 上，可能需要加载依赖库
        if (platform.name.startsWith("macos")) {
            setupMacLibraryPath(tempDir);
        }

        // 加载库
        System.load(libFile.toAbsolutePath().toString());
        LOGGER.info("Successfully loaded native library: " + libFile);
    }

    /**
     * 设置 macOS 库路径
     */
    private static void setupMacLibraryPath(Path tempDir) {
        try {
            String currentPath = System.getProperty("java.library.path", "");
            String newPath = tempDir.toAbsolutePath().toString() + File.pathSeparator + currentPath;
            System.setProperty("java.library.path", newPath);

            // 触发 ClassLoader 重新加载库路径
            java.lang.reflect.Field field = ClassLoader.class.getDeclaredField("sys_paths");
            field.setAccessible(true);
            field.set(null, null);
        } catch (Exception e) {
            LOGGER.fine("Could not update java.library.path: " + e.getMessage());
        }
    }

    /**
     * 显式加载指定平台的库（用于测试或特殊场景）
     */
    public static synchronized boolean loadForPlatform(Platform platform) {
        if (loaded) {
            return true;
        }

        try {
            loadFromJar(platform);
            loaded = true;
            LOGGER.info("Loaded native library for specified platform: " + platform.getName());
            return true;
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Failed to load native library for platform " + platform.getName() + ": " + e.getMessage(), e);
            return false;
        }
    }

    /**
     * 检查本地库是否已加载
     */
    public static boolean isLoaded() {
        return loaded;
    }

    /**
     * 获取库加载状态信息
     */
    public static String getLoadStatus() {
        StringBuilder sb = new StringBuilder();
        sb.append("Native Library Load Status:\n");
        sb.append("  Loaded: ").append(loaded).append("\n");
        sb.append("  Detected Platform: ").append(detectPlatform().getName()).append("\n");
        sb.append("  OS Name: ").append(System.getProperty("os.name")).append("\n");
        sb.append("  OS Arch: ").append(System.getProperty("os.arch")).append("\n");
        sb.append("  Java Library Path: ").append(System.getProperty("java.library.path")).append("\n");

        String customPath = System.getProperty("vectordb.native.path");
        if (customPath != null) {
            sb.append("  Custom Path (property): ").append(customPath).append("\n");
        }

        String envPath = System.getenv("VECTORDB_NATIVE_PATH");
        if (envPath != null) {
            sb.append("  Custom Path (env): ").append(envPath).append("\n");
        }

        return sb.toString();
    }
}
