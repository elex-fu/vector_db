cmake_minimum_required(VERSION 3.16)
project(vectordb_native VERSION 3.0.0 LANGUAGES CXX)

# C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译优化选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=core-avx2")

# 检测平台
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(X86_64 ON)
    message(STATUS "Building for x86_64 architecture")

    # SIMD支持检测
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

    if(COMPILER_SUPPORTS_AVX2)
        message(STATUS "AVX2 support enabled")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
        add_definitions(-DHAVE_AVX2)
    endif()

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ARM64 ON)
    message(STATUS "Building for ARM64 architecture")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+fp+simd")
    add_definitions(-DHAVE_NEON)
endif()

# 源文件
set(CORE_SOURCES
    core/VectorStore.cpp
)

set(COMPUTE_SOURCES
    compute/DistanceScalar.cpp
    compute/DistanceAVX2.cpp
    compute/SIMDDispatcher.cpp
)

set(INDEX_SOURCES
    index/HNSWIndex.cpp
    index/PQIndex.cpp
    index/IVFIndex.cpp
    index/LSHIndex.cpp
    index/AnnoyIndex.cpp
)

set(BRIDGE_SOURCES
    bridge/VectorDBJNI.cpp
)

set(TEST_SOURCES
    test/test_hnsw.cpp
    test/test_simple.cpp
    test/test_performance.cpp
)

# JNI支持 (可选)
find_package(JNI)
if(JNI_FOUND)
    message(STATUS "JNI found: ${JNI_INCLUDE_DIRS}")
    include_directories(${JNI_INCLUDE_DIRS})
    set(BRIDGE_SOURCES
        bridge/VectorDBJNI.cpp
    )
else()
    message(STATUS "JNI not found, building without Java support")
    set(BRIDGE_SOURCES "")
endif()

# 创建共享库
add_library(vectordb SHARED
    ${CORE_SOURCES}
    ${COMPUTE_SOURCES}
    ${INDEX_SOURCES}
    ${BRIDGE_SOURCES}
)

target_include_directories(vectordb PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(JNI_FOUND)
    target_include_directories(vectordb PUBLIC ${JNI_INCLUDE_DIRS})
endif()

# 链接线程库
find_package(Threads REQUIRED)
target_link_libraries(vectordb Threads::Threads)

# 设置库属性
set_target_properties(vectordb PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER ""
)

# 测试可执行文件
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz
    )
    FetchContent_MakeAvailable(googletest)

    add_executable(vectordb_test ${TEST_SOURCES})
    target_link_libraries(vectordb_test vectordb gtest_main)

    enable_testing()
    add_test(NAME vectordb_test COMMAND vectordb_test)
endif()

# 安装规则
install(TARGETS vectordb
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# 打印配置信息
message(STATUS "")
message(STATUS "========== VectorDB Native Build Configuration ==========")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "AVX2 Support: ${COMPILER_SUPPORTS_AVX2}")
message(STATUS "====================================================")
message(STATUS "")
