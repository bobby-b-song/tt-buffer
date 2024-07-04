cmake_minimum_required(VERSION 3.16)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Set the compiler
set(CMAKE_CXX_COMPILER /home/boyaosong/external/llvm-project/build/bin/clang++-17)

project(summa LANGUAGES CXX)

# Add compile options
add_compile_options(
    -DBOOST_ALL_NO_LIB
    -DBOOST_ATOMIC_DYN_LINK
    -DBOOST_FILESYSTEM_DYN_LINK
    -DBOOST_REGEX_DYN_LINK
    -DBOOST_SYSTEM_DYN_LINK
    -DBOOST_THREAD_DYN_LINK
    -DFMT_HEADER_ONLY
    -DTT_METAL_VERSIM_DISABLED
    -O3
    -g
    -fno-omit-frame-pointer
    -Wno-int-to-pointer-cast
    -mavx2
    -fPIC
    -fvisibility-inlines-hidden
    -fno-lto
    -Werror
    -Wdelete-non-virtual-dtor
    -Wreturn-type
    -Wswitch
    -Wuninitialized
    -Wno-unused-parameter
    -Wsometimes-uninitialized
    -Wno-c++11-narrowing
    -Wno-c++20-extensions
    -Wno-c++23-extensions
    -Wno-error=local-type-template-args
    -Wno-delete-non-abstract-non-virtual-dtor
    -Wno-c99-designator
    -Wno-shift-op-parentheses
    -Wno-non-c-typedef-for-linkage
)

# Include directories
include_directories(
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
    $ENV{TT_METAL_HOME}
    $ENV{TT_METAL_HOME}/tt_metal
    $ENV{TT_METAL_HOME}/tt_metal/common
    $ENV{TT_METAL_HOME}/tt_metal/programming_examples
    $ENV{TT_METAL_HOME}/tt_metal/third_party/fmt
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/grayskull
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/grayskull
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/grayskull
)

link_directories(
    $ENV{TT_METAL_HOME}/build/lib/
)

# Add the executable
add_executable(matmul_multi_core
    main.cpp
)

# Link libraries
target_link_libraries(matmul_multi_core PUBLIC
    tt_metal
    tt_eager
    stdc++fs
    yaml-cpp
    m
    pthread
)

