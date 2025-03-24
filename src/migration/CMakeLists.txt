cmake_minimum_required(VERSION 3.10)

project(_migration C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

add_library(
    _migration
    SHARED
    transfer_engine.cpp
    rdma_transport.cpp
    ibv_helper.cpp
    ops/gather.cu
    ops/scatter.cu
)
target_link_libraries(_migration PUBLIC ibverbs CUDA::cudart)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Python Wrapper
find_package(pybind11 REQUIRED)
pybind11_add_module(
    _migration_c
    bind.cpp
)
target_link_libraries(_migration_c PRIVATE _migration)
