cmake_minimum_required(VERSION 3.10)

project(_migration C CXX CUDA)

include_directories(${PROJECT_SOURCE_DIR})

add_library(
    _migration
    SHARED
    transfer_engine.cpp
    rdma_transport.cpp
    ibv_helper.cpp
)
target_link_libraries(_migration PUBLIC ibverbs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Python Wrapper
find_package(pybind11 REQUIRED)
pybind11_add_module(
    _migration_c
    bind.cpp
)
target_link_libraries(_migration_c PUBLIC _migration)
