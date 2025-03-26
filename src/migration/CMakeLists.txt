cmake_minimum_required(VERSION 3.10)

project(_migration C CXX CUDA)

option(BUILD_BENCH   "Build transfer engine benchmark"  OFF)
option(BUILD_PYTHON  "Build transfer engine benchmark"  ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(ops)
add_subdirectory(engine)

if (BUILD_PYTHON)
add_subdirectory(python)
endif (BUILD_PYTHON)

if (BUILD_BENCH)
add_subdirectory(bench)
endif (BUILD_BENCH)
