cmake_minimum_required(VERSION 3.10)

project(_slime C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_BENCH   "Build transfer engine benchmark"  OFF)
option(BUILD_PYTHON  "Build python wrapper"  ON)
option(BUILD_TEST    "Build test"  OFF)

include_directories(csrc)
add_subdirectory(csrc)

if (BUILD_BENCH)
add_subdirectory(bench)
endif (BUILD_BENCH)

if (BUILD_TEST)
add_subdirectory(tests/cpp)
endif (BUILD_TEST)
