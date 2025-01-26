
cmake_minimum_required(VERSION 3.8)

add_library(comm STATIC nccl.cu)
target_link_libraries(comm PRIVATE nccl)

set_property(TARGET comm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET comm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_subdirectory(custom)

add_executable(test_comm test_comm.cu)
target_link_libraries(test_comm PRIVATE comm custom_comm pthread nvtx_utils)
target_compile_options(test_comm PRIVATE -O3 -march=native -mtune=native)
