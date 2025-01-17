
cmake_minimum_required(VERSION 3.8)

add_library(comm STATIC nccl.cu)

set_property(TARGET comm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET comm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)


add_executable(test_comm test_comm.cu)
target_link_libraries(test_comm PRIVATE comm pthread nccl)