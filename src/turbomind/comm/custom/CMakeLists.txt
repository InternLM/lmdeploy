cmake_minimum_required(VERSION 3.8)

FetchContent_Declare(
                repo-mscclpp
                GIT_REPOSITORY https://github.com/microsoft/mscclpp.git
                GIT_TAG        11e62024d3eb190e005b4689f8c8443d91a6c82e
                )

set(MSCCLPP_BUILD_TESTS OFF)
set(MSCCLPP_BUILD_PYTHON_BINDINGS OFF)
set(MSCCLPP_BUILD_APPS_NCCL OFF)
set(MSCCLPP_USE_CUDA ON)
set(MSCCLPP_BYPASS_GPU_CHECK ON)

FetchContent_MakeAvailable(repo-mscclpp)


add_library(custom_comm STATIC custom_comm.cu allreduce.cu allgather.cu fused_rms_norm.cu)
target_link_libraries(custom_comm PRIVATE mscclpp_static rms_norm)

set_property(TARGET custom_comm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET custom_comm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)