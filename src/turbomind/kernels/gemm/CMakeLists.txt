# Copyright (c) OpenMMLab. All rights reserved.

FetchContent_Declare(
  repo-nvbench
  GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
  GIT_TAG        d8dced8a64d9ce305add92fa6d274fd49b569b7e
)

# set(BUILD_SHARED_LIBS OFF)
# set(NVBench_ENABLE_EXAMPLES OFF)

# FetchContent_MakeAvailable(repo-nvbench)

# add_executable(gemm_bench gemm_bench.cu)
# target_link_libraries(gemm_bench PRIVATE nvbench::main)

add_library(gemm2 gemm.cu kernel.cu transcript.cu
        codegen/gemm_sm80_f16_u4_f16_asym_g128.cu)
target_compile_options(gemm2 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v --generate-line-info>)

add_executable(test_gemm_kernels test_gemm.cu test_utils.cu cache_utils.cu)
target_link_libraries(test_gemm_kernels PRIVATE gemm2 cublas)