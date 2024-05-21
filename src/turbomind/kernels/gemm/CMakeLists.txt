# Copyright (c) OpenMMLab. All rights reserved.

FetchContent_Declare(
  repo-nvbench
  GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
  GIT_TAG        d8dced8a64d9ce305add92fa6d274fd49b569b7e
)

set(BUILD_SHARED_LIBS OFF)
set(NVBench_ENABLE_EXAMPLES OFF)

FetchContent_MakeAvailable(repo-nvbench)

add_library(gemm2 gemm.cu kernel.cu transcript.cu dispatch_cache.cu gpu_metric.cu
        # codegen/gemm_sm80_f16_u4_f16_asym_g128_basic.cu
        # codegen/gemm_sm80_f16_u4_f16_asym_g128_extra.cu
        # codegen/gemm_sm80_f16_f16_f16.cu
        # codegen/sm80_s16816gemm_f16_f16.cu
        # codegen/sm70_s884gemm_f16_f16.cu
        codegen/sm80_s16816gemm_f16_f16_v2.cu
        codegen/sm80_s16816gemm_f16_f16_nt.cu
        )

add_executable(gemm_bench gemm_bench.cu test_utils.cu cache_utils.cu)
target_link_libraries(gemm_bench PRIVATE gemm2 nvbench::nvbench cublas)

target_compile_options(gemm2 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v --generate-line-info>)

add_executable(gemm_test gemm_test.cu test_utils.cu cache_utils.cu)
target_link_libraries(gemm_test PRIVATE gemm2 cublas)