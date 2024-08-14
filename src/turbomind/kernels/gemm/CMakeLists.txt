# Copyright (c) OpenMMLab. All rights reserved.

add_library(gemm2
        gemm.cu
        kernel.cu
        registry.cu
        dispatch_cache.cu
        gpu_metric.cu
        convert_v2.cu
        cast.cu
        unpack.cu
        tuner/cache_utils.cu
        tuner/measurer.cu
        tuner/sampler.cu
        tuner/stopping_criterion.cc
        tuner/params.cc
        kernel/f16_u4g128_f16_tnt_sm90_s16816.cu
        kernel/f16_u4g128_f16_tnt_sm80_s16816.cu
        kernel/f16_u4g128_f16_tnt_sm75_s16816.cu
        kernel/f16_u4g128_f16_tnt_sm70_s884.cu
        kernel/f16_u4g128_f16_tnt_sm75_simt.cu
        kernel/u4g128_f16_f16_nnn_sm80_s16816.cu
)

target_link_libraries(gemm2 PRIVATE parser)


target_compile_options(gemm2 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
                -Xptxas=-v
                --generate-line-info
                --threads 8>
)
set_property(TARGET gemm2 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET gemm2 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if (BUILD_TEST)
        add_executable(gemm_test
                test/gemm_test.cu
                test/test_utils.cu
                test/quantization.cu
                test/reference.cu)
        target_link_libraries(gemm_test PRIVATE gemm2 cublas)

        if (NOT MSVC)
                FetchContent_Declare(
                repo-nvbench
                GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
                GIT_TAG        d8dced8a64d9ce305add92fa6d274fd49b569b7e
                )

                set(NVBench_ENABLE_EXAMPLES OFF)
                set(BUILD_SHARED_LIBS OFF)

                FetchContent_MakeAvailable(repo-nvbench)

                add_executable(gemm_bench
                        test/gemm_bench.cu
                        test/test_utils.cu
                        test/quantization.cu
                        test/reference.cu)
                target_link_libraries(gemm_bench PRIVATE gemm2 nvbench::nvbench cublas)
        endif ()
endif ()
