# Copyright (c) OpenMMLab. All rights reserved.

add_library(gemm2
        gemm.cu
        kernel.cu
        registry.cu
        dispatch_cache.cu
        gpu_metric.cu
        convert_v3.cu
        cast.cu
        unpack.cu
        context.cu
        tma.cu
        tuner/cache_utils.cu
        tuner/measurer.cu
        tuner/sampler.cu
        tuner/stopping_criterion.cc
        tuner/params.cc
        kernel/f16_u4g128_f16_tnt_sm90_s16816.cu
        kernel/f16_u4g128_f16_tnt_sm80_s16816.cu
        kernel/f16_u4g128_f16_tnt_sm75_s16816.cu
        kernel/f16_u4g128_f16_tnt_sm70_s884.cu
        kernel/sm90_mxfp4.cu
        kernel/sm80_mxfp4.cu
        kernel/sm70_s884_dynamic.cu
        kernel/sm75_s16816_dynamic.cu
        kernel/sm80_s16816_dynamic.cu
        kernel/sm90_s16816_dynamic.cu
        kernel/sm90_q64n32.cu
        cublas.cu
        moe_utils_v2.cu
        test/test_utils.cu
)

target_link_libraries(gemm2 PRIVATE parser nvidia::cutlass::cutlass CUDA::cuda_driver)


target_compile_definitions(gemm2 PRIVATE -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)

target_compile_options(gemm2 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
                -Xptxas=-v
                --generate-line-info
                --threads 16>
)
set_property(TARGET gemm2 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET gemm2 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)



if (BUILD_TEST)

        add_executable(test_gemm_v2
                test/test_gemm_v2.cc
                ../../models/llama/LlamaLinear.cu
                ../../models/llama/LlamaDenseWeight.cc
                test/reference.cu)
        target_link_libraries(test_gemm_v2 PRIVATE gemm2 core cublas quantization_kernels gpt_kernels)

        add_executable(test_moe_utils test/test_moe_utils.cu test/test_utils.cu)
        target_link_libraries(test_moe_utils PRIVATE gemm2 core cublas)

        # if (NOT MSVC)
        #         FetchContent_Declare(
        #         repo-nvbench
        #         GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
        #         GIT_TAG        d8dced8a64d9ce305add92fa6d274fd49b569b7e
        #         )

        #         set(NVBench_ENABLE_EXAMPLES OFF)
        #         set(NVBench_ENABLE_TESTING OFF)
        #         set(BUILD_SHARED_LIBS OFF)

        #         FetchContent_MakeAvailable(repo-nvbench)

        #         add_executable(gemm_bench
        #                 test/gemm_bench.cu
        #                 # test/test_utils.cu
        #                 test/quantization.cu
        #                 test/reference.cu)
        #         target_link_libraries(gemm_bench PRIVATE gemm2 core nvbench::nvbench cublas)
        # endif ()
endif ()
