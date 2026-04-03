# Copyright (c) OpenMMLab. All rights reserved.

set(GEMM2_KERNELS_SM70
    kernel/sm70_884_4.cu
    kernel/sm70_884_8.cu
    kernel/sm70_884_16.cu
)
set(GEMM2_KERNELS_SM75
    kernel/sm75_16816_4.cu
    kernel/sm75_16816_8.cu
    kernel/sm75_16816_16.cu
)
set(GEMM2_KERNELS_SM80
    kernel/sm80_16816_4.cu
    kernel/sm80_16816_8.cu
    kernel/sm80_16816_16.cu
)
set(GEMM2_KERNELS_SM90
    kernel/sm90_16816_4.cu
    kernel/sm90_16816_8.cu
    kernel/sm90_16816_16.cu
    kernel/sm90_64n32_8.cu
)

set(GEMM2_ARCH_90_ENABLED FALSE)
set(_sm90_archs "${CMAKE_CUDA_ARCHITECTURES}")
list(FILTER _sm90_archs INCLUDE REGEX "^90")
if(_sm90_archs)
    set(GEMM2_ARCH_90_ENABLED TRUE)
else()
    # When building for SM100+ without explicit SM90, still compile SM90 CUTLASS
    # kernels so the fat binary can run MoE models on H100 (CUTLASS fused path).
    set(_sm100_archs "${CMAKE_CUDA_ARCHITECTURES}")
    list(FILTER _sm100_archs INCLUDE REGEX "^100")
    if(_sm100_archs)
        set(GEMM2_ARCH_90_ENABLED TRUE)
        set(_sm90_archs "90")
        message(STATUS "GEMM: auto-enabling SM90 CUTLASS kernels for H100 backward compatibility")
    endif()
endif()

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
        ${GEMM2_KERNELS_SM70}
        ${GEMM2_KERNELS_SM75}
        ${GEMM2_KERNELS_SM80}
        cublas.cu
        moe_utils_v2.cu
        test/test_utils.cu
)

target_link_libraries(gemm2 PRIVATE parser nvidia::cutlass::cutlass CUDA::cuda_driver)

# cublasGemmGroupedBatchedEx (CUDA 12.5+): grouped batched GEMM for MoE on SM100
set(_has_sm100 FALSE)
set(_archs_100 "${CMAKE_CUDA_ARCHITECTURES}")
list(FILTER _archs_100 INCLUDE REGEX "^100")
if(_archs_100 AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.5")
    set(_has_sm100 TRUE)
    target_compile_definitions(gemm2 PRIVATE ENABLE_CUBLAS_GROUPED=1)
    message(STATUS "GEMM: ENABLE_CUBLAS_GROUPED=1 (cublasGemmGroupedBatchedEx for MoE on SM100)")
endif()

target_compile_options(gemm2 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
                -Xptxas=-v
                --generate-line-info
                --threads 16>
)
set_property(TARGET gemm2 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET gemm2 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(GEMM2_ARCH_90_ENABLED)
    # SM90 kernels only compile for 90/90a; avoid building them for sm_100.
    add_library(gemm2_sm90 STATIC ${GEMM2_KERNELS_SM90})
    set_target_properties(gemm2_sm90 PROPERTIES
        CUDA_ARCHITECTURES "${_sm90_archs}"
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_compile_definitions(gemm2_sm90 PRIVATE -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
    target_compile_options(gemm2_sm90 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xptxas=-v
            --generate-line-info
            --threads 16>
    )
    target_link_libraries(gemm2_sm90 PRIVATE parser nvidia::cutlass::cutlass CUDA::cuda_driver)
    target_link_libraries(gemm2 PRIVATE gemm2_sm90)

    target_compile_definitions(gemm2 PRIVATE GEMM2_ARCH_90_ENABLED)
endif()

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
