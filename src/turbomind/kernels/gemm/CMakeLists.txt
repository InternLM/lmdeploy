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
    tma.cu
    kernel/sm90_16816_4.cu
    kernel/sm90_16816_8.cu
    kernel/sm90_64n32_8.cu
    kernel/sm90_64n16_16.cu
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

add_library(gemm2_kernels STATIC
        ${GEMM2_KERNELS_SM70}
        ${GEMM2_KERNELS_SM75}
        ${GEMM2_KERNELS_SM80}
        cublas.cu
)
set_property(TARGET gemm2_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET gemm2_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(gemm2_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
                -Xptxas=-v
                --generate-line-info
                --threads 16>
)
target_link_libraries(gemm2_kernels PRIVATE parser nvidia::cutlass::cutlass CUDA::cuda_driver)

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
        tuner/cache_utils.cu
        tuner/measurer.cu
        tuner/sampler.cu
        tuner/stopping_criterion.cc
        tuner/params.cc
        moe_utils_v2.cu
        test/test_utils.cu
)

target_link_libraries(gemm2 PRIVATE parser nvidia::cutlass::cutlass CUDA::cuda_driver)

# Retain file-scope Registrar objects (self-registration). Same pattern as attention_kernels.
target_link_libraries(gemm2 PUBLIC $<LINK_LIBRARY:WHOLE_ARCHIVE,gemm2_kernels>)

# cublasGemmGroupedBatchedEx (CUDA 12.5+): grouped batched GEMM for MoE on SM100
set(_has_sm100 FALSE)
set(_archs_100 "${CMAKE_CUDA_ARCHITECTURES}")
list(FILTER _archs_100 INCLUDE REGEX "^100")
if(_archs_100 AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.5")
    set(_has_sm100 TRUE)
    target_compile_definitions(gemm2_kernels PRIVATE ENABLE_CUBLAS_GROUPED=1)
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
    target_link_libraries(gemm2 PUBLIC $<LINK_LIBRARY:WHOLE_ARCHIVE,gemm2_sm90>)
endif()
