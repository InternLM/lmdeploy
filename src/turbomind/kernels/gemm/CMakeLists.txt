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

# Bucket the selected archs by kernel family (ranges mirror arch.h).
set(GEMM2_SM70_ARCHS "")
set(GEMM2_SM75_ARCHS "")
set(GEMM2_SM80_ARCHS "")
set(GEMM2_SM90_ARCHS "")
foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(arch MATCHES "^([0-9]+)(a)?(-real)?$")
        set(arch_number "${CMAKE_MATCH_1}")
        if(arch_number GREATER_EQUAL 70 AND arch_number LESS 75)
            list(APPEND GEMM2_SM70_ARCHS "${arch}")
        elseif(arch_number GREATER_EQUAL 75 AND arch_number LESS 80)
            list(APPEND GEMM2_SM75_ARCHS "${arch}")
        elseif(arch_number GREATER_EQUAL 80 AND arch_number LESS 90)
            list(APPEND GEMM2_SM80_ARCHS "${arch}")
        elseif(arch_number EQUAL 90)
            list(APPEND GEMM2_SM90_ARCHS "${arch}")
        elseif(arch_number GREATER_EQUAL 100)
            # SM100/SM120 run the SM80 (s16816 mma.sync) kernels for now.
            list(APPEND GEMM2_SM80_ARCHS "${arch}")
        endif()
    endif()
endforeach()

if(NOT GEMM2_SM90_ARCHS)
    # When building for SM100+ without explicit SM90, still compile SM90 CUTLASS
    # kernels so the fat binary can run MoE models on H100 (CUTLASS fused path).
    set(_sm100_archs "${CMAKE_CUDA_ARCHITECTURES}")
    list(FILTER _sm100_archs INCLUDE REGEX "^100")
    if(_sm100_archs)
        set(GEMM2_SM90_ARCHS "90")
        message(STATUS "GEMM: auto-enabling SM90 CUTLASS kernels for H100 backward compatibility")
    endif()
endif()

message(STATUS "GEMM SM70 CUDA archs: ${GEMM2_SM70_ARCHS}")
message(STATUS "GEMM SM75 CUDA archs: ${GEMM2_SM75_ARCHS}")
message(STATUS "GEMM SM80 CUDA archs: ${GEMM2_SM80_ARCHS}")
message(STATUS "GEMM SM90 CUDA archs: ${GEMM2_SM90_ARCHS}")

function(configure_gemm2_kernel_target target)
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ${target} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                    -Xptxas=-v
                    --generate-line-info
                    --threads 16>
    )
    target_link_libraries(${target} PRIVATE parser nvidia::cutlass::cutlass CUDA::cuda_driver)
endfunction()

# cublas.cu registers for every arch.
add_library(gemm2_kernels STATIC cublas.cu)
configure_gemm2_kernel_target(gemm2_kernels)

set(GEMM2_KERNEL_TARGETS gemm2_kernels)

if(GEMM2_SM70_ARCHS)
    add_library(gemm2_kernels_sm70 STATIC ${GEMM2_KERNELS_SM70})
    set_property(TARGET gemm2_kernels_sm70 PROPERTY
            CUDA_ARCHITECTURES "${GEMM2_SM70_ARCHS}")
    configure_gemm2_kernel_target(gemm2_kernels_sm70)
    list(APPEND GEMM2_KERNEL_TARGETS gemm2_kernels_sm70)
endif()

if(GEMM2_SM75_ARCHS)
    add_library(gemm2_kernels_sm75 STATIC ${GEMM2_KERNELS_SM75})
    set_property(TARGET gemm2_kernels_sm75 PROPERTY
            CUDA_ARCHITECTURES "${GEMM2_SM75_ARCHS}")
    configure_gemm2_kernel_target(gemm2_kernels_sm75)
    list(APPEND GEMM2_KERNEL_TARGETS gemm2_kernels_sm75)
endif()

if(GEMM2_SM80_ARCHS)
    add_library(gemm2_kernels_sm80 STATIC ${GEMM2_KERNELS_SM80})
    set_property(TARGET gemm2_kernels_sm80 PROPERTY
            CUDA_ARCHITECTURES "${GEMM2_SM80_ARCHS}")
    configure_gemm2_kernel_target(gemm2_kernels_sm80)
    list(APPEND GEMM2_KERNEL_TARGETS gemm2_kernels_sm80)
endif()

if(GEMM2_SM90_ARCHS)
    # SM90 kernels only compile for 90/90a; avoid building them for sm_100.
    add_library(gemm2_sm90 STATIC ${GEMM2_KERNELS_SM90})
    set_target_properties(gemm2_sm90 PROPERTIES
        CUDA_ARCHITECTURES "${GEMM2_SM90_ARCHS}"
    )
    target_compile_definitions(gemm2_sm90 PRIVATE -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
    configure_gemm2_kernel_target(gemm2_sm90)
    list(APPEND GEMM2_KERNEL_TARGETS gemm2_sm90)
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
        tuner/cache_utils.cu
        tuner/measurer.cu
        tuner/sampler.cu
        tuner/stopping_criterion.cc
        tuner/params.cc
        moe_utils_v2.cu
        test/test_utils.cu
)

target_link_libraries(gemm2 PRIVATE parser nvidia::cutlass::cutlass CUDA::cuda_driver)

# Registry/kernel TUs are static plugins: they must be linked before libgemm2.a so
# Kernel base symbols (vtable/GetName) resolve when consumers link executables.
# INTERFACE_LINK_LIBRARIES_DIRECT places them ahead of gemm2 (same as linear_attn).
# Keep every archive in one WHOLE_ARCHIVE expression.
list(JOIN GEMM2_KERNEL_TARGETS "," GEMM2_KERNEL_TARGET_LIST)
set_property(TARGET gemm2 PROPERTY INTERFACE_LINK_LIBRARIES_DIRECT
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,${GEMM2_KERNEL_TARGET_LIST}>")

# cublasGemmGroupedBatchedEx (CUDA 12.5+): grouped batched GEMM for MoE on SM100
set(_archs_100 "${CMAKE_CUDA_ARCHITECTURES}")
list(FILTER _archs_100 INCLUDE REGEX "^100")
if(_archs_100 AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.5")
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
