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
    # kernel/sm90_16816_4.cu
    # kernel/sm90_16816_8.cu
    kernel/sm90_64n32_8.cu
    kernel/sm90_64n16_4.cu
    kernel/sm90_64n16_mxfp4.cu
    kernel/sm90_64n32_mxfp4_fp8_folded.cu
    # kernel/sm90_64n32_mxfp4_fp8_unfolded.cu
    kernel/sm90_64n16_nvfp4.cu
    kernel/sm90_64n16_e4m3.cu
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

set(GEMM2_HAS_SM90_MIXED 0)
if(GEMM2_SM90_ARCHS AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.3")
    set(GEMM2_HAS_SM90_MIXED 1)
endif()
message(STATUS "GEMM native SM90 mixed precision: ${GEMM2_HAS_SM90_MIXED}")

function(configure_gemm2_kernel_target target)
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ${target} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                    -Xptxas=-v
                    --generate-line-info>
    )
    target_link_libraries(${target} PRIVATE parser nvidia::cutlass::cutlass CUDA::cuda_driver)
endfunction()

# cublas.cu registers for every arch.
add_library(gemm2_kernels STATIC cublas.cu)
configure_gemm2_kernel_target(gemm2_kernels)
target_compile_options(gemm2_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--threads 16>
)

set(GEMM2_KERNEL_TARGETS gemm2_kernels)

if(GEMM2_SM70_ARCHS)
    add_library(gemm2_kernels_sm70 STATIC ${GEMM2_KERNELS_SM70})
    set_property(TARGET gemm2_kernels_sm70 PROPERTY
            CUDA_ARCHITECTURES "${GEMM2_SM70_ARCHS}")
    configure_gemm2_kernel_target(gemm2_kernels_sm70)
    target_compile_options(gemm2_kernels_sm70 PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--threads 64>
    )
    list(APPEND GEMM2_KERNEL_TARGETS gemm2_kernels_sm70)
endif()

if(GEMM2_SM75_ARCHS)
    add_library(gemm2_kernels_sm75 STATIC ${GEMM2_KERNELS_SM75})
    set_property(TARGET gemm2_kernels_sm75 PROPERTY
            CUDA_ARCHITECTURES "${GEMM2_SM75_ARCHS}")
    configure_gemm2_kernel_target(gemm2_kernels_sm75)
    target_compile_options(gemm2_kernels_sm75 PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--threads 64>
    )
    list(APPEND GEMM2_KERNEL_TARGETS gemm2_kernels_sm75)
endif()

if(GEMM2_SM80_ARCHS)
    add_library(gemm2_kernels_sm80 STATIC ${GEMM2_KERNELS_SM80})
    set_property(TARGET gemm2_kernels_sm80 PROPERTY
            CUDA_ARCHITECTURES "${GEMM2_SM80_ARCHS}")
    configure_gemm2_kernel_target(gemm2_kernels_sm80)
    target_compile_options(gemm2_kernels_sm80 PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--threads 64>
    )
    list(APPEND GEMM2_KERNEL_TARGETS gemm2_kernels_sm80)
endif()

if(GEMM2_SM90_ARCHS)
    # SM90 kernels only compile for 90/90a; avoid building them for sm_100.
    add_library(gemm2_sm90 STATIC ${GEMM2_KERNELS_SM90})
    set_target_properties(gemm2_sm90 PROPERTIES
        CUDA_ARCHITECTURES "${GEMM2_SM90_ARCHS}"
    )
    target_compile_definitions(gemm2_sm90 PRIVATE
        CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
        TM_GEMM_HAS_SM90_MIXED=${GEMM2_HAS_SM90_MIXED}
    )
    configure_gemm2_kernel_target(gemm2_sm90)
    target_compile_options(gemm2_sm90 PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                    -Xptxas=-Werror
                    --threads 64>
    )
    list(APPEND GEMM2_KERNEL_TARGETS gemm2_sm90)
endif()

add_library(gemm2_core
        family.cc
        weight_plan.cc
        gemm.cu
        kernel.cu
        registry.cu
        dispatch_cache.cu
        gpu_metric.cu
        convert_v3.cu
        sm90_mixed_pack.cu
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

target_link_libraries(gemm2_core PRIVATE parser nvidia::cutlass::cutlass CUDA::cuda_driver)
target_compile_definitions(gemm2_core PRIVATE TM_GEMM_HAS_SM90_MIXED=${GEMM2_HAS_SM90_MIXED})

# The registrar archives call back into gemm2_core, while gemm2_core needs every
# file-scope Registrar retained. Rescan the whole group to resolve that static
# dependency cycle without relying on consumer link order.
set(_gemm2_link_group "gemm2_core")
foreach(target IN LISTS GEMM2_KERNEL_TARGETS)
    string(APPEND _gemm2_link_group ",$<LINK_LIBRARY:WHOLE_ARCHIVE,${target}>")
endforeach()
add_library(gemm2 INTERFACE)
target_link_libraries(gemm2 INTERFACE "$<LINK_GROUP:RESCAN,${_gemm2_link_group}>")

# cublasGemmGroupedBatchedEx (CUDA 12.5+): grouped batched GEMM for MoE
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.5")
    target_compile_definitions(gemm2_kernels PRIVATE ENABLE_CUBLAS_GROUPED=1)
    message(STATUS "GEMM: ENABLE_CUBLAS_GROUPED=1 (cublasGemmGroupedBatchedEx for MoE)")
endif()

target_compile_options(gemm2_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
                -Xptxas=-v
                --generate-line-info
                --threads 16>
)
set_target_properties(gemm2_core PROPERTIES
        OUTPUT_NAME gemm2
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
