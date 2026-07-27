set(LINEAR_ATTN_GDR_PRE_SM90_KERNEL_SOURCES
        pre_sm90/entry.cu
        pre_sm90/plan.cc
        pre_sm90/recurrent.cu
        pre_sm90/chunked.cu)

set(LINEAR_ATTN_GDR_SM90_KERNEL_SOURCES
        sm_90/entry.cu
        sm_90/plan.cc
        sm_90/tma_desc_prepare.cu
        sm_90/kkt_solve.cu
        sm_90/recurrent.cu)

set(LINEAR_ATTN_GDR_SM120_KERNEL_SOURCES
        sm_120/entry.cu
        sm_120/plan.cc
        sm_120/cumsum.cu
        sm_120/tma_desc_prepare.cu
        sm_120/kkt_solve.cu
        sm_120/recurrent.cu)

set(LINEAR_ATTN_GDR_PRE_SM90_ARCHS "")
set(LINEAR_ATTN_GDR_SM90_ARCHS "")
set(LINEAR_ATTN_GDR_SM100_PLUS_ARCHS "")
foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(arch MATCHES "^([0-9]+)(a)?(-real)?$")
        set(arch_number "${CMAKE_MATCH_1}")
        if(arch_number LESS 90)
            list(APPEND LINEAR_ATTN_GDR_PRE_SM90_ARCHS "${arch}")
        elseif(arch_number EQUAL 90)
            list(APPEND LINEAR_ATTN_GDR_SM90_ARCHS "${arch}")
        elseif(arch_number GREATER_EQUAL 100 AND arch_number LESS 1000)
            list(APPEND LINEAR_ATTN_GDR_SM100_PLUS_ARCHS "${arch}")
        endif()
    endif()
endforeach()

message(STATUS "GDR pre-SM90 CUDA archs: ${LINEAR_ATTN_GDR_PRE_SM90_ARCHS}")
message(STATUS "GDR SM90 CUDA archs: ${LINEAR_ATTN_GDR_SM90_ARCHS}")
message(STATUS "GDR SM100+ CUDA archs: ${LINEAR_ATTN_GDR_SM100_PLUS_ARCHS}")

function(configure_linear_attn_kernel_target target)
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ${target} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(${target} PRIVATE
            core
            CUDA::cudart
            CUDA::cuda_driver
            nvidia::cutlass::cutlass)
endfunction()

add_library(linear_attn_kernels STATIC
        plan.cc
        tma_desc.cu)
configure_linear_attn_kernel_target(linear_attn_kernels)

set(LINEAR_ATTN_GDR_KERNEL_TARGETS "")

if(LINEAR_ATTN_GDR_PRE_SM90_ARCHS)
    add_library(linear_attn_kernels_pre_sm90 STATIC ${LINEAR_ATTN_GDR_PRE_SM90_KERNEL_SOURCES})
    set_property(TARGET linear_attn_kernels_pre_sm90 PROPERTY
            CUDA_ARCHITECTURES "${LINEAR_ATTN_GDR_PRE_SM90_ARCHS}")
    configure_linear_attn_kernel_target(linear_attn_kernels_pre_sm90)
    list(APPEND LINEAR_ATTN_GDR_KERNEL_TARGETS linear_attn_kernels_pre_sm90)
endif()

if(LINEAR_ATTN_GDR_SM90_ARCHS)
    add_library(linear_attn_kernels_sm90 STATIC ${LINEAR_ATTN_GDR_SM90_KERNEL_SOURCES})
    set_property(TARGET linear_attn_kernels_sm90 PROPERTY
            CUDA_ARCHITECTURES "${LINEAR_ATTN_GDR_SM90_ARCHS}")
    configure_linear_attn_kernel_target(linear_attn_kernels_sm90)
    list(APPEND LINEAR_ATTN_GDR_KERNEL_TARGETS linear_attn_kernels_sm90)
endif()

if(LINEAR_ATTN_GDR_SM100_PLUS_ARCHS)
    add_library(linear_attn_kernels_sm120 STATIC ${LINEAR_ATTN_GDR_SM120_KERNEL_SOURCES})
    set_property(TARGET linear_attn_kernels_sm120 PROPERTY
            CUDA_ARCHITECTURES "${LINEAR_ATTN_GDR_SM100_PLUS_ARCHS}")
    configure_linear_attn_kernel_target(linear_attn_kernels_sm120)
    list(APPEND LINEAR_ATTN_GDR_KERNEL_TARGETS linear_attn_kernels_sm120)
endif()

# Keep family plugins before the common planner archive in the final link.
list(APPEND LINEAR_ATTN_GDR_KERNEL_TARGETS linear_attn_kernels)
set(LINEAR_ATTN_GDR_KERNEL_TARGETS "${LINEAR_ATTN_GDR_KERNEL_TARGETS}" PARENT_SCOPE)
