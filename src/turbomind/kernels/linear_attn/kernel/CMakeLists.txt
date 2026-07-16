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

add_library(linear_attn_kernels STATIC
        plan.cc
        tma_desc.cu
        ${LINEAR_ATTN_GDR_PRE_SM90_KERNEL_SOURCES}
        ${LINEAR_ATTN_GDR_SM90_KERNEL_SOURCES}
        ${LINEAR_ATTN_GDR_SM120_KERNEL_SOURCES})

set_property(TARGET linear_attn_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET linear_attn_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(linear_attn_kernels PRIVATE
        core
        CUDA::cudart
        CUDA::cuda_driver
        nvidia::cutlass::cutlass)
