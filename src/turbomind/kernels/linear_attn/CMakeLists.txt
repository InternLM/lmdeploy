add_subdirectory(kernel)

add_library(linear_attn STATIC
        delta_rule.cu
        registry.cu)

set_property(TARGET linear_attn PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET linear_attn PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(linear_attn PUBLIC
        core
        CUDA::cudart
        CUDA::cuda_driver
        nvidia::cutlass::cutlass)

# Registry entry translation units act as static plugins: they must be linked
# before the host archive whose planner and registry symbols they reference.
# Keep every archive in one feature expression: some CMake versions only apply
# WHOLE_ARCHIVE to the first of several expressions in this direct-link list.
list(JOIN LINEAR_ATTN_GDR_KERNEL_TARGETS "," LINEAR_ATTN_GDR_KERNEL_TARGET_LIST)
set_property(TARGET linear_attn PROPERTY INTERFACE_LINK_LIBRARIES_DIRECT
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LINEAR_ATTN_GDR_KERNEL_TARGET_LIST}>")
