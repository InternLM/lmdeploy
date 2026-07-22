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
set(LINEAR_ATTN_GDR_WHOLE_ARCHIVES "")
foreach(target IN LISTS LINEAR_ATTN_GDR_KERNEL_TARGETS)
    list(APPEND LINEAR_ATTN_GDR_WHOLE_ARCHIVES "$<LINK_LIBRARY:WHOLE_ARCHIVE,${target}>")
endforeach()
set_property(TARGET linear_attn PROPERTY INTERFACE_LINK_LIBRARIES_DIRECT
        ${LINEAR_ATTN_GDR_WHOLE_ARCHIVES})
