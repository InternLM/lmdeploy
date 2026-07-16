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
set_property(TARGET linear_attn PROPERTY INTERFACE_LINK_LIBRARIES_DIRECT
        $<LINK_LIBRARY:WHOLE_ARCHIVE,linear_attn_kernels>)
