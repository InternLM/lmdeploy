find_package(CUDAToolkit REQUIRED)

add_library(
    _migration_ops
    SHARED
    gather.cu
    scatter.cu
)

target_link_libraries(_migration_ops PUBLIC CUDA::cudart)
