find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")

add_library(
    _slime_ops
    SHARED
    gather.cu
    scatter.cu
)

target_link_libraries(_slime_ops PUBLIC CUDA::cudart)
