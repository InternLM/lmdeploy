find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")

add_library(
    _slime_utils
    SHARED
    ibv_helper.cpp
    utils.cpp
)

target_link_libraries(_slime_utils PUBLIC ibverbs)
