cmake_minimum_required(VERSION 3.21)

project(slime_extensions LANGUAGES CXX)

# option(SLIME_TARGET_DEVICE "Target device backend for Slime, CUDA Only" "cuda")
# option(NCCL_PATH "nccl path" "/usr/")

# message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
# message(STATUS "Target device: ${SLIME_TARGET_DEVICE}")

# include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# set(SLIME_TARGET_DEVICE "cuda")

# # Supported python versions
# set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11")

# # Supported NVIDIA architectures
# set(CMAKE_CUDA_ARCHITECTURES 75 80 90)
# set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")

# # Supported torch version for CUDA
# set(TORCH_SUPPORTED_VERSION_CUDA "2.5.1")

# # Find Python executable
# if (SLIME_PYTHON_EXECUTABLE)
#   find_python_from_executable(${SLIME_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
# else()
#   message(FATAL_ERROR "Please set SLIME_PYTHON_EXECUTABLE to the path of the desired python version")
# endif()

# # Setup CMAKE_PREFIX_PATH for torch
# append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

# # Verify nvcc is available
# find_program(NVCC_EXECUTABLE nvcc)
# if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
#     message(FATAL_ERROR "nvcc not found")
# endif()

# # Find torch package
# find_package(Torch REQUIRED)

# include_directories(${NCCL_PATH}/include/)

# # Find torch_python library
# find_library(torch_python_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# # Set GPU language to CUDA
# set(SLIME_GPU_LANG "CUDA")

# # Verify CUDA torch version
# if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
#   message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} expected for CUDA build, saw ${Torch_VERSION} instead.")
# endif()

# # Set GPU architectures
# override_gpu_arches(SLIME_GPU_ARCHES ${SLIME_GPU_LANG} "${CUDA_SUPPORTED_ARCHS}")

# # Get CUDA compiler flags
# get_torch_gpu_compiler_flags(SLIME_GPU_FLAGS ${SLIME_GPU_LANG})

# # Set nvcc parallelism
# if(NVCC_THREADS)
#   list(APPEND SLIME_GPU_FLAGS "--threads=${NVCC_THREADS}")
# endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /nvme1/majinming/miniconda3/envs/lmdeploy/lib/python3.12/site-packages/torch/share/cmake)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(pybind11)
add_subdirectory(Mooncake)

include_directories(csrc)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(Mooncake/mooncake-transfer-engine/include)

pybind11_add_module(_C
  "csrc/migration_manager.cu"
  "csrc/migration_manager.cpp"
  "csrc/lmdeploy_adaptor.cpp"
  "csrc/pybind.cpp"
)
target_link_libraries(_C PUBLIC transfer_engine ${TORCH_LIBRARIES})



