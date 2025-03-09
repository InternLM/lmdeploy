cmake_minimum_required(VERSION 3.8)

project(_migration C CXX CUDA)

include(${PROJECT_SOURCE_DIR}/../../cmake/utils.cmake)
message(${PROJECT_SOURCE_DIR}/../../cmake/utils.cmake)
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG)
find_package(torch REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -I/usr/local/cuda/include)

if(NOT pybind11_FOUND)
    execute_process(COMMAND "pybind11-config" "--cmakedir"
                    RESULT_VARIABLE _COMMAND_SUCCESS
                    OUTPUT_VARIABLE pybind11_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 CONFIG)
endif()


include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/manager)
include_directories(${TORCH_INCLUDE_DIRS})

add_library(
    _migration
    manager/migration_manager.cpp
    manager/migration_manager.cu
)
target_link_libraries(_migration ${TORCH_LIBRARIES})

add_library(_C python/bind.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE migration_manager)

