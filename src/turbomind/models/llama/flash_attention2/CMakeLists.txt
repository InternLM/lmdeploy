
cmake_minimum_required(VERSION 3.8)
project(flash_attention2)

add_library(${PROJECT_NAME} STATIC
    flash_api.cpp
    # flash_fwd_hdim32_fp16_sm80.cu
    # flash_fwd_hdim64_fp16_sm80.cu
    flash_fwd_hdim128_fp16_sm80.cu
    flash_fwd_hdim128_bf16_sm80.cu
    # flash_fwd_hdim256_fp16_sm80.cu
    )
target_include_directories(${PROJECT_NAME} PRIVATE ${CUTLASS_DIR} / include)
target_link_libraries(${PROJECT_NAME} PRIVATE nvidia::cutlass::cutlass)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
