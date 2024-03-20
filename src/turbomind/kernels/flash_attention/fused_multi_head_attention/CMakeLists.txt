
cmake_minimum_required(VERSION 3.8)

add_library(llama_fmha STATIC llama_flash_attention_kernel.cu)
target_include_directories(llama_fmha PRIVATE ${CUTLASS_DIR}/examples)
target_link_libraries(llama_fmha PRIVATE nvidia::cutlass::cutlass)
set_property(TARGET llama_fmha PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET llama_fmha PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
