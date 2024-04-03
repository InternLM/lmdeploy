cmake_minimum_required(VERSION 3.8)

add_subdirectory(fused_multi_head_attention)

add_library(flash_attention STATIC flash_attention.cu)
set_property(TARGET flash_attention PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET flash_attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(flash_attention PRIVATE llama_fmha)

if (NOT MSVC)
    add_subdirectory(flash_attention2)
    target_link_libraries(flash_attention PRIVATE flash_attention2)
endif()
