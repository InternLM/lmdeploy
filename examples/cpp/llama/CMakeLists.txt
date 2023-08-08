# Copyright (c) OpenMMLab. All rights reserved.

add_executable(llama_triton_example llama_triton_example.cc)
target_link_libraries(llama_triton_example PUBLIC -lcublas -lcublasLt -lcudart
        LlamaTritonBackend TransformerTritonBackend mpi_utils nccl_utils
        nvtx_utils word_list -lpthread -ldl)

install(TARGETS llama_triton_example DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
