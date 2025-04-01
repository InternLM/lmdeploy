add_library(
    _slime
    SHARED
    memory_pool.cpp
    rdma_transport.cpp
)

target_link_libraries(_slime PUBLIC _slime_ops _slime_utils ibverbs)
