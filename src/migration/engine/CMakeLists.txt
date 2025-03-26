add_library(
    _migration
    SHARED
    ibv_helper.cpp
    memory_pool.cpp
    rdma_transport.cpp
    transfer_engine.cpp
)

target_link_libraries(_migration PUBLIC _migration_ops ibverbs)
