add_library(
    _migration
    SHARED
    transfer_engine.cpp
    rdma_transport.cpp
    ibv_helper.cpp
)

target_link_libraries(_migration PUBLIC _migration_ops ibverbs)
