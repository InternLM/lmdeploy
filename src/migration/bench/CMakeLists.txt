add_executable(
    transfer_bench
    transfer_bench.cpp
)

target_link_libraries(transfer_bench PUBLIC _migration gflags)
