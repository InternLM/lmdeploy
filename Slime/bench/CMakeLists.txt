add_executable(
    transfer_bench
    transfer_bench.cpp
)

target_include_directories(transfer_bench PUBLIC ${ZeroMQ_INCLUDE_DIRS})

target_link_libraries(
    transfer_bench
    PUBLIC
    _slime gflags zmq
)
