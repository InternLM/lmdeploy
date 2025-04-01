#include "engine/config.h"
#include "engine/rdma_transport.h"
#include "utils/json.hpp"
#include "utils/logging.h"

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <future>
#include <gflags/gflags.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <zmq.h>
#include <zmq.hpp>

#include <cstdlib>

using json = nlohmann::json;
using namespace slime;

#define TERMINATE 0

DEFINE_string(mode, "target", "initiator or target");

DEFINE_string(device_name, "mlx5_bond_0", "device name");
DEFINE_uint32(ib_port, 1, "device name");
DEFINE_string(link_type, "Ethernet", "IB or Ethernet");

DEFINE_string(local_endpoint, "", "local endpoint");
DEFINE_string(remote_endpoint, "", "remote endpoint");

DEFINE_uint64(buffer_size, 1ull << 30, "total size of data buffer");
DEFINE_uint64(block_size, 32768, "block size");
DEFINE_uint64(batch_size, 80, "batch size");

DEFINE_uint64(duration, 10, "duration (s)");

json mr_info;

void* memory_allocate()
{
    SLIME_ASSERT(FLAGS_buffer_size > FLAGS_batch_size * FLAGS_block_size, "buffer_size < batch_size * block_size");
    void* data = (void*)malloc(FLAGS_buffer_size);
    return data;
}

int connect(RDMAContext& rdma_context, zmq::socket_t& send, zmq::socket_t& recv)
{
    json local_info = rdma_context.local_info();

    zmq::message_t local_msg(local_info.dump());
    send.send(local_msg, zmq::send_flags::none);

    zmq::message_t remote_msg;
    recv.recv(remote_msg, zmq::recv_flags::none);
    std::string remote_msg_str(static_cast<const char*>(remote_msg.data()), remote_msg.size());

    json remote_info = json::parse(remote_msg_str);

    rdma_context.modify_qp_to_rtsr(RDMAInfo(remote_info["rdma_info"]));
    for (auto& item : remote_info["mr_info"].items()) {
        rdma_context.register_remote_memory_region(item.key(), item.value());
    }

    return 0;
}

int target(RDMAContext& rdma_context)
{
    zmq::context_t context(2);
    zmq::socket_t  send(context, ZMQ_PUSH);
    zmq::socket_t  recv(context, ZMQ_PULL);

    send.connect("tcp://" + FLAGS_remote_endpoint);
    recv.bind("tcp://" + FLAGS_local_endpoint);

    rdma_context.init_rdma_context(FLAGS_device_name, FLAGS_ib_port, FLAGS_link_type);

    void* data = memory_allocate();
    rdma_context.register_memory_region("buffer", (uintptr_t)data, FLAGS_buffer_size);

    SLIME_ASSERT_EQ(connect(rdma_context, send, recv), 0, "Connect Error");

    zmq::message_t term_msg;
    recv.recv(term_msg, zmq::recv_flags::none);
    std::string signal = std::string(static_cast<char*>(term_msg.data()), term_msg.size());
    SLIME_ASSERT(!strcmp(signal.c_str(), "TERMINATE"), "signal error");

    return 0;
}

int initiator(RDMAContext& rdma_context)
{
    zmq::context_t context(2);
    zmq::socket_t  send(context, ZMQ_PUSH);
    zmq::socket_t  recv(context, ZMQ_PULL);

    send.connect("tcp://" + FLAGS_remote_endpoint);
    recv.bind("tcp://" + FLAGS_local_endpoint);

    rdma_context.init_rdma_context(FLAGS_device_name, FLAGS_ib_port, FLAGS_link_type);

    void* data = memory_allocate();
    rdma_context.register_memory_region("buffer", (uintptr_t)data, FLAGS_buffer_size);

    SLIME_ASSERT_EQ(connect(rdma_context, send, recv), 0, "Connect Error");

    rdma_context.launch_cq_future();

    // 新增变量：统计相关
    uint64_t total_bytes = 0;
    uint64_t total_trips = 0;
    size_t   step        = 0;
    auto     start_time  = std::chrono::steady_clock::now();
    auto     deadline    = start_time + std::chrono::seconds(FLAGS_duration);

    while (std::chrono::steady_clock::now() < deadline) {

        std::vector<uintptr_t> target_offsets, source_offsets;

        for (int i = 0; i < FLAGS_batch_size; ++i) {
            source_offsets.emplace_back(i * FLAGS_block_size);
            target_offsets.emplace_back(i * FLAGS_block_size);
        }

        int done = false;
        rdma_context.batch_r_rdma_async(
            "buffer", target_offsets, source_offsets, FLAGS_block_size, [&done](int code) { done = true; });

        while (!done) {}
        total_bytes += FLAGS_batch_size * FLAGS_block_size;
        total_trips += 1;
    }

    auto   end_time   = std::chrono::steady_clock::now();
    double duration   = std::chrono::duration<double>(end_time - start_time).count();
    double throughput = total_bytes / duration / (1 << 20);  // MB/s

    std::cout << "Batch size        : " << FLAGS_batch_size << std::endl;
    std::cout << "Block size        : " << FLAGS_block_size << std::endl;

    std::cout << "Total trips       : " << total_trips << std::endl;
    std::cout << "Total transferred : " << total_bytes / (1 << 20) << " MiB" << std::endl;
    std::cout << "Duration          : " << duration << " seconds" << std::endl;
    std::cout << "Average Latency   : " << duration / total_trips * 1000 << " ms/trip" << std::endl;
    std::cout << "Throughput        : " << throughput << " MiB/s" << std::endl;

    zmq::message_t term_msg("TERMINATE");
    send.send(term_msg, zmq::send_flags::none);

    rdma_context.stop_cq_future();

    return 0;
}

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    RDMAContext context;
    if (FLAGS_mode == "initiator") {
        return initiator(context);
    }
    else if (FLAGS_mode == "target") {
        return target(context);
    }
    SLIME_ABORT("Unsupported mode: must be 'initiator' or 'target'");
}
