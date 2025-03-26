#include "engine/config.h"
#include "engine/rdma_transport.h"
#include "utils/json.hpp"
#include "utils/logging.h"

#include <future>
#include <gflags/gflags.h>
#include <stdexcept>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>
#include <zmq.h>
#include <zmq.hpp>

#include <cstdlib>

using json = nlohmann::json;
using namespace migration;

DEFINE_string(mode, "target", "initiator or target");

DEFINE_string(device_name, "mlx5_bond_0", "device name");
DEFINE_uint32(ib_port, 1, "device name");
DEFINE_string(link_type, "Ethernet", "IB or Ethernet");

DEFINE_string(local_endpoint, "", "local endpoint");
DEFINE_string(remote_endpoint, "", "remote endpoint");

DEFINE_uint64(buffer_size, 1ull << 30, "total size of data buffer");
DEFINE_uint64(block_size, 32768, "block size");
DEFINE_uint64(batch_size, 80, "batch size");

json mr_info;

void* memory_allocate()
{
    MIGRATION_ASSERT(FLAGS_buffer_size > FLAGS_batch_size * FLAGS_block_size, "buffer_size < batch_size * block_size");
    void* data = (void*)malloc(FLAGS_buffer_size);
    return data;
}

int connect(RDMAContext& rdma_context)
{
    zmq::context_t context(2);
    zmq::socket_t  send(context, ZMQ_PUSH);
    zmq::socket_t  recv(context, ZMQ_PULL);

    send.connect("tcp://" + FLAGS_remote_endpoint);
    recv.bind("tcp://" + FLAGS_local_endpoint);

    json local_info = rdma_context.exchange_info();

    zmq::message_t local_msg(local_info.dump());
    send.send(local_msg, zmq::send_flags::none);

    zmq::message_t remote_msg;
    recv.recv(remote_msg, zmq::recv_flags::none);

    const char* data = static_cast<const char*>(remote_msg.data());
    size_t      size = remote_msg.size();
    std::string exchange_message(data, size);

    json exchange_info = json::parse(exchange_message);
    std::cout << exchange_info.dump() << std::endl;

    rdma_context.modify_qp_to_rtsr(RDMAInfo(exchange_info["rdma_info"]));
    mr_info = exchange_info["mr_info"];

    return 0;
}

int target(RDMAContext& rdma_context)
{
    rdma_context.init_rdma_context(FLAGS_device_name, FLAGS_ib_port, FLAGS_link_type);
    void* data = memory_allocate();
    rdma_context.register_memory("buffer", (uintptr_t)data, FLAGS_buffer_size);

    MIGRATION_ASSERT_EQ(connect(rdma_context), 0, "Connect Error");
    while (true)
        sleep(1);

    return 0;
}

int initiator(RDMAContext& rdma_context)
{
    rdma_context.init_rdma_context(FLAGS_device_name, FLAGS_ib_port, FLAGS_link_type);
    void* data = memory_allocate();
    rdma_context.register_memory("buffer", (uintptr_t)data, FLAGS_buffer_size);

    MIGRATION_ASSERT_EQ(connect(rdma_context), 0, "Connect Error");

    std::vector<uintptr_t> target_addrs, source_addrs;

    for (int i = 0; i < FLAGS_batch_size; ++i) {
        source_addrs.emplace_back((uintptr_t)data + i * FLAGS_block_size);
        target_addrs.emplace_back((uintptr_t)data + i * FLAGS_block_size);
    }

    // TODO: Add a future event for asynchronization
    throw std::runtime_error("UnimplementedError");

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
    MIGRATION_ABORT("Unsupported mode: must be 'initiator' or 'target'");
}
