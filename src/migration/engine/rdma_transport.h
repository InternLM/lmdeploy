#pragma once

#include "engine/config.h"
#include "engine/memory_pool.h"
#include "utils/json.hpp"

#include <cstdint>
#include <functional>
#include <future>
#include <infiniband/verbs.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace migration {

using json = nlohmann::json;

class RDMAContext {
public:
    /*
      A link of rdma QP.
    */
    RDMAContext() {}

    ~RDMAContext() {}

    /* Initialize */
    int64_t init_rdma_context(std::string dev_name, uint8_t ib_port, std::string link_type);

    /* RDMA Info Exchange */
    void modify_qp_to_rtsr(RDMAInfo remote_rdma_info);

    /* Memory Allocation */
    int64_t register_memory(std::string mr_key, uintptr_t data_ptr, size_t length)
    {
        memory_pool_.register_memory_region(mr_key, data_ptr, length);
        return 0;
    }

    int64_t register_remote_memory(std::string mr_key, json mr_info)
    {
        memory_pool_.register_remote_memory_region(mr_key, mr_info);
        return 0;
    }

    /* Async RDMA Read */
    int64_t r_rdma_async(uint64_t                         target_addr,
                         uint64_t                         source_addr,
                         uint64_t                          length,
                         std::string                       mr_key,
                         std::function<void(unsigned int)> callback);

    int64_t batch_r_rdma_async(const std::vector<uint64_t>&     target_addrs,
                               const std::vector<uint64_t>&     source_addrs,
                               uint64_t                          length,
                               std::string                       mr_key,
                               std::function<void(unsigned int)> callback);

    /* Completion Queue Polling */
    void cq_poll_handle();

    void launch_cq_future();
    void stop_cq_future();

    rdma_info_t get_local_rdma_info()
    {
        return local_rdma_info_;
    }
    rdma_info_t get_remote_rdma_info()
    {
        return remote_rdma_info_;
    }

    json exchange_info()
    {
        return json{{"rdma_info", local_rdma_info_.to_json()}, {"mr_info", memory_pool_.mr_info()}};
    }

private:
    std::string device_name_ = "";

    /* RDMA Configuration */
    struct ibv_context*      ib_ctx_;
    struct ibv_pd*           pd_           = nullptr;
    struct ibv_comp_channel* comp_channel_ = nullptr;
    struct ibv_cq*           cq_           = nullptr;
    struct ibv_qp*           qp_           = nullptr;
    uint8_t                  ib_port_      = -1;

    MemoryPool memory_pool_;

    /* RDMA Exchange Information */
    rdma_info_t remote_rdma_info_;
    rdma_info_t local_rdma_info_;

    /* State Management */
    bool              initialized_ = false;
    bool              connected_   = false;
    std::atomic<int>  outstanding_rdma_reads_{0};
    std::atomic<bool> stop_{false};

    /* Send Mutex */
    std::mutex rdma_post_send_mutex_;

    /* async cq handler */
    std::future<void> cq_future_;

    std::mutex mutex_;
};

}  // namespace migration
