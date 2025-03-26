#pragma once

#include "engine/config.h"

#include <cstdint>
#include <functional>
#include <future>
#include <infiniband/verbs.h>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace migration {

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
    int64_t registerMemoryRegion(std::string mem_key, int64_t addr, size_t length);

    /* Async RDMA Read */
    int64_t r_rdma_async(uintptr_t                         target_addr,
                         uintptr_t                         source_addr,
                         uint64_t                          length,
                         std::string                       mr_key,
                         int64_t                           remote_rkey,
                         std::function<void(unsigned int)> callback);

    int64_t batch_r_rdma_async(const std::vector<uintptr_t>&     target_addrs,
                               const std::vector<uintptr_t>&     source_addrs,
                               uint64_t                          length,
                               std::string                       mr_key,
                               int64_t                           remote_key,
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

    /* Add a memory pool management */
    uint32_t getLKey(std::string mr_key)
    {
        return memory_region_[mr_key]->lkey;
    }
    uint32_t getRKey(std::string mr_key)
    {
        return memory_region_[mr_key]->rkey;
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

    /* TODO: Memory Pool */
    std::unordered_map<std::string, ibv_mr*> memory_region_;

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