#pragma once

#include "config.hpp"

#include "gin_backend.h"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"
#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/tensor.h"

#include <cuda.h>

#include <cstdint>
#include <tuple>
#include <vector>

using turbomind::comm::HostComm;
using turbomind::comm::DeviceComm;
using turbomind::core::Tensor;
using turbomind::core::Tensor_;
using turbomind::core::Buffer_;

namespace shared_memory {

union MemHandleInner {
    cudaIpcMemHandle_t cuda_ipc_mem_handle;
    CUmemFabricHandle  cu_mem_fabric_handle;
};

struct MemHandle {
    MemHandleInner inner;
    size_t         size;
};

constexpr size_t HANDLE_SIZE = sizeof(MemHandle);

class SharedMemoryAllocator {
public:
    SharedMemoryAllocator(bool use_fabric);
    void malloc(void** ptr, size_t size);
    void free(void* ptr);
    void get_mem_handle(MemHandle* mem_handle, void* ptr);
    void open_mem_handle(void** ptr, MemHandle* mem_handle);
    void close_mem_handle(void* ptr);

private:
    bool use_fabric;
};
}  // namespace shared_memory

namespace deep_ep {

class Buffer {
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");

public:
    HostComm   h_comm;
    DeviceComm ipc_comm;
    int        num_sms{24};

    std::shared_ptr<internode::NCCLGINBackend> comm;

    // Low-latency mode buffer
    int  low_latency_buffer_idx = 0;
    bool low_latency_mode       = false;

    // NVLink Buffer
    int64_t num_nvl_bytes;
    void*   buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    void**  buffer_ptrs_gpu                = nullptr;

    // NVSHMEM Buffer
    int64_t num_rdma_bytes;
    int64_t num_ll_rdma_bytes;
    void*   rdma_buffer_ptr    = nullptr;
    void*   rdma_ll_buffer_ptr = nullptr;

    // Shrink mode buffer
    bool enable_shrink   = false;
    int* mask_buffer_ptr = nullptr;
    int* sync_buffer_ptr = nullptr;

    // Device info and communication
    int                      device_id;
    int                      num_device_sms;
    int                      rank, rdma_rank, nvl_rank;
    int                      num_ranks, num_rdma_ranks, num_nvl_ranks;
    int                      qps_per_rank;
    shared_memory::MemHandle ipc_handles[NUM_MAX_NVL_PEERS];

    // After IPC/NVSHMEM synchronization, this flag will be true
    bool available = false;

    // After `destroy()` be called, this flag will be true
    bool destroyed = false;

    // Barrier signals
    int*  barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    int** barrier_signal_ptrs_gpu                = nullptr;

    // Workspace
    void* workspace = nullptr;

    // Host-side MoE info
    volatile int* moe_recv_counter        = nullptr;
    int*          moe_recv_counter_mapped = nullptr;

    // Host-side expert-level MoE info
    volatile int* moe_recv_expert_counter        = nullptr;
    int*          moe_recv_expert_counter_mapped = nullptr;

    // Host-side RDMA-level MoE info
    volatile int* moe_recv_rdma_counter        = nullptr;
    int*          moe_recv_rdma_counter_mapped = nullptr;

    shared_memory::SharedMemoryAllocator shared_memory_allocator;

    Buffer(int      rank,  //
           int      num_ranks,
           int64_t  num_nvl_bytes,
           int64_t  num_rdma_bytes,
           int64_t  num_ll_rdma_bytes,
           bool     low_latency_mode,
           bool     enable_shrink,
           bool     use_fabric,
           int      qps_per_rank,
           HostComm h_comm);

    Buffer(): shared_memory_allocator{false} {};

    ~Buffer() = default;

    void allocate_sync_nvl_buffer();

    void allocate_rdma_buffer();

    bool is_available() const;

    bool is_internode_available() const;

    int get_num_rdma_ranks() const;

    int get_rdma_rank() const;

    int get_root_rdma_rank(bool global) const;

    int get_local_device_id() const;

    void destroy();

    std::tuple<Tensor, std::optional<Tensor>, Tensor, Tensor>  //
    get_dispatch_layout(const Tensor& topk_idx, int num_experts);

    std::tuple<Tensor,
               std::optional<Tensor>,
               std::optional<Tensor>,
               std::optional<Tensor>,
               std::vector<int>,
               Tensor,
               Tensor,
               Tensor,
               Tensor,
               Tensor,
               Tensor>
    intranode_dispatch(const Tensor&                x,
                       const std::optional<Tensor>& x_scales,
                       const std::optional<Tensor>& topk_idx,
                       const std::optional<Tensor>& topk_weights,
                       const std::optional<Tensor>& num_tokens_per_rank,
                       const Tensor&                is_token_in_rank,
                       const std::optional<Tensor>& num_tokens_per_expert,
                       int                          cached_num_recv_tokens,
                       const std::optional<Tensor>& cached_rank_prefix_matrix,
                       const std::optional<Tensor>& cached_channel_prefix_matrix,
                       int                          expert_alignment,
                       int                          num_worst_tokens,
                       const Config&                config);

    std::tuple<Tensor,  //
               std::optional<Tensor>>
    intranode_combine(const Tensor&                x,
                      const std::optional<Tensor>& topk_weights,
                      const std::optional<Tensor>& bias_0,
                      const std::optional<Tensor>& bias_1,
                      const Tensor&                src_idx,
                      const Tensor&                rank_prefix_matrix,
                      const Tensor&                channel_prefix_matrix,
                      Tensor&                      send_head,
                      const Config&                config);

    std::tuple<Tensor,  //
               std::optional<Tensor>,
               Tensor,
               Tensor,
               Tensor>
    low_latency_dispatch(const Tensor&                x,
                         const Tensor&                topk_idx,
                         const std::optional<Tensor>& cumulative_local_expert_recv_stats,
                         const std::optional<Tensor>& dispatch_wait_recv_cost_stats,
                         int                          num_max_dispatch_tokens_per_rank,
                         int                          num_experts,
                         bool                         use_fp8,
                         bool                         round_scale,
                         bool                         use_ue8m0);

    std::tuple<Tensor>  //
    low_latency_combine(const Tensor&                x,
                        const Tensor&                expert_offsets,
                        const Tensor&                topk_idx,
                        const Tensor&                topk_weights,
                        const Tensor&                src_info,
                        const Tensor&                layout_range,
                        const std::optional<Tensor>& combine_wait_recv_cost_stats,
                        int                          num_max_dispatch_tokens_per_rank,
                        int                          num_experts,
                        bool                         use_logfmt,
                        bool                         zero_copy,
                        const std::optional<Tensor>& out = std::nullopt);

    std::tuple<Tensor,
               std::optional<Tensor>,
               std::optional<Tensor>,
               std::optional<Tensor>,
               std::vector<int>,
               Tensor,
               Tensor,
               Tensor,
               std::optional<Tensor>,
               Tensor,
               std::optional<Tensor>,
               Tensor,
               std::optional<Tensor>,
               std::optional<Tensor>,
               std::optional<Tensor>>
    internode_dispatch(const Tensor&                x,
                       const std::optional<Tensor>& x_scales,
                       const std::optional<Tensor>& topk_idx,
                       const std::optional<Tensor>& topk_weights,
                       const std::optional<Tensor>& num_tokens_per_rank,
                       const std::optional<Tensor>& num_tokens_per_rdma_rank,
                       const Tensor&                is_token_in_rank,
                       const std::optional<Tensor>& num_tokens_per_expert,
                       int                          cached_num_recv_tokens,
                       int                          cached_num_rdma_recv_tokens,
                       const std::optional<Tensor>& cached_rdma_channel_prefix_matrix,
                       const std::optional<Tensor>& cached_recv_rdma_rank_prefix_sum,
                       const std::optional<Tensor>& cached_gbl_channel_prefix_matrix,
                       const std::optional<Tensor>& cached_recv_gbl_rank_prefix_sum,
                       int                          expert_alignment,
                       int                          num_worst_tokens,
                       const Config&                config);

    std::tuple<Tensor, std::optional<Tensor>>  //
    internode_combine(const Tensor&                x,
                      const std::optional<Tensor>& topk_weights,
                      const std::optional<Tensor>& bias_0,
                      const std::optional<Tensor>& bias_1,
                      const Tensor&                src_meta,
                      const Tensor&                is_combined_token_in_rank,
                      const Tensor&                rdma_channel_prefix_matrix,
                      const Tensor&                rdma_rank_prefix_sum,
                      const Tensor&                gbl_channel_prefix_matrix,
                      Tensor&                      combined_rdma_head,
                      Tensor&                      combined_nvl_head,
                      const Config&                config);

    Config get_dispatch_config();

    Config get_combine_config();
};

};  // namespace deep_ep
