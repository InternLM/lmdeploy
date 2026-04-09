#pragma once

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <vector>

#define DEEP_EP_GIN_MAX_CONTEXTS 32
#define DEEP_EP_NCCL_GIN_CTXS_PER_COMM 4
#define DEEP_EP_NCCL_MAX_NUM_CHANNELS 32  // Max number of local experts per GPU

namespace deep_ep {
namespace internode {

struct NcclGinMemHandle {
    void* ptr = nullptr;
};

class NCCLGINBackend {
public:
    NCCLGINBackend(): initialized_(false), rank_(-1), num_ranks_(-1) {}

    ~NCCLGINBackend();

    // Required interface methods
    int init(const std::vector<uint8_t>& root_unique_id_val,
             int                         rank,
             int                         num_ranks,
             bool                        low_latency_mode,
             int                         qps_per_rank);

    void finalize();
    void barrier();

    // Memory management interface methods
    void* alloc(size_t size, size_t alignment);
    void  register_memory(void* ptr, size_t size);  // NCCL-specific: register allocated memory with communicators
    void  free(void* ptr);

    int get_rank() const;
    int get_num_ranks() const;

    // NCCL-specific methods
    bool is_p2p_disabled() const;

    // NCCL specific methods
    unsigned get_signals_base(int buffer_idx, bool low_latency_mode) const;

    // Device arrays for kernels
    ncclWindow_t get_device_nccl_window(void* ptr);
    ncclDevComm  get_device_communicator(bool low_latency_mode) const;

private:
    bool initialized_  = false;
    bool p2p_disabled_ = false;  // True if P2P/NVLink is disabled
    int  rank_         = -1;     // Global rank (for external API)
    int  num_ranks_    = -1;     // Global num_ranks (for external API)
    int  comm_rank_    = -1;     // Rank within NCCL communicator
    int  comm_nranks_  = -1;     // Number of ranks in NCCL communicator

    ncclComm_t nccl_comm_;

    ncclDevComm_t dev_ht_comm_{};
    ncclDevComm_t dev_ll_comm_{};

    std::unordered_map<void*, ncclWindow_t> wins_;
    std::unordered_map<void*, size_t>       buffers_;

    // GIN signal management
    int num_ht_signals_ = 0;
    int num_ll_signals_ = 0;

    // GIN barriers -- assume 32 rdma ranks
    const int MAX_BARRIER_SESSIONS = 32;

    // Barrier variable
    int* d_barrier_var_ = nullptr;
};

}  // namespace internode
}  // namespace deep_ep
