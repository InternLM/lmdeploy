#include "src/turbomind/comm/nccl/deep_ep/gin_backend.h"

#include "src/turbomind/comm/nccl/deep_ep/kernels/configs.cuh"
#include "src/turbomind/comm/nccl/deep_ep/kernels/exception.cuh"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/utils/logger.h"

#include <cstring>

namespace deep_ep {
namespace internode {

NCCLGINBackend::~NCCLGINBackend()
{
    if (initialized_) {
        finalize();
    }
}

int NCCLGINBackend::init(
    const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode, int qps_per_rank)
{
    if (initialized_) {
        return rank_;
    }
    TM_CHECK_EQ(low_latency_mode, true);  // compatible with low latency mode

    // Check if P2P/NVLink is disabled via environment variable
    const char* nccl_disable_p2p = std::getenv("NCCL_P2P_DISABLE");
    p2p_disabled_                = (nccl_disable_p2p != nullptr && std::string(nccl_disable_p2p) == "1");

    // Determine communication topology based on mode
    const int gpus_per_server = NUM_MAX_NVL_PEERS;
    int       comm_rank;        // Rank to use for NCCL initialization
    int       comm_nranks;      // Number of ranks in communicator
    int       color      = -1;  // Symmetric group ID (only for high throughput mode)
    int       group_rank = -1;  // Rank within symmetric group

    if (low_latency_mode) {
        // LOW LATENCY MODE: Connect to all ranks
        comm_rank   = rank;
        comm_nranks = num_ranks;
    }
    else {
        // HIGH THROUGHPUT MODE: Connect only to symmetric RDMA ranks
        color       = rank % gpus_per_server;
        group_rank  = rank / gpus_per_server;
        comm_nranks = (num_ranks + gpus_per_server - 1) / gpus_per_server;
        comm_rank   = group_rank;
    }

    size_t single_id_size = sizeof(ncclUniqueId);
    size_t expected_ids   = gpus_per_server;
    EP_HOST_ASSERT(root_unique_id_val.size() == expected_ids * single_id_size
                   && "Number of unique IDs doesn't match NUM_MAX_NVL_PEERS");

    if (rank == 0) {
        // Print NCCL version from the actually loaded library
        int nccl_version;
        NCCL_CHECK(ncclGetVersion(&nccl_version));
        TM_LOG_DEBUG("[NCCLEP] NCCL version: %d.%d.%d (loaded library)",
                     nccl_version / 10000,
                     (nccl_version % 10000) / 100,
                     nccl_version % 100);
    }

    // All gpus form a group for low latency compatible,
    // otherwise, gpus with the same index across different nodes form a group.
    ncclUniqueId id;
    const int    id_offset = (low_latency_mode) ? 0 : color * single_id_size;
    std::memcpy(&id, root_unique_id_val.data() + id_offset, single_id_size);
    NCCL_CHECK(ncclCommInitRank(&nccl_comm_, comm_nranks, id, comm_rank));

    // The assumption is that kDecoupled is false when initializing SymBuffers in internode.cu
    // IMPORTANT: Use global num_ranks, not comm_nranks, because kernels use global topology
    const auto num_rdma_ranks            = std::max(num_ranks / NUM_MAX_NVL_PEERS, 1);
    int        rdma_channel_head_signals = num_rdma_ranks * DEEP_EP_NCCL_MAX_NUM_CHANNELS;
    int        rdma_channel_tail_signals = num_rdma_ranks * DEEP_EP_NCCL_MAX_NUM_CHANNELS;
    //
    num_ht_signals_ = rdma_channel_head_signals + rdma_channel_tail_signals;
    num_ll_signals_ = qps_per_rank * comm_nranks * 2;

    // Initialize Device Communicators
    auto CreateDevComm = [&](ncclDevComm_t& comm, int signals) {
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.barrierCount            = MAX_BARRIER_SESSIONS;
        reqs.ginSignalCount          = signals + MAX_BARRIER_SESSIONS;
        reqs.ginConnectionType       = NCCL_GIN_CONNECTION_FULL;
        reqs.ginContextCount         = qps_per_rank;
        NCCL_CHECK(ncclDevCommCreate(nccl_comm_, &reqs, &comm));
    };
    CreateDevComm(dev_ll_comm_, num_ll_signals_);  // low latency mode
    CreateDevComm(dev_ht_comm_, num_ht_signals_);  // high throughput mode

    // Allocate barrier dummy variable
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_barrier_var_), sizeof(int)));
    CUDA_CHECK(cudaMemset(d_barrier_var_, 0, sizeof(int)));

    // Store global rank and num_ranks (for external API)
    rank_      = rank;
    num_ranks_ = num_ranks;

    // Store communicator-specific ranks for internal use
    comm_rank_   = comm_rank;
    comm_nranks_ = comm_nranks;

    initialized_ = true;
    TM_LOG_DEBUG(
        "[NCCLEP] Initialized global rank %d/%d (comm rank %d/%d)", rank_, num_ranks_, comm_rank_, comm_nranks_);

    return rank_;
}

void NCCLGINBackend::finalize()
{
    TM_LOG_DEBUG("[NCCLEP][%d] Finalizing", rank_);
    if (!initialized_) {
        return;
    }

    // Destroy device communicators
    auto DestroyDevComm = [&](ncclDevComm_t& comm, std::string_view key) {
        ncclResult_t res = ncclDevCommDestroy(nccl_comm_, &comm);
        if (res != ncclSuccess) {
            TM_LOG_ERROR("[NCCLEP][%d] Failed to destroy device communication %s: %s",
                         rank_,
                         key.data(),
                         ncclGetErrorString(res));
        }
    };
    DestroyDevComm(dev_ll_comm_, "low latency mode");
    DestroyDevComm(dev_ht_comm_, "high throughput mode");

    for (auto& [ptr, win] : wins_) {
        TM_LOG_WARNING("[NCCLEP][%d] Memory %p is not deregistered", rank_, ptr);
    }
    for (auto& [ptr, size] : buffers_) {
        TM_LOG_WARNING("[NCCLEP][%d] Allocation (%p, %lu) is not freed", rank_, ptr, size);
    }

    // Free barrier dummy variable
    if (d_barrier_var_ != nullptr) {
        cudaFree(d_barrier_var_);
        d_barrier_var_ = nullptr;
    }
    // Destroy all communicators
    ncclCommFinalize(nccl_comm_);
    ncclCommDestroy(nccl_comm_);

    TM_LOG_DEBUG("[NCCLEP][%d] Destroyed NCCL communicator", rank_);
    initialized_ = false;
}

void NCCLGINBackend::barrier()
{
    TM_CHECK_EQ(initialized_, true);
    TM_CHECK_NE(d_barrier_var_, nullptr);

    cudaStream_t stream = turbomind::core::Context::stream().handle();
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclAllReduce(d_barrier_var_, d_barrier_var_, 1, ncclInt, ncclSum, nccl_comm_, stream));
    NCCL_CHECK(ncclGroupEnd());
}

void* NCCLGINBackend::alloc(size_t size, size_t /*alignment*/)
{
    TM_CHECK_EQ(initialized_, true);

    void* ptr = nullptr;
    // NCCL memory is already aligned to page size, so alignment parameter is ignored for now.
    NCCL_CHECK(ncclMemAlloc(&ptr, size));
    buffers_.emplace(ptr, size);
    return ptr;
}

void NCCLGINBackend::register_memory(void* ptr, size_t size)
{
    TM_CHECK_EQ(initialized_, true);
    TM_CHECK_EQ(buffers_.find(ptr) != buffers_.end(), true);
    TM_CHECK_EQ(wins_.find(ptr) == wins_.end(), true);
    ncclWindow_t win{};
    NCCL_CHECK(ncclCommWindowRegister(nccl_comm_, ptr, size, &win, 0));
    wins_.emplace(ptr, win);
}

void NCCLGINBackend::free(void* ptr)
{
    TM_CHECK_EQ(initialized_, true);
    auto it = wins_.find(ptr);
    TM_CHECK_EQ(it != wins_.end(), true);
    NCCL_CHECK(ncclCommWindowDeregister(nccl_comm_, it->second));
    NCCL_CHECK(ncclMemFree(ptr));
    wins_.erase(it);
    buffers_.erase(ptr);
}

int NCCLGINBackend::get_rank() const
{
    TM_CHECK_NE(rank_, -1);
    return rank_;
}

int NCCLGINBackend::get_num_ranks() const
{
    TM_CHECK_NE(num_ranks_, -1);
    return num_ranks_;
}

bool NCCLGINBackend::is_p2p_disabled() const
{
    return p2p_disabled_;
}

unsigned NCCLGINBackend::get_signals_base(int buffer_idx, bool low_latency_mode) const
{
    if (low_latency_mode) {
        EP_HOST_ASSERT(buffer_idx == 0 || buffer_idx == 1);
        TM_CHECK_NE(num_ll_signals_, 0);
        return buffer_idx * num_ll_signals_ / 2;
    }
    else {
        EP_HOST_ASSERT(buffer_idx == 0);
        TM_CHECK_NE(num_ht_signals_, 0);
        return 0;
    }
}

ncclWindow_t NCCLGINBackend::get_device_nccl_window(void* ptr)
{
    TM_CHECK_EQ(initialized_, true);
    auto it = wins_.find(ptr);
    TM_CHECK_EQ(it != wins_.end(), true);
    return it->second;
}

ncclDevComm NCCLGINBackend::get_device_communicator(bool low_latency_mode) const
{
    TM_CHECK_EQ(initialized_, true);
    return low_latency_mode ? dev_ll_comm_ : dev_ht_comm_;
}

}  // namespace internode
}  // namespace deep_ep
