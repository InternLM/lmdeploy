#include <cstring>
#include <vector>

#include "../gin_backend.h"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "src/turbomind/core/check.h"
#include "utils.cuh"

#include <nccl.h>

namespace deep_ep {
namespace intranode {

template<int kNumRanks>
__global__ void barrier(int** barrier_signal_ptrs, int rank)
{
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream)
{
#define BARRIER_LAUNCH_CASE(ranks)                                                                                     \
    LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, rank);                                                    \
    break

    SETUP_LAUNCH_CONFIG(1, 32, stream);
    SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

}  // namespace intranode

namespace internode {

std::vector<uint8_t> get_unique_id()
{
    std::vector<uint8_t> result;

    int num_total_ids = NUM_MAX_NVL_PEERS;

    // Generate unique IDs and pack them
    for (int i = 0; i < num_total_ids; i++) {
        ncclUniqueId unique_id;
        NCCL_CHECK(ncclGetUniqueId(&unique_id));

        size_t offset = result.size();
        result.resize(offset + sizeof(ncclUniqueId));
        std::memcpy(result.data() + offset, &unique_id, sizeof(ncclUniqueId));
    }

    return result;
}

int init(const std::vector<uint8_t>& root_unique_id_val,
         int                         rank,
         int                         num_ranks,
         bool                        low_latency_mode,
         int                         qps_per_rank,
         NCCLGINBackend*             comm)
{
    TM_CHECK_NE(comm, nullptr);
    TM_CHECK_EQ(comm->init(root_unique_id_val, rank, num_ranks, low_latency_mode, qps_per_rank), rank);

    comm->barrier();
    return comm->get_rank();
}

void* alloc(size_t size, size_t alignment, NCCLGINBackend* comm)
{
    return TM_CHECK_NOTNULL(comm)->alloc(size, alignment);
}

void register_memory(void* ptr, size_t size, NCCLGINBackend* comm)
{
    return TM_CHECK_NOTNULL(comm)->register_memory(ptr, size);
}

void free(void* ptr, NCCLGINBackend* comm)
{
    return TM_CHECK_NOTNULL(comm)->free(ptr);
}

void barrier(NCCLGINBackend* comm)
{
    return TM_CHECK_NOTNULL(comm)->barrier();
}

void finalize(NCCLGINBackend* comm)
{
    return TM_CHECK_NOTNULL(comm)->finalize();
}

}  // namespace internode
}  // namespace deep_ep
