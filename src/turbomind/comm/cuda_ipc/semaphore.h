#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/comm/cuda_ipc/common.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

struct SystemSemaphoreInfo {
    uint64_t* outbound[kMaxChannels * kMaxRanks];
    uint64_t* inbound[kMaxChannels * kMaxRanks];
    uint64_t  expected[kMaxChannels * kMaxRanks];
    // uint64_t* mc_ptr[kMaxChannels];
};

struct SystemSemaphoreStorage {

    uint64_t*            data_{};  // uint32[kMaxChannels][kMaxRanks], symmetric
    SystemSemaphoreInfo* info_{};

    template<class AllocReg>
    void Allocate(int ranks, int rank, AllocReg alloc_reg)
    {
        const size_t byte_size = sizeof(uint64_t) * kMaxChannels * kMaxRanks;

        SymmetricPtr_V2<uint64_t> v = alloc_reg(byte_size);

        data_ = v.uc[rank];

        SystemSemaphoreInfo info{};

        for (int c = 0; c < kMaxChannels; ++c) {  // block idx
            for (int r = 0; r < ranks; ++r) {     // thread idx
                info.inbound[c * kMaxRanks + r]  = v.uc[rank] + c * kMaxRanks + r;
                info.outbound[c * kMaxRanks + r] = v.uc[r] + c * kMaxRanks + rank;
                // info.mc_ptr[c]                   = v.mc + c * kMaxRanks + rank;
            }
        }

        check_cuda_error(cudaMallocAsync(&info_, sizeof(SystemSemaphoreInfo), 0));
        check_cuda_error(cudaMemcpyAsync(info_, &info, sizeof(SystemSemaphoreInfo), cudaMemcpyDefault, 0));

        check_cuda_error(cudaStreamSynchronize(0));
    }

    template<class DeregFree>
    void Free(DeregFree dereg_free)
    {
        check_cuda_error(cudaFreeAsync(info_, 0));
        info_ = {};

        dereg_free(data_);
        data_ = {};
    }

    SystemSemaphoreInfo* handle()
    {
        return info_;
    }
};

}  // namespace turbomind::comm
