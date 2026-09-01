#pragma once

#include "csrc/jit/no_ref.hpp"
#include "csrc/kernels/backend/symmetric.hpp"
#include "deep_ep/include/deep_ep/common/compiled.cuh"
#include "src/turbomind/comm/env.h"
#include "src/turbomind/comm/nccl/deepep/utils.h"

#include <nccl.h>
#include <nccl_device.h>
#include <nccl_device/core.h>

#include <fmt/format.h>

#include <memory>
#include <stdexcept>
#include <vector>

#ifndef NCCLCHECK
#define NCCLCHECK(e)                                                                                                   \
    if (auto ec = e; ec != ncclSuccess) {                                                                              \
        auto msg = fmt::format("NCCL error {}:{} '{}'", __FILE__, __LINE__, ncclGetErrorString(ec));                   \
        throw std::runtime_error(msg.c_str());                                                                         \
    }
#endif

using namespace deep_ep;
using namespace deep_ep::elastic;

namespace turbomind::comm {

TM_ENV_VAR(COMM, SL_IDX, 0);
TM_ENV_VAR(COMM, EP_NIC_NAME, std::string("mlx5_0"));
TM_ENV_VAR(COMM, EP_NUM_QPS, 0);

struct NCCLSymmetricMemoryContext {
private:
    // Can not use this unmapped pointer from outside
    void*                                       raw_window_ptr_;
    std::shared_ptr<symmetric::SymmetricMemory> symmetric_memory_;

public:
    NCCLSymmetricMemoryContext(ncclComm_t comm, const int& num_ranks, const int& rank_idx):
        rank_idx_{rank_idx}, num_ranks_{num_ranks}, comm_{comm}
    {

        num_allocated_qps_ = GetEnv<COMM_EP_NUM_QPS>();
        if (num_allocated_qps_ <= 0) {
            num_allocated_qps_ = IsFastRdmaAtomicSupport(GetEnv<COMM_EP_NIC_NAME>()) ? 65 : 129;
        }

        int nccl_runtime_version;
        NCCLCHECK(ncclGetVersion(&nccl_runtime_version));

        // Initialize NCCL device communicator
        ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
        NCCLCHECK(ncclCommQueryProperties(comm, &props));
        ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        if (num_ranks > 1) {
            TM_CHECK(props.railedGinType != NCCL_GIN_TYPE_NONE);

            reqs.ginContextCount      = num_allocated_qps_;
            reqs.ginExclusiveContexts = true;
            reqs.ginQueueDepth        = kGinQPDepth;
            reqs.ginTrafficClass      = GetEnv<COMM_SL_IDX>();
            // Customized RDMA barrier needs extra signals
            reqs.ginSignalCount    = num_ranks + 2 * 2;
            reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
        }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)
        reqs.useRuntimeVersion = true;
        dev_comm_.ptr          = malloc(props.devCommRuntimeVersionSize);
#else
        TM_CHECK_EQ(NCCL_VERSION_CODE, nccl_runtime_version)
            << "Prior to NCCL 2.31, NCCL compile-time and runtime versions must be the same. Please use the version "
            << NCCL_VERSION_CODE;
        dev_comm_.ptr = malloc(sizeof(ncclDevComm_t));
#endif
        TM_CHECK(dev_comm_.ptr != nullptr);
        NCCLCHECK(ncclDevCommCreate(comm, &reqs, static_cast<ncclDevComm_t*>(dev_comm_.ptr)));

        // Now we know the NVLink domain size
        ncclTeam_t lsaTeam = ncclTeamLsa(comm);
        num_nvl_ranks_     = lsaTeam.nRanks;
        nvl_rank_idx_      = lsaTeam.rank;
        num_rdma_ranks_    = num_ranks / num_nvl_ranks_;
        rdma_rank_idx_     = rank_idx / num_nvl_ranks_;
        TM_CHECK(num_ranks % num_nvl_ranks_ == 0 && nvl_rank_idx_ == rank_idx % num_nvl_ranks_);
        TM_CHECK(rank_idx == rdma_rank_idx_ * num_nvl_ranks_ + nvl_rank_idx_);

        num_scaleout_ranks_ = num_rdma_ranks_;
        num_scaleup_ranks_  = num_nvl_ranks_;
        scaleout_rank_idx_  = rdma_rank_idx_;
        scaleup_rank_idx_   = nvl_rank_idx_;
        is_scaleup_nvlink_  = num_scaleup_ranks_ == num_nvl_ranks_;
    }

    void Init(const int64_t& num_bytes, const int64_t& num_cpu_bytes)
    {
        TM_CHECK(window_ == nullptr);

        // Create symmetric memory
        // num_bytes = GPU + CPU, derive GPU portion
        this->symmetric_memory_ = symmetric::alloc(num_bytes - num_cpu_bytes,
                                                   num_cpu_bytes,
                                                   true /* allow_hybrid_mode */,
                                                   num_scaleup_ranks_,
                                                   scaleout_rank_idx_,
                                                   {} /* cpu_comm */);
        // Create window
        // NOTES: `ncclCommWindowRegister` is collective: it internally calls bootstrapBarrier
        // across all ranks, so no explicit barrier is needed after this call.
        raw_window_ptr_      = this->symmetric_memory_->ptr;
        this->num_gpu_bytes_ = this->symmetric_memory_->num_gpu_bytes;
        this->num_cpu_bytes_ = this->symmetric_memory_->num_cpu_bytes;
        NCCLCHECK(ncclCommWindowRegister(
            comm_, raw_window_ptr_, this->symmetric_memory_->num_bytes, &window_, NCCL_WIN_STRICT_ORDERING));
        NCCLCHECK(ncclGetLsaDevicePointer(window_, 0, nvl_rank_idx_, &mapped_window_ptr_));

        // Get LSA pointers for all LSA peers
        // TODO: check whether this is correct for network with RDMA
        nvl_window_ptrs_.resize(num_nvl_ranks_);
        for (int i = 0; i < num_nvl_ranks_; ++i)
            NCCLCHECK(ncclGetLsaDevicePointer(window_, 0, i, &nvl_window_ptrs_[i]));
    }

    void* get_sym_ptr(void* ptr, const int& dst_rank_idx) const
    {
        const auto offset = static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(mapped_window_ptr_);
        return static_cast<uint8_t*>(nvl_window_ptrs_[dst_rank_idx]) + offset;
    }

    ~NCCLSymmetricMemoryContext()
    {
        // Deregister window
        if (window_) {
            if (auto ec = ncclCommWindowDeregister(comm_, window_); ec != ncclSuccess) {
                TM_LOG_ERROR("Rank {}: Failed to destory windows: {}", rank_idx_, ncclGetErrorString(ec));
            }
        }
        symmetric_memory_.reset();

        // Destroy device communicator
        if (auto ec = ncclDevCommDestroy(comm_, static_cast<ncclDevComm_t*>(dev_comm_.ptr)); ec != ncclSuccess) {
            TM_LOG_ERROR("Rank {}: Failed to destory device communicator: {}", rank_idx_, ncclGetErrorString(ec));
        }
        free(dev_comm_.ptr);
    }

    // Global
    int rank_idx_;
    int num_ranks_;

    // Logical
    int num_scaleout_ranks_;
    int num_scaleup_ranks_;
    int scaleout_rank_idx_;
    int scaleup_rank_idx_;

    // Physical
    int  num_rdma_ranks_;
    int  num_nvl_ranks_;
    int  rdma_rank_idx_;
    int  nvl_rank_idx_;
    bool is_scaleup_nvlink_;

    // NCCL handles
    ncclComm_t         comm_;
    jit::NoRefPtr      dev_comm_;
    ncclWindow_t       window_{};
    void*              mapped_window_ptr_;
    std::vector<void*> nvl_window_ptrs_;

    // Configs
    int num_allocated_qps_{};

    // Buffer size
    int64_t num_gpu_bytes_{};
    int64_t num_cpu_bytes_{};
};

}  // namespace turbomind::comm
