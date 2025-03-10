// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>
#include <memory>
#include <numeric>
#include <type_traits>
#include <unordered_map>

#include <nccl.h>

#include "src/turbomind/comm/comm.h"
#include "src/turbomind/comm/host.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/string_utils.h"

#include "src/turbomind/kernels/norm/rms_norm.h"

#define NCCLCHECK(e)                                                                                                   \
    if (auto ec = e; ec != ncclSuccess) {                                                                              \
        auto msg = fmtstr("NCCL error %s:%d '%s'", __FILE__, __LINE__, ncclGetErrorString(ec));                        \
        throw std::runtime_error(msg.c_str());                                                                         \
    }

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
#define NCCL_BUFF_REG 1
#else
#define NCCL_BUFF_REG 0
#endif

namespace turbomind::comm {

static inline ncclDataType_t getNcclDataType(DataType type)
{
    switch (type) {
        case DataType::TYPE_FP32:
            return ncclFloat;
        case DataType::TYPE_FP16:
            return ncclHalf;
        case DataType::TYPE_BF16:
            return ncclBfloat16;
        case DataType::TYPE_UINT8:
            return ncclUint8;
        default:
            throw std::runtime_error("not supported");
    }
}

class NcclComm: public Comm {
public:
    NcclComm(ncclComm_t comm, int world_size, int rank, std::shared_ptr<HostComm> host_comm):
        Comm(world_size, rank), comm_{comm}, groups_{comm}, host_comm_(host_comm)
    {
    }

    ~NcclComm()
    {
        for (const auto& [ptr, _] : handles_) {
            TM_LOG_WARNING("[NCCL][%d] Buffer %p is not deregistered", rank_, ptr);
        }

        for (const auto& [ptr, size] : buffers_) {
            TM_LOG_WARNING("[NCCL][%d] Allocation (%p, %lu) is not freed", rank_, ptr, size);
        }

        for (auto& c : groups_) {
            if (auto ec = ncclCommDestroy(c); ec != ncclSuccess) {
                TM_LOG_ERROR("[NCCL][%d] Failed to destroy communicator: %s", rank_, ncclGetErrorString(ec));
            }
        }
    }

    void* Allocate(size_t size) override
    {
        void* ptr{};
#if NCCL_BUFF_REG
        NCCLCHECK(ncclMemAlloc(&ptr, size));
#else
        check_cuda_error(cudaMalloc(&ptr, size));
#endif
        buffers_.emplace(ptr, size);
        return ptr;
    }

    void Free(void* ptr) override
    {

        if (auto it = buffers_.find(ptr); it != buffers_.end()) {
#if NCCL_BUFF_REG
            NCCLCHECK(ncclMemFree(ptr));
#else
            check_cuda_error(cudaFree(ptr));
#endif
            buffers_.erase(ptr);
        }
        else {
            TM_LOG_WARNING("[TM][NCCL][%d] Freeing %p which is not allocated by NcclComm", rank_, ptr);
        }
    }

    void Register(void* ptr, size_t size) override
    {
#if NCCL_BUFF_REG
        if (!handles_.count(ptr)) {
            void* handle{};
            NCCLCHECK(ncclCommRegister(comm_, ptr, size, &handle));
            handles_.emplace(ptr, handle);
        }
        else {
            TM_LOG_WARNING("[TM][NCCL][%d] Duplicated registration on (%p, %lu)", rank_, ptr, size);
        }
#endif
    }

    void Deregister(void* ptr) override
    {
#if NCCL_BUFF_REG
        if (auto it = handles_.find(ptr); it != handles_.end()) {
            NCCLCHECK(ncclCommDeregister(comm_, it->second));
            handles_.erase(it);
        }
        else {
            TM_LOG_WARNING("[TM][NCCL][%d] Deregistering non-registered address %p", rank_, ptr);
        }
#endif
    }

    int Split(int color, int key, int group) override
    {
        ncclComm_t comm{};
        NCCLCHECK(ncclCommSplit(groups_.at(group), color, key, &comm, nullptr));
        int index = groups_.size();
        groups_.push_back(comm);
        return index;
    }

    int Query(QueryAttr attr) const noexcept override
    {
        return 0;
    }

    void AllReduceSum(const void* sendbuff, void* recvbuff, size_t count, DataType type, cudaStream_t stream) override
    {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, getNcclDataType(type), ncclSum, comm_, stream));
        NCCLCHECK(ncclGroupEnd());
    }

    void AllGather(
        const void* sendbuff, void* recvbuff, size_t sendcount, DataType type, int group, cudaStream_t stream) override
    {
        auto comm = groups_.at(group);
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclAllGather(sendbuff, recvbuff, sendcount, getNcclDataType(type), comm, stream));
        NCCLCHECK(ncclGroupEnd());
    }

    void
    ReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, DataType type, cudaStream_t stream) override
    {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclReduceScatter(sendbuff, recvbuff, recvcount, getNcclDataType(type), ncclSum, comm_, stream));
        NCCLCHECK(ncclGroupEnd());
    }

    void AllreduceResidualBiasRMSnorm(void*        hidden,
                                      void*        residual,
                                      const void*  bias,
                                      const void*  weights,
                                      float        eps,
                                      int          dim,
                                      int          token_num,
                                      DataType     dtype,
                                      cudaStream_t stream) override
    {
        const auto elem_size = get_elem_size(dtype);

        auto rms_norm = [&](int64_t first, int64_t count) {
            invokeResidualBiasRMSNorm((char*)hidden + elem_size * first * dim,
                                      (char*)residual + elem_size * first * dim,
                                      weights,
                                      bias,
                                      dtype,
                                      dim,
                                      count,
                                      eps,
                                      stream);
        };

        if (1) {
            AllReduceSum(hidden, hidden, token_num * dim, dtype, stream);
            rms_norm(0, token_num);
        }
        else {  // Only useful for large input size
            const int    slice     = (token_num + world_size_ - 1) / world_size_;
            const size_t recvcount = slice * dim;
            auto         sendbuff  = hidden;
            auto         recvbuff  = (char*)hidden + elem_size * rank() * recvcount;
            ReduceScatter(sendbuff, recvbuff, recvcount, dtype, stream);
            rms_norm(rank_ * slice, slice);
            AllGather(recvbuff, sendbuff, recvcount, dtype, world_size_, stream);
        }
    }

    void AllGatherAsym(
        const void* sendbuff, void* recvbuff, const size_t* sendcount, DataType type, cudaStream_t stream) override
    {
        const size_t elem_size = get_elem_size(type);

        const char* sendbuf = (const char*)sendbuff;
        char*       recvbuf = (char*)recvbuff;

        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < world_size_; ++i) {
            NCCLCHECK(ncclBroadcast(sendbuf, recvbuf, sendcount[i], getNcclDataType(type), i, comm_, stream));
            sendbuf += elem_size * sendcount[i];
            recvbuf += elem_size * sendcount[i];
        }
        NCCLCHECK(ncclGroupEnd());
    }

    void ReduceScatterAsym(
        const void* sendbuff, void* recvbuff, const size_t* recvcount, DataType type, cudaStream_t stream) override
    {
        const size_t elem_size = get_elem_size(type);

        const char* sendbuf = (const char*)sendbuff;
        char*       recvbuf = (char*)recvbuff;

        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < world_size_; ++i) {
            NCCLCHECK(ncclReduce(sendbuf, recvbuf, recvcount[i], getNcclDataType(type), ncclSum, i, comm_, stream));
            sendbuf += elem_size * recvcount[i];
            recvbuf += elem_size * recvcount[i];
        }
        NCCLCHECK(ncclGroupEnd());
    }

    void AllreduceResidualBiasRMSnormEx(void*        hidden,  // offset by caller
                                        void*        residual,
                                        const void*  bias,
                                        const void*  weights,
                                        float        eps,
                                        int          dim,
                                        DataType     type,
                                        int          group0,
                                        int          group1,
                                        const int*   local_token_nums,
                                        cudaStream_t stream) override
    {

        const size_t         elem_size = get_elem_size(type);
        const ncclDataType_t nccl_type = getNcclDataType(type);

        ncclComm_t comm0 = groups_.at(group0);
        ncclComm_t comm1 = groups_.at(group1);

        int tp0{}, tp1{};
        NCCLCHECK(ncclCommCount(comm0, &tp0));
        NCCLCHECK(ncclCommCount(comm1, &tp1));

        const int inner_tp = std::min(tp0, tp1);

        FT_CHECK(tp0 % inner_tp == 0 && tp1 % inner_tp == 0);

        std::vector<std::tuple<int, int, int>> tasks;
        tasks.reserve(world_size_);

        for (int i = 0; i < world_size_; ++i) {
            const int num   = local_token_nums[i / inner_tp];
            const int slice = (num + inner_tp - 1) / inner_tp;
            const int first = i % inner_tp * slice;
            const int last  = std::min(first + slice, num);
            if (int step = last - first; step > 0) {
                tasks.emplace_back(i, step, first);
            }
        }

        if (tp0 > 1) {
            char* buff = (char*)hidden;
            NCCLCHECK(ncclGroupStart());
            for (const auto& [i, num, _] : tasks) {
                NCCLCHECK(ncclReduce(buff, buff, (size_t)num * dim, nccl_type, ncclSum, i % tp0, comm0, stream));
                buff += elem_size * num * dim;
            }
            NCCLCHECK(ncclGroupEnd());
            sync_check_cuda_error();
        }

        {
            char* buff = (char*)hidden;
            for (const auto& [i, num, first] : tasks) {
                if (i == rank_) {
                    invokeResidualBiasRMSNorm(
                        buff, (char*)residual + elem_size * first * dim, weights, bias, type, dim, num, eps, stream);
                    sync_check_cuda_error();
                }
                buff += elem_size * num * dim;
            }
        }

        if (tp1 > 1) {
            char* buff = (char*)hidden;
            NCCLCHECK(ncclGroupStart());
            for (const auto& [i, num, _] : tasks) {
                NCCLCHECK(ncclBroadcast(buff, buff, (size_t)num * dim, nccl_type, i % tp1, comm1, stream));
                buff += elem_size * num * dim;
            }
            NCCLCHECK(ncclGroupEnd());
            sync_check_cuda_error();
        }
    }

private:
    ncclComm_t comm_;

    std::unordered_map<void*, void*>  handles_;
    std::unordered_map<void*, size_t> buffers_;

    std::vector<ncclComm_t> groups_;

    std::shared_ptr<HostComm> host_comm_;
};

std::unique_ptr<Comm> CreateNcclCommunicator(int rank, int world_size, std::shared_ptr<HostComm> host_comm)
{
    ncclUniqueId uid{};
    if (rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&uid));
    }

    static_assert(std::is_trivially_copyable_v<ncclUniqueId>);
    Broadcast(*host_comm, uid, 0);

    ncclComm_t comm{};
    NCCLCHECK(ncclCommInitRank(&comm, world_size, uid, rank));
    return std::make_unique<NcclComm>(comm, world_size, rank, host_comm);
}

}  // namespace turbomind::comm
