// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>
#include <memory>
#include <numeric>
#include <type_traits>
#include <unordered_map>

#include <nccl.h>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/host_comm.h"
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
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 18, 0)
#define NCCL_COMM_SPLIT 1
#endif

namespace turbomind::comm {

static inline ncclDataType_t to_nccl_dtype(DataType type)
{
    switch (type) {
        case kFloat32:
            return ncclFloat;
        case kFloat16:
            return ncclHalf;
        case kBfloat16:
            return ncclBfloat16;
        case kUint8:
            return ncclUint8;
        default:
            throw std::runtime_error("not supported");
    }
}

class NcclCommImpl: public DeviceCommImpl {
public:
    NcclCommImpl(ncclComm_t comm, int n_ranks, int rank, HostComm h_comm):
        h_comm_{h_comm}, global_n_ranks_{n_ranks}, global_rank_{rank}, groups_{comm}
    {
    }

    ~NcclCommImpl()
    {
        for (const auto& [ptr, _] : handles_) {
            TM_LOG_WARNING("[NCCL][%d] Buffer %p is not deregistered", global_rank_, ptr);
        }

        for (const auto& [ptr, size] : buffers_) {
            TM_LOG_WARNING("[NCCL][%d] Allocation (%p, %lu) is not freed", global_rank_, ptr, size);
        }

        for (auto& c : groups_) {
            if (auto ec = ncclCommDestroy(c); ec != ncclSuccess) {
                TM_LOG_ERROR("[NCCL][%d] Failed to destroy communicator: %s", global_rank_, ncclGetErrorString(ec));
            }
        }
    }

    int rank(int group) const override
    {
        int rank{};
        NCCLCHECK(ncclCommUserRank(groups_.at(group), &rank));
        return rank;
    }

    int n_ranks(int group) const override
    {
        int n_ranks{};
        NCCLCHECK(ncclCommCount(groups_.at(group), &n_ranks));
        return n_ranks;
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
            TM_LOG_WARNING("[NCCL][%d] Freeing %p which is not allocated by NcclComm", global_rank_, ptr);
        }
    }

    void Register(void* ptr, size_t size) override
    {
#if NCCL_BUFF_REG
        if (!handles_.count(ptr)) {
            void* handle{};
            NCCLCHECK(ncclCommRegister(groups_.at(0), ptr, size, &handle));
            handles_.emplace(ptr, handle);
        }
        else {
            TM_LOG_WARNING("[NCCL][%d] Duplicated registration on (%p, %lu)", global_rank_, ptr, size);
        }
#endif
    }

    void Deregister(void* ptr) override
    {
#if NCCL_BUFF_REG
        if (auto it = handles_.find(ptr); it != handles_.end()) {
            NCCLCHECK(ncclCommDeregister(groups_.at(0), it->second));
            handles_.erase(it);
        }
        else {
            TM_LOG_WARNING("[NCCL][%d] Deregistering non-registered address %p", global_rank_, ptr);
        }
#endif
    }

    int Split(int color, int key, int group) override
    {
#if NCCL_COMM_SPLIT
        ncclComm_t comm{};
        NCCLCHECK(ncclCommSplit(groups_.at(group), color, key, &comm, nullptr));
        int index = groups_.size();
        groups_.push_back(comm);
        return index;
#else
        TM_LOG_ERROR("[NCCL] This binary is compiled with NCCL version < 2.18 that has no `ncclCommSplit` support");
        FT_CHECK(0);
        return {};
#endif
    }

    int Query(QueryAttr attr) const noexcept override
    {
        return 0;
    }

    void AllReduceSum(
        const void* sendbuff, void* recvbuff, size_t count, DataType type, int group, cudaStream_t stream) override
    {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, to_nccl_dtype(type), ncclSum, groups_.at(group), stream));
        NCCLCHECK(ncclGroupEnd());
    }

    void AllGather(
        const void* sendbuff, void* recvbuff, size_t sendcount, DataType type, int group, cudaStream_t stream) override
    {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclAllGather(sendbuff, recvbuff, sendcount, to_nccl_dtype(type), groups_.at(group), stream));
        NCCLCHECK(ncclGroupEnd());
    }

    void ReduceScatter(
        const void* sendbuff, void* recvbuff, size_t recvcount, DataType type, int group, cudaStream_t stream) override
    {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(
            ncclReduceScatter(sendbuff, recvbuff, recvcount, to_nccl_dtype(type), ncclSum, groups_.at(group), stream));
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
                                      int          group,
                                      cudaStream_t stream) override
    {
        const auto elem_size = byte_size(dtype);

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
            AllReduceSum(hidden, hidden, token_num * dim, dtype, group, stream);
            rms_norm(0, token_num);
        }
        else {  // Only useful for large input size
            const int    n_ranks   = this->n_ranks(group);
            const int    rank      = this->rank(group);
            const int    slice     = (token_num + n_ranks - 1) / n_ranks;
            const size_t recvcount = slice * dim;
            auto         sendbuff  = hidden;
            auto         recvbuff  = (char*)hidden + elem_size * rank * recvcount;
            ReduceScatter(sendbuff, recvbuff, recvcount, dtype, group, stream);
            rms_norm(rank * slice, slice);
            AllGather(recvbuff, sendbuff, recvcount, dtype, group, stream);
        }
    }

    void AllreduceResidualBiasRMSnormEx(void*        hidden,
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
        const size_t         elem_size = byte_size(type);
        const ncclDataType_t nccl_type = to_nccl_dtype(type);

        FT_CHECK(group0 == 0 || group1 == 0);

        ncclComm_t comm0 = groups_.at(group0);
        ncclComm_t comm1 = groups_.at(group1);

        int tp0{}, tp1{};
        NCCLCHECK(ncclCommCount(comm0, &tp0));
        NCCLCHECK(ncclCommCount(comm1, &tp1));

        const int inner_tp = std::min(tp0, tp1);

        FT_CHECK(tp0 % inner_tp == 0 && tp1 % inner_tp == 0);

        std::vector<std::tuple<int, int, int>> tasks;
        tasks.reserve(global_n_ranks_);

        for (int i = 0, offset = 0; i < global_n_ranks_; ++i) {
            const int num   = local_token_nums[i / inner_tp];
            const int slice = (num + inner_tp - 1) / inner_tp;
            const int first = std::min(num, i % inner_tp * slice);
            const int last  = std::min(num, first + slice);
            tasks.emplace_back(offset, first, last - first);
            if ((i + 1) % inner_tp == 0) {
                offset += num;
            }
        }

        if (tp0 > 1) {
            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < global_n_ranks_; ++i) {
                if (auto& [offset, first, num] = tasks[i]; num > 0) {
                    char* buff = (char*)hidden + elem_size * (offset + first) * dim;
                    NCCLCHECK(ncclReduce(buff, buff, (size_t)num * dim, nccl_type, ncclSum, i % tp0, comm0, stream));
                }
            }
            NCCLCHECK(ncclGroupEnd());
            sync_check_cuda_error();
        }

        if (auto& [offset, first, num] = tasks[global_rank_]; num > 0) {
            char* buff = (char*)hidden + elem_size * (offset + first) * dim;
            invokeResidualBiasRMSNorm(
                buff, (char*)residual + elem_size * first * dim, weights, bias, type, dim, num, eps, stream);
            sync_check_cuda_error();
        }

        if (tp1 > 1) {
            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < global_n_ranks_; ++i) {
                if (auto& [offset, first, num] = tasks[i]; num > 0) {
                    char* buff = (char*)hidden + elem_size * (offset + first) * dim;
                    NCCLCHECK(ncclBroadcast(buff, buff, (size_t)num * dim, nccl_type, i % tp1, comm1, stream));
                }
            }
            NCCLCHECK(ncclGroupEnd());
            sync_check_cuda_error();
        }
    }

    int CreateOrGetP2PGroupIndex(int src, int dst, int group)
    {
        int         low_rank  = src < dst ? src : dst;
        int         high_rank = src < dst ? dst : src;
        std::string key = std::to_string(group) + ":" + std::to_string(low_rank) + ":" + std::to_string(high_rank);

        if (p2p_group_index_map_.count(key) == 0) {
            ncclUniqueId uid{};
            static_assert(std::is_trivially_copyable_v<ncclUniqueId>);
            if (src == rank(group)) {
                NCCLCHECK(ncclGetUniqueId(&uid));
                ::turbomind::comm::Send(h_comm_, uid, dst);
            }
            else {
                ::turbomind::comm::Recv(h_comm_, uid, src);
            }

            int        new_rank = low_rank == rank(group) ? 0 : 1;
            ncclComm_t comm{};
            NCCLCHECK(ncclCommInitRank(&comm, 2, uid, new_rank));
            groups_.push_back(comm);
            p2p_group_index_map_[key] = groups_.size() - 1;
        }
        return p2p_group_index_map_[key];
    }

    void Send(const void* sendbuff, size_t count, DataType type, int dst, int group, cudaStream_t stream) override
    {
        int        peer = rank(group) < dst ? 1 : 0;
        ncclComm_t comm = groups_.at(CreateOrGetP2PGroupIndex(rank(group), dst, group));
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclSend(sendbuff, count, to_nccl_dtype(type), peer, comm, stream));
        NCCLCHECK(ncclGroupEnd());
    }

    void Recv(void* recvbuff, size_t count, DataType type, int src, int group, cudaStream_t stream) override
    {
        int        peer = rank(group) < src ? 1 : 0;
        ncclComm_t comm = groups_.at(CreateOrGetP2PGroupIndex(src, rank(group), group));
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclRecv(recvbuff, count, to_nccl_dtype(type), peer, comm, stream));
        NCCLCHECK(ncclGroupEnd());
    }

private:
    HostComm h_comm_;

    int global_n_ranks_;
    int global_rank_;

    std::vector<ncclComm_t> groups_;

    std::unordered_map<void*, void*>  handles_;
    std::unordered_map<void*, size_t> buffers_;

    std::unordered_map<std::string, int> p2p_group_index_map_;
};

DeviceComm CreateNcclCommunicator(int n_ranks, int rank, HostComm h_comm)
{
    ncclUniqueId uid{};
    if (rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&uid));
    }

    static_assert(std::is_trivially_copyable_v<ncclUniqueId>);
    Broadcast(h_comm, uid, 0);

    ncclComm_t comm{};
    NCCLCHECK(ncclCommInitRank(&comm, n_ranks, uid, rank));

    return DeviceComm{std::make_unique<NcclCommImpl>(comm, n_ranks, rank, h_comm)};
}

}  // namespace turbomind::comm
