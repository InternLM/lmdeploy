// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/nccl/nccl_comm.h"

#include <cstdint>
#include <numeric>
#include <type_traits>

#include <dlfcn.h>

#include "src/turbomind/comm/nccl/deep_ep/deep_ep.hpp"
#include "src/turbomind/core/check.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/string_utils.h"

#include "src/turbomind/kernels/norm/rms_norm.h"

#define NCCLCHECK(e)                                                                                                   \
    if (auto ec = e; ec != ncclSuccess) {                                                                              \
        auto msg = fmtstr("NCCL error %s:%d '%s'", __FILE__, __LINE__, ncclGetErrorString(ec));                        \
        throw std::runtime_error(msg.c_str());                                                                         \
    }

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 27, 0)
/* Window Registration flags */
#define NCCL_WIN_DEFAULT 0x00
#define NCCL_WIN_COLL_SYMMETRIC 0x01
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

struct NcclApis {
    ncclResult_t (*ncclMemAlloc)(void** ptr, size_t size);
    ncclResult_t (*ncclMemFree)(void* ptr);
    ncclResult_t (*ncclCommRegister)(const ncclComm_t comm, void* buff, size_t size, void** handle);
    ncclResult_t (*ncclCommDeregister)(const ncclComm_t comm, void* handle);
    ncclResult_t (*ncclCommWindowRegister)(ncclComm_t comm, void* buff, size_t size, void** win, int winFlags);
    ncclResult_t (*ncclCommWindowDeregister)(ncclComm_t comm, void* win);
    // `ncclConfig_t` varies between versions, should be fine as long as we are passing nullptr to it
    ncclResult_t (*ncclCommSplit)(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, void* config);
};

static NcclApis& nccl_apis()
{
    static auto value = [] {
        int version{};
        ncclGetVersion(&version);
        auto     handle = dlopen("libnccl.so.2", RTLD_LAZY);
        NcclApis apis{};
        if (!handle) {
            return apis;
        }
        auto load_symbol = [&](auto& dst, auto name) {
            using T = std::remove_reference_t<decltype(dst)>;
            dst     = reinterpret_cast<T>(dlsym(handle, name));
        };
        if (version >= NCCL_VERSION(2, 27, 0)) {
            if (version < NCCL_VERSION(2, 28, 0)) {
                TM_LOG_WARNING(
                    "[NCCL] Window registration may cause memory leaks in NCCL 2.27, use NCCL 2.28+ or disable the feature by setting NCCL_WIN_ENABLE=0.");
            }
            load_symbol(apis.ncclCommWindowRegister, "ncclCommWindowRegister");
            load_symbol(apis.ncclCommWindowDeregister, "ncclCommWindowDeregister");
        }
        else {
            TM_LOG_WARNING(
                "[NCCL] Window registration is not supported by NCCL %d, use NCCL 2.28+ for better performance.",
                version);
        }
        if (version >= NCCL_VERSION(2, 19, 0)) {
            load_symbol(apis.ncclMemAlloc, "ncclMemAlloc");
            load_symbol(apis.ncclMemFree, "ncclMemFree");
            load_symbol(apis.ncclCommRegister, "ncclCommRegister");
            load_symbol(apis.ncclCommDeregister, "ncclCommDeregister");
        }
        if (version >= NCCL_VERSION(2, 18, 0)) {
            load_symbol(apis.ncclCommSplit, "ncclCommSplit");
        }
        else {
            TM_LOG_WARNING("[NCCL] Splitting communicators is not supported by NCCL %d, use NCCL 2.18+ if needed.",
                           version);
        }
        return apis;
    }();
    return value;
}

NcclCommImpl::NcclCommImpl(ncclComm_t comm, int n_ranks, int rank, HostComm h_comm):
    h_comm_{h_comm}, global_n_ranks_{n_ranks}, global_rank_{rank}, groups_{comm}
{
    handles_.emplace_back();
}

NcclCommImpl::~NcclCommImpl()
{
    for (const auto& [ptr, _] : handles_.at(0)) {
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
    if (buffer_) {
        buffer_->destroy();
    }
}

int NcclCommImpl::rank(int group) const
{
    int rank{};
    NCCLCHECK(ncclCommUserRank(groups_.at(group), &rank));
    return rank;
}

int NcclCommImpl::n_ranks(int group) const
{
    int n_ranks{};
    NCCLCHECK(ncclCommCount(groups_.at(group), &n_ranks));
    return n_ranks;
}

void* NcclCommImpl::Allocate(size_t size)
{
    void* ptr{};
    if (auto alloc_fn = nccl_apis().ncclMemAlloc) {
        NCCLCHECK(alloc_fn(&ptr, size));
    }
    else {
        check_cuda_error(cudaMalloc(&ptr, size));
    }
    buffers_.emplace(ptr, size);
    return ptr;
}

void NcclCommImpl::Free(void* ptr)
{
    if (auto it = buffers_.find(ptr); it != buffers_.end()) {
        if (auto free_fn = nccl_apis().ncclMemFree) {
            NCCLCHECK(free_fn(ptr));
        }
        else {
            check_cuda_error(cudaFree(ptr));
        }
        buffers_.erase(ptr);
    }
    else {
        TM_LOG_WARNING("[NCCL][%d] Freeing %p which is not allocated by NcclComm", global_rank_, ptr);
    }
}

void NcclCommImpl::Register(void* ptr, size_t size)
{
    if (!handles_.at(0).count(ptr)) {
        for (size_t i = 0; i < handles_.size(); ++i) {
            Register(i, ptr, size);
        }
    }
    else {
        TM_LOG_WARNING("[NCCL][%d] Duplicated registration on (%p, %lu)", global_rank_, ptr, size);
    }
}

void NcclCommImpl::Deregister(void* ptr)
{
    if (handles_.at(0).count(ptr)) {
        for (size_t i = 0; i < handles_.size(); ++i) {
            Deregister(i, ptr);
        }
    }
    else {
        TM_LOG_WARNING("[NCCL][%d] Deregistering non-registered address %p", global_rank_, ptr);
    }
}

void NcclCommImpl::Register(int group, void* buff, size_t size)
{
    void* handle{};
    auto  comm = groups_.at(group);
    if (auto func = nccl_apis().ncclCommWindowRegister) {
        NCCLCHECK(func(comm, buff, size, &handle, NCCL_WIN_COLL_SYMMETRIC));
    }
    else if (auto func = nccl_apis().ncclCommRegister) {
        NCCLCHECK(func(comm, buff, size, &handle));
    }
    handles_.at(group).emplace(buff, std::make_pair(handle, size));
}

void NcclCommImpl::Deregister(int group, void* buff)
{
    auto& handles = handles_.at(group);
    if (auto it = handles.find(buff); it != handles.end()) {
        if (auto func = nccl_apis().ncclCommWindowDeregister) {
            NCCLCHECK(func(groups_.at(group), it->second.first));
        }
        else if (auto func = nccl_apis().ncclCommDeregister) {
            NCCLCHECK(func(groups_.at(group), it->second.first));
        }
        handles.erase(it);
    }
}

int NcclCommImpl::Split(int color, int key, int group)
{
    auto split_fn = TM_CHECK_NOTNULL(nccl_apis().ncclCommSplit);

    ncclComm_t comm{};
    NCCLCHECK(split_fn(groups_.at(group), color, key, &comm, nullptr));

    int index = groups_.size();
    groups_.push_back(comm);
    handles_.emplace_back();

    // register all existing buffers on the group
    for (const auto& [k, v] : handles_.at(0)) {
        Register(index, k, v.second);
    }

    return index;
}

int NcclCommImpl::Query(QueryAttr attr) const noexcept
{
    return 0;
}

void NcclCommImpl::AllReduceSum(
    const void* sendbuff, void* recvbuff, size_t count, DataType type, int group, cudaStream_t stream)
{
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, to_nccl_dtype(type), ncclSum, groups_.at(group), stream));
    NCCLCHECK(ncclGroupEnd());
}

void NcclCommImpl::AllGather(
    const void* sendbuff, void* recvbuff, size_t sendcount, DataType type, int group, cudaStream_t stream)
{
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclAllGather(sendbuff, recvbuff, sendcount, to_nccl_dtype(type), groups_.at(group), stream));
    NCCLCHECK(ncclGroupEnd());
}

void NcclCommImpl::ReduceScatter(
    const void* sendbuff, void* recvbuff, size_t recvcount, DataType type, int group, cudaStream_t stream)
{
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(
        ncclReduceScatter(sendbuff, recvbuff, recvcount, to_nccl_dtype(type), ncclSum, groups_.at(group), stream));
    NCCLCHECK(ncclGroupEnd());
}

void NcclCommImpl::ReduceScatterV(const void*   sendbuff,  //
                                  void*         recvbuff,
                                  const size_t* counts,
                                  DataType      type,
                                  int           group,
                                  cudaStream_t  stream)
{
    std::vector<size_t> offsets(n_ranks(group));
    std::exclusive_scan(counts, counts + n_ranks(group), offsets.begin(), 0);

    const auto elem_size = byte_size(type);
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < n_ranks(group); ++i) {
        NCCLCHECK(ncclReduce((char*)sendbuff + offsets[i] * elem_size,
                             recvbuff,
                             counts[i],
                             to_nccl_dtype(type),
                             ncclSum,
                             i,
                             groups_.at(group),
                             stream));
    }
    NCCLCHECK(ncclGroupEnd());
}

void NcclCommImpl::AllGatherV(const void*   sendbuff,  //
                              void*         recvbuff,
                              const size_t* counts,
                              DataType      type,
                              int           group,
                              cudaStream_t  stream)
{
    std::vector<size_t> offsets(n_ranks(group));
    std::exclusive_scan(counts, counts + n_ranks(group), offsets.begin(), 0);

    const auto elem_size = byte_size(type);
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < n_ranks(group); ++i) {
        NCCLCHECK(ncclBroadcast(sendbuff,
                                (char*)recvbuff + offsets[i] * elem_size,
                                counts[i],
                                to_nccl_dtype(type),
                                i,
                                groups_.at(group),
                                stream));
    }
    NCCLCHECK(ncclGroupEnd());
}

void NcclCommImpl::AllreduceResidualBiasRMSnorm(void*        hidden,
                                                void*        residual,
                                                const void*  bias,
                                                const void*  weights,
                                                float        eps,
                                                int          dim,
                                                int          token_num,
                                                DataType     dtype,
                                                int          group,
                                                cudaStream_t stream)
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

void NcclCommImpl::AllreduceResidualBiasRMSnormEx(void*        hidden,
                                                  void*        residual,
                                                  const void*  bias,
                                                  const void*  weights,
                                                  float        eps,
                                                  int          dim,
                                                  DataType     type,
                                                  int          group0,
                                                  int          group1,
                                                  const int*   local_token_nums,
                                                  cudaStream_t stream)
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

void NcclCommImpl::Broadcast(const void*  sendbuff,  //
                             void*        recvbuff,
                             size_t       count,
                             DataType     type,
                             int          root,
                             int          group,
                             cudaStream_t stream)
{
    NCCLCHECK(ncclBroadcast(recvbuff, recvbuff, count, to_nccl_dtype(type), root, groups_.at(group), stream));
}

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
