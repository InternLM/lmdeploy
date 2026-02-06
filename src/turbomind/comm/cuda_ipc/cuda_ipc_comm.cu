// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdlib>
#include <memory>
#include <numeric>
#include <vector>

#include <cuda.h>

#include "src/turbomind/comm/cuda_ipc/common.h"
#include "src/turbomind/kernels/core/math.h"

#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/env.h"
#include "src/turbomind/comm/host_comm.h"

#include "src/turbomind/comm/cuda_ipc/semaphore.h"

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind::comm {

TM_ENV_VAR(COMM, MAX_CTAS, 0);
TM_ENV_VAR(COMM, NVLS_ENABLE, 1);
// per-rank send size threshold to use copy engine instead of p2p for all-gather colls
TM_ENV_VAR(COMM, COPY_THRESHOLD, INT64_MAX);

int CudaIpcCommImpl::Split(int color, int key, int group)
{
    FT_CHECK(color >= 0);
    FT_CHECK(rank(group) >= 0);

    auto& parent = groups_.at(group);

    auto vec = comm::AllGather(h_comm_, std::make_tuple(color, key, parent.g2l[global_rank_]));

    auto last = std::stable_partition(vec.begin(), vec.end(), [&](auto x) {  //
        return std::get<0>(x) == color;
    });
    vec.erase(last, vec.end());
    std::stable_sort(vec.begin(), vec.end(), [](auto& a, auto& b) {  //
        return a < b;
    });

    std::vector<int> l2g;
    std::vector<int> g2l(parent.g2l.size(), -1);

    for (size_t local = 0; local < vec.size(); ++local) {
        const auto r      = std::get<2>(vec[local]);
        int        global = parent.l2g.at(r);
        l2g.push_back(global);
        g2l[global] = local;
    }

    int index = groups_.size();

    auto& g = groups_.emplace_back(Group{l2g, g2l});

    for (auto& a : allocation_) {
        Register(a, index);
    }

    g.semaphore.Allocate(l2g.size(), g2l[global_rank_], [&](size_t size) {
        auto buf = (uint64_t*)Allocate(size);
        check_cuda_error(cudaMemsetAsync(buf, 0, size));
        check_cuda_error(cudaStreamSynchronize(0));
        Register(buf, size);
        return get_symmetric_v2(buf, index);
    });

    return index;
};

CudaIpcCommImpl::CudaIpcCommImpl(HostComm h_comm):
    h_comm_{h_comm}, global_n_ranks_{h_comm->n_ranks()}, global_rank_{h_comm->rank()}
{
    h_comm_ = h_comm;

    const int n_ranks = global_n_ranks_;
    const int rank    = global_rank_;

    // Exchange device ordinals
    ordinals_.resize(n_ranks);
    check_cuda_error(cudaGetDevice(&ordinals_[rank]));
    comm::AllGather(h_comm_, ordinals_.data(), 1);

    max_ctas_ = {std::min(getSMCount(), kMaxChannels)};
    if (auto v = GetEnv<COMM_MAX_CTAS>()) {
        max_ctas_.set_value(std::min(v, max_ctas_.value()));
    }
    auto minval = comm::AllReduce(h_comm_, max_ctas_.value(), RedOp::kMin);
    TM_CHECK_EQ(max_ctas_.value(), minval) << "MAX_CTAS set to different values";

#if __CUDACC_VER_MAJOR__ >= 12
    if (global_n_ranks_ >= 4 && GetEnv<COMM_NVLS_ENABLE>()) {  // solve 2n-2>n+1 -> n>3
        CUDRVCHECK(
            cuDeviceGetAttribute(&multicast_capability_, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, ordinals_[rank]));
        multicast_capability_ = comm::AllReduce(h_comm_, multicast_capability_, RedOp::kMin);
    }
#endif

    copy_threshold_ = GetEnv<COMM_COPY_THRESHOLD>();

    // Prepare access descriptors
    alloc_access_descs_.resize(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
        alloc_access_descs_[r].location.id   = ordinals_[r];
        alloc_access_descs_[r].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        alloc_access_descs_[r].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    // Initialize group mapping
    std::vector<int> idxs(n_ranks);
    std::iota(idxs.begin(), idxs.end(), 0);
    auto& g = groups_.emplace_back();
    g.l2g = g.g2l = idxs;

    // Prepare packet buffer
    packet_buff_ = Allocate(kPacketBuffSize);
    check_cuda_error(cudaMemsetAsync(packet_buff_, 0, kPacketBuffSize));

    // Prepare scratch buffer
    scratch_buff_ = Allocate(kScratchBuffSize);
    check_cuda_error(cudaMemsetAsync(scratch_buff_, 0, kScratchBuffSize));

    /// TODO: release
    g.semaphore.Allocate(global_n_ranks_, global_rank_, [this](size_t size) {
        auto buf = (uint64_t*)Allocate(size);
        check_cuda_error(cudaMemsetAsync(buf, 0, size));
        check_cuda_error(cudaStreamSynchronize(0));
        Register(buf, size);
        return get_symmetric_v2(buf, 0);
    });

    check_cuda_error(cudaStreamSynchronize(0));

    Register(packet_buff_, kPacketBuffSize);
    Register(scratch_buff_, kScratchBuffSize);
}

CudaIpcCommImpl::~CudaIpcCommImpl()
{
    Deregister(scratch_buff_);
    Deregister(packet_buff_);

    Free(scratch_buff_);
    Free(packet_buff_);

    for (auto i = (int)groups_.size() - 1; i >= 0; --i) {
        groups_[i].semaphore.Free([this](void* ptr) {
            Deregister(ptr);
            Free(ptr);
        });
    }

    for (const auto& a : allocation_) {
        TM_LOG_WARNING("[COMM][%d] Allocation (%p, %lu) is not freed", global_rank_, a.uc_beg, a.size);
    }

    cudaStreamSynchronize(0);
}

void* CudaIpcCommImpl::Allocate(size_t size)
{
    size_t              granularity{};
    CUmemAllocationProp prop{};

    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = ordinals_[global_rank_];

    if (multicast_capability_) {
#if __CUDACC_VER_MAJOR__ >= 12
        CUmulticastObjectProp prop{};
        prop.numDevices = alloc_access_descs_.size();
        prop.size       = size;
        CUDRVCHECK(cuMulticastGetGranularity(&granularity, &prop, CU_MULTICAST_GRANULARITY_MINIMUM));
#else
        TM_CHECK(0);
#endif
    }
    else {
        CUDRVCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    }

    size = round_up(size, granularity);

    CUmemGenericAllocationHandle handle{};
    CUDRVCHECK(cuMemCreate(&handle, size, &prop, 0));

    CUdeviceptr ptr{};
    CUDRVCHECK(cuMemAddressReserve(&ptr, size, granularity, 0, 0));
    CUDRVCHECK(cuMemMap(ptr, size, 0, handle, 0));
    CUDRVCHECK(cuMemSetAccess(ptr, size, alloc_access_descs_.data(), alloc_access_descs_.size()));

    Allocation a{};
    a.handle    = handle;
    a.size      = size;
    a.uc_beg    = reinterpret_cast<void*>(ptr);
    a.uc_end    = (char*)a.uc_beg + size;
    a.alignment = granularity;

    a.uc_ptrs = comm::AllGather(h_comm_, a.uc_beg);

    allocation_.emplace(a);

    return a.uc_beg;
}

void CudaIpcCommImpl::Free(void* ptr)
{
    if (auto it = allocation_.find(ptr); it != allocation_.end()) {
        auto& a    = *it;
        auto  dptr = reinterpret_cast<CUdeviceptr>(ptr);
        CUDRVCHECK(cuMemUnmap(dptr, a.size));
        CUDRVCHECK(cuMemRelease(a.handle));
        CUDRVCHECK(cuMemAddressFree(dptr, a.size));
        allocation_.erase(it);
    }
    else {
        TM_LOG_WARNING("[TM][COMM][%d] Freeing %p which is not allocated by this module", global_rank_, ptr);
    }
}

void CudaIpcCommImpl::Register(void* ptr, size_t size)
{
    // register for all groups
    auto& symm = groups_.at(0).symmetric;

    if (symm.find(ptr) != symm.end()) {
        TM_LOG_WARNING("[TM][COMM][%d] Duplicated registration on (%p, %lu)", global_rank_, ptr, size);
        return;
    }

    auto alloc = allocation_.find(ptr);
    TM_CHECK(alloc != allocation_.end());

    for (size_t i = 0; i < groups_.size(); ++i) {
        Register(*alloc, i);
    }
}

void CudaIpcCommImpl::Register(const Allocation& alloc, int group)
{
    auto size = alloc.size;

    auto& g = groups_.at(group);

    Symmetric s{};
    s.size   = size;
    s.uc_beg = alloc.uc_beg;
    s.uc_end = alloc.uc_end;

    for (auto r : g.l2g) {
        s.uc_ptrs.push_back(alloc.uc_ptrs[r]);
    }

    const int ranks = n_ranks(group);
    const int rank  = this->rank(group);

    if (multicast_capability_ && ranks > 1) {  // ! `cuMulticastCreate` fails for `ranks == 1`
#if __CUDACC_VER_MAJOR__ >= 12
        CUmulticastObjectProp mc_prop{};
        mc_prop.numDevices = ranks;
        mc_prop.size       = size;
        if (rank == 0) {
            CUDRVCHECK(cuMulticastCreate(&s.mc_handle, &mc_prop));
        }
        auto handles = comm::AllGather(h_comm_, s.mc_handle);
        s.mc_handle  = handles.at(g.l2g[0]);
        CUDRVCHECK(cuMulticastAddDevice(s.mc_handle, ordinals_[global_rank_]));
        CUDRVCHECK(cuMulticastBindMem(s.mc_handle, 0, alloc.handle, 0, size, 0));
        CUdeviceptr mc_ptr{};
        CUDRVCHECK(cuMemAddressReserve(&mc_ptr, size, alloc.alignment, 0, 0));
        CUDRVCHECK(cuMemMap(mc_ptr, size, 0, s.mc_handle, 0));
        CUDRVCHECK(cuMemSetAccess(mc_ptr, size, &alloc_access_descs_[global_rank_], 1));
        s.mc_ptr = reinterpret_cast<void*>(mc_ptr);
        if (rank != 0) {
            // Increase reference count to the original handle so that all handles can be released
            // without explicit synchronization
            CUDRVCHECK(cuMemRetainAllocationHandle(&s.mc_handle, s.mc_ptr));
        }
#else
        TM_CHECK(0);
#endif
    }

    g.symmetric.insert(std::move(s));
}

void CudaIpcCommImpl::Deregister(Symmetric& s)
{
    if (s.mc_handle) {
#if __CUDACC_VER_MAJOR__ >= 12
        auto deviceptr = reinterpret_cast<CUdeviceptr>(s.mc_ptr);
        CUDRVCHECK(cuMemUnmap(deviceptr, s.size));
        CUDRVCHECK(cuMemAddressFree(deviceptr, s.size));
        CUDRVCHECK(cuMulticastUnbind(s.mc_handle, ordinals_.at(global_rank_), 0, s.size));
        CUDRVCHECK(cuMemRelease(s.mc_handle));
        s.mc_handle = {};
        s.mc_ptr    = {};
#else
        TM_CHECK(0);
#endif
    }
}

void CudaIpcCommImpl::Deregister(void* ptr)
{
    std::vector<CUmemGenericAllocationHandle> handles;

    for (size_t i = 0; i < groups_.size(); ++i) {
        auto& s = groups_[i].symmetric;
        if (auto it = s.find(ptr); it != s.end()) {
            Deregister(s.extract(it).value());
        }
        else {
            TM_LOG_WARNING("[TM][COMM][%d] Deregistering non-registered address %p", global_rank_, ptr);
        }
    }
}

int CudaIpcCommImpl::Query(QueryAttr attr) const noexcept
{
    if (attr == kHasAllGather2D) {
        return 1;
    }
    return 0;
}

auto CudaIpcCommImpl::get_symmetric_v2_impl(void* ptr, int group) -> SymmetricPtr_V2<void>
{
    auto& g = groups_.at(group);

    auto symm = g.symmetric.find(ptr);
    TM_CHECK(symm != g.symmetric.end());

    auto offset = (char*)ptr - (char*)symm->uc_beg;

    SymmetricPtr_V2<void> p{};

    TM_CHECK_LE((int)symm->uc_ptrs.size(), p.uc.size());

    for (size_t i = 0; i < symm->uc_ptrs.size(); ++i) {
        p.uc[i] = (char*)symm->uc_ptrs[i] + offset;
    }

    if (symm->mc_ptr) {
        p.mc = (char*)symm->mc_ptr + offset;
    }

    return p;
}

DeviceComm CreateCudaIpcCommunicator(int n_ranks, int rank, HostComm h_comm)
{
    auto comm = std::make_unique<CudaIpcCommImpl>(h_comm);

    return DeviceComm{std::move(comm)};
}

}  // namespace turbomind::comm
