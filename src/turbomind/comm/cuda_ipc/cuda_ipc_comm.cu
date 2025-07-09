// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>
#include <numeric>
#include <vector>

#include <cuda.h>

#include "src/turbomind/kernels/core/math.h"

#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/host_comm.h"

#include "src/turbomind/comm/cuda_ipc/semaphore.h"

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind::comm {

int CudaIpcCommImpl::Split(int color, int key, int group)
{
    FT_CHECK(color >= 0);
    FT_CHECK(rank(group) >= 0);

    auto& t = groups_.at(group);

    auto buffer = create_semaphore_buffer();

    auto vec = comm::AllGather(h_comm_, std::make_tuple(color, key, t.g2l[global_rank_], buffer));

    auto last = std::stable_partition(vec.begin(), vec.end(), [&](auto x) {  //
        return std::get<0>(x) == color;
    });
    vec.erase(last, vec.end());
    std::stable_sort(vec.begin(), vec.end(), [](auto& a, auto& b) {  //
        return a < b;
    });

    std::vector<int> l2g;
    std::vector<int> g2l(t.g2l.size(), -1);

    std::vector<uint64_t*> buffers;

    for (size_t local = 0; local < vec.size(); ++local) {
        const auto& [c, k, r, b] = vec[local];
        buffers.push_back(b);
        int global = t.l2g.at(r);
        l2g.push_back(global);
        g2l[global] = local;
    }

    int index = groups_.size();

    auto& g = groups_.emplace_back(Group{l2g, g2l});

    g.d2d_semaphore_data = buffer;
    g.d2d_semaphores     = init_semaphores(buffers, index);

    for (auto& a : allocation_) {
        register_for_group(a, a.uc_ptrs, index);
    }

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

    CUDRVCHECK(cuDeviceGetAttribute(&multicast_capability_, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, ordinals_[rank]));

    multicast_capability_ = comm::AllReduce(h_comm_, multicast_capability_, RedOp::kMin);

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

    // Exchange data buffers
    std::vector<uint64_t*> buffers = comm::AllGather(h_comm_, create_semaphore_buffer());
    // Initialize D2D semaphores
    g.d2d_semaphore_data = buffers[rank];
    g.d2d_semaphores     = init_semaphores(buffers, 0);

    // Prepare packet buffer
    packet_buff_ = Allocate(kPacketBuffSize);
    check_cuda_error(cudaMemsetAsync(packet_buff_, 0, kPacketBuffSize));

    // Prepare scratch buffer
    scratch_buff_ = Allocate(kScratchBuffSize);
    check_cuda_error(cudaMemsetAsync(scratch_buff_, 0, kScratchBuffSize));

    /// TODO: release
    semaphore_.Allocate(global_n_ranks_, global_rank_, [this](size_t size) {
        auto ptr = (uint64_t*)Allocate(size);
        Register(ptr, size);
        return get_symmetric_v2(ptr, 0);
    });

    check_cuda_error(cudaStreamSynchronize(0));

    Register(packet_buff_, kPacketBuffSize);
    Register(scratch_buff_, kScratchBuffSize);
}

CudaIpcCommImpl::~CudaIpcCommImpl()
{
    Deregister(scratch_buff_);
    Deregister(packet_buff_);
    // device_semaphores_ is not registered

    Free(scratch_buff_);
    Free(packet_buff_);

    for (auto i = (int)groups_.size() - 1; i >= 0; --i) {
        Free(groups_[i].d2d_semaphore_data);
    }

    semaphore_.Free([this](void* ptr) {
        Deregister(ptr);
        Free(ptr);
    });

    // for (const auto& [ptr, _] : registered_memories_) {
    //     TM_LOG_WARNING("[COMM][%d] Buffer %p is not deregistered", global_rank_, ptr);
    // }

    for (const auto& a : allocation_) {
        TM_LOG_WARNING("[COMM][%d] Allocation (%p, %lu) is not freed", global_rank_, a.uc_beg, a.size);
    }

    for (auto i = (int)groups_.size() - 1; i >= 0; --i) {
        cudaFreeAsync(groups_[i].d2d_semaphores, 0);
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
        CUmulticastObjectProp prop{};
        prop.numDevices = alloc_access_descs_.size();
        prop.size       = size;
        CUDRVCHECK(cuMulticastGetGranularity(&granularity, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    }
    else {
        CUDRVCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
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
        register_for_group(*alloc, alloc->uc_ptrs, i);
    }
}

void CudaIpcCommImpl::register_for_group(const Allocation& alloc, const std::vector<void*>& ucps, int group)
{
    auto size = alloc.size;

    auto& g = groups_.at(group);

    Symmetric s{};
    s.size   = size;
    s.uc_beg = alloc.uc_beg;
    s.uc_end = alloc.uc_end;

    for (auto r : g.l2g) {
        s.uc_ptrs.push_back(ucps[r]);
    }

    if (multicast_capability_) {
        CUmulticastObjectProp prop{};
        prop.numDevices = n_ranks(group);
        prop.size       = size;

        if (rank(group) == 0) {
            CUDRVCHECK(cuMulticastCreate(&s.mc_handle, &prop));
        }

        auto handles = comm::AllGather(h_comm_, s.mc_handle);
        s.mc_handle  = handles.at(g.l2g[0]);

        CUDRVCHECK(cuMulticastAddDevice(s.mc_handle, ordinals_[global_rank_]));

        // wait for all `cuMulticastAddDevice`
        h_comm_->Sync();

        CUDRVCHECK(cuMulticastBindMem(s.mc_handle, 0, alloc.handle, 0, size, 0));

        static_assert(sizeof(CUdeviceptr) == sizeof(void*));

        CUdeviceptr mc_ptr{};

        CUDRVCHECK(cuMemAddressReserve(&mc_ptr, size, alloc.alignment, 0, 0));
        CUDRVCHECK(cuMemMap(mc_ptr, size, 0, s.mc_handle, 0));
        CUDRVCHECK(cuMemSetAccess(mc_ptr, size, &alloc_access_descs_[global_rank_], 1));

        s.mc_ptr = reinterpret_cast<void*>(mc_ptr);
    }

    g.symmetric.insert(std::move(s));
}

void CudaIpcCommImpl::Deregister(void* ptr)
{
    // TODO: release multicast object
    for (size_t i = 0; i < groups_.size(); ++i) {
        auto& s = groups_[i].symmetric;
        if (auto it = s.find(ptr); it != s.end()) {
            s.erase(it);
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

uint64_t* CudaIpcCommImpl::create_semaphore_buffer()
{
    const int flags_size = 3 * sizeof(uint64_t) * kMaxChannels * (global_n_ranks_ - 1);
    uint64_t* flags      = (uint64_t*)Allocate(flags_size);
    check_cuda_error(cudaMemsetAsync(flags, 0, flags_size));
    return flags;
}

mscclpp::D2DSemaphoreHandle* CudaIpcCommImpl::init_semaphores(const std::vector<uint64_t*>& buffers, int group)
{
    const int n_ranks = this->n_ranks(group);
    const int rank    = this->rank(group);

    const int peers = n_ranks - 1;

    std::vector<mscclpp::D2DSemaphoreHandle> h_semaphores;
    for (int c = 0; c < kMaxChannels; ++c) {
        for (int r = 0; r < n_ranks; ++r) {
            if (r != rank) {
                const int p     = r < rank ? r : r - 1;
                const int inv_p = Rank{rank, peers}.inverse_peer(p);
                //
                mscclpp::D2DSemaphoreHandle handle{};
                handle.inboundSemaphoreId         = buffers[rank] + c * peers + p;                      // local
                handle.outboundSemaphoreId        = handle.inboundSemaphoreId + kMaxChannels * peers;   // local
                handle.expectedInboundSemaphoreId = handle.outboundSemaphoreId + kMaxChannels * peers;  // local
                handle.remoteInboundSemaphoreId   = buffers[r] + c * peers + inv_p;                     // near
                h_semaphores.push_back(handle);
            }
        }
    }

    mscclpp::D2DSemaphoreHandle* d_semaphores{};

    check_cuda_error(cudaMallocAsync(&d_semaphores, sizeof(mscclpp::D2DSemaphoreHandle) * h_semaphores.size(), 0));

    check_cuda_error(cudaMemcpyAsync(d_semaphores,
                                     h_semaphores.data(),
                                     sizeof(mscclpp::D2DSemaphoreHandle) * h_semaphores.size(),
                                     cudaMemcpyHostToDevice));

    return d_semaphores;
}

auto CudaIpcCommImpl::get_symmetric_impl(void* ptr, int group) -> SymmetricPtr<void>
{
    auto& g = groups_.at(group);

    auto symm = g.symmetric.find(ptr);
    TM_CHECK(symm != g.symmetric.end());

    auto offset = (char*)ptr - (char*)symm->uc_beg;

    const int rank = this->rank(group);

    SymmetricPtr<void> p{};

    TM_CHECK_LE(symm->uc_ptrs.size(), p.uc.size() + 1);

    for (size_t i = 0, j = 0; i < symm->uc_ptrs.size(); ++i) {
        if (i != rank) {
            p.uc[j++] = (char*)symm->uc_ptrs[i] + offset;
        }
    }
    if (symm->mc_ptr) {
        p.mc = (char*)symm->mc_ptr + offset;
    }

    return p;
}

auto CudaIpcCommImpl::get_symmetric_v2_impl(void* ptr, int group) -> SymmetricPtr_V2<void>
{
    auto& g = groups_.at(group);

    auto symm = g.symmetric.find(ptr);
    TM_CHECK(symm != g.symmetric.end());

    auto offset = (char*)ptr - (char*)symm->uc_beg;

    SymmetricPtr_V2<void> p{};

    TM_CHECK_LE(symm->uc_ptrs.size(), p.uc.size());

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
