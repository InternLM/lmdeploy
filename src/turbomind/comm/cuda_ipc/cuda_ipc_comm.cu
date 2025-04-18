// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>
#include <numeric>
#include <vector>

#include <cuda.h>

#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/host_comm.h"

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

    // Prepare allocation properties & granularity
    alloc_prop_.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    alloc_prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    alloc_prop_.location.id   = ordinals_[rank];
    CUDRVCHECK(cuMemGetAllocationGranularity(&alloc_granularity_, &alloc_prop_, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

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

    for (const auto& [ptr, _] : registered_memories_) {
        TM_LOG_WARNING("[COMM][%d] Buffer %p is not deregistered", global_rank_, ptr);
    }

    for (const auto& [ptr, alloc] : allocations_) {
        TM_LOG_WARNING("[COMM][%d] Allocation (%p, %lu) is not freed", global_rank_, ptr, alloc.size);
    }

    for (auto i = (int)groups_.size() - 1; i >= 0; --i) {
        cudaFreeAsync(groups_[i].d2d_semaphores, 0);
    }

    cudaStreamSynchronize(0);
}

void CudaIpcCommImpl::Initialize()
{
#if 0
    const int flags_size = 3 * sizeof(uint64_t) * kChannelsPerConn * (n_ranks_ - 1);
    uint64_t* flags      = (uint64_t*)Allocate(flags_size);
    check_cuda_error(cudaMemsetAsync(flags, 0, flags_size));
    device_semaphore_data_ = flags;

    auto all_flags = comm::AllGather(h_comm_, flags);

    const int peers = n_ranks_ - 1;

    std::vector<mscclpp::SmDevice2DeviceSemaphoreDeviceHandle> device_semaphores;
    for (int c = 0; c < kChannelsPerConn; ++c) {
        for (int r = 0; r < n_ranks_; ++r) {
            if (r != rank_) {
                const int p     = r < rank_ ? r : r - 1;
                const int inv_p = Rank{rank_, peers}.inverse_peer(p);
                //
                mscclpp::SmDevice2DeviceSemaphoreDeviceHandle handle{};
                handle.inboundSemaphoreId         = flags + c * peers + p;                                  // local
                handle.outboundSemaphoreId        = handle.inboundSemaphoreId + kChannelsPerConn * peers;   // local
                handle.expectedInboundSemaphoreId = handle.outboundSemaphoreId + kChannelsPerConn * peers;  // local
                handle.remoteInboundSemaphoreId   = all_flags[r] + c * peers + inv_p;                       // near
                device_semaphores.push_back(handle);
            }
        }
    }
    check_cuda_error(cudaMallocAsync(
        &device_semaphores_, sizeof(mscclpp::SmDevice2DeviceSemaphoreDeviceHandle) * device_semaphores.size(), 0));
    check_cuda_error(cudaMemcpyAsync(device_semaphores_,
                                     device_semaphores.data(),
                                     sizeof(mscclpp::SmDevice2DeviceSemaphoreDeviceHandle) * device_semaphores.size(),
                                     cudaMemcpyHostToDevice));
#endif
}

void* CudaIpcCommImpl::Allocate(size_t size)
{
    CUmemGenericAllocationHandle handle{};
    size = (size + alloc_granularity_ - 1) / alloc_granularity_ * alloc_granularity_;
    CUDRVCHECK(cuMemCreate(&handle, size, &alloc_prop_, 0));
    CUdeviceptr dptr{};
    CUDRVCHECK(cuMemAddressReserve(&dptr, size, 0, 0, 0));
    CUDRVCHECK(cuMemMap(dptr, size, 0, handle, 0));
    CUDRVCHECK(cuMemSetAccess(dptr, size, alloc_access_descs_.data(), alloc_access_descs_.size()));
    void* ptr = reinterpret_cast<void*>(dptr);
    allocations_.emplace(ptr, Allocation{handle, size});
    return ptr;
}

void CudaIpcCommImpl::Free(void* ptr)
{
    if (auto it = allocations_.find(ptr); it != allocations_.end()) {
        auto allocation = it->second;
        auto dptr       = reinterpret_cast<CUdeviceptr>(ptr);
        CUDRVCHECK(cuMemUnmap(dptr, allocation.size));
        CUDRVCHECK(cuMemRelease(allocation.handle));
        CUDRVCHECK(cuMemAddressFree(dptr, allocation.size));
        allocations_.erase(it);
    }
    else {
        TM_LOG_WARNING("[TM][COMM][%d] Freeing %p which is not allocated by this module", global_rank_, ptr);
    }
}

void CudaIpcCommImpl::Register(void* ptr, size_t size)
{
    if (!registered_memories_.count(ptr)) {
        auto buffers = comm::AllGather(h_comm_, std::make_pair(ptr, size));
        buffers.erase(buffers.begin() + global_rank_);
        registered_memories_.emplace(ptr, std::move(buffers));
    }
    else {
        TM_LOG_WARNING("[TM][COMM][%d] Duplicated registration on (%p, %lu)", global_rank_, ptr, size);
    }
}

void CudaIpcCommImpl::Deregister(void* ptr)
{
    if (int erased = registered_memories_.erase(ptr); erased == 0) {
        TM_LOG_WARNING("[TM][COMM][%d] Deregistering non-registered address %p", global_rank_, ptr);
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
    const int flags_size = 3 * sizeof(uint64_t) * kChannelsPerConn * (global_n_ranks_ - 1);
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
    for (int c = 0; c < kChannelsPerConn; ++c) {
        for (int r = 0; r < n_ranks; ++r) {
            if (r != rank) {
                const int p     = r < rank ? r : r - 1;
                const int inv_p = Rank{rank, peers}.inverse_peer(p);
                //
                mscclpp::D2DSemaphoreHandle handle{};
                handle.inboundSemaphoreId         = buffers[rank] + c * peers + p;                          // local
                handle.outboundSemaphoreId        = handle.inboundSemaphoreId + kChannelsPerConn * peers;   // local
                handle.expectedInboundSemaphoreId = handle.outboundSemaphoreId + kChannelsPerConn * peers;  // local
                handle.remoteInboundSemaphoreId   = buffers[r] + c * peers + inv_p;                         // near
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

Array<void*, kMaxNearPeers> CudaIpcCommImpl::get_symmetric_impl(void* ptr, int group)
{
    auto& memories = registered_memories_.at(ptr);
    FT_CHECK(memories.size() <= kMaxNearPeers);
    std::vector<void*> tmp(memories.size());
    for (size_t i = 0; i < memories.size(); ++i) {
        tmp[i] = memories[i].first;
    }
    // Put current rank back
    tmp.insert(tmp.begin() + global_rank_, ptr);
    Array<void*, kMaxNearPeers> ret{};
    // Indexed copy by l2g map
    int p = 0;
    for (const auto& r : groups_.at(group).l2g) {
        if (r != global_rank_) {  // Skip current rank
            ret[p++] = tmp[r];
        }
    }
    return ret;
}

DeviceComm CreateCudaIpcCommunicator(int n_ranks, int rank, HostComm h_comm)
{
    auto comm = std::make_unique<CudaIpcCommImpl>(h_comm);

    comm->Initialize();

    return DeviceComm{std::move(comm)};
}

}  // namespace turbomind::comm
