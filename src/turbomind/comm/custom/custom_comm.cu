// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>
#include <mutex>

#include <cuda.h>
#include <vector>

#include "src/turbomind/comm/comm.h"
#include "src/turbomind/comm/custom/custom_comm.h"

#include "mscclpp/core.hpp"
#include "mscclpp/semaphore_device.hpp"

#include "src/turbomind/comm/custom/bootstrap.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind::comm {

CustomComm::CustomComm(std::shared_ptr<mscclpp::Bootstrap> bootstrap):
    Comm{bootstrap->getNranks(), bootstrap->getRank()}
{
    comm_ = std::make_shared<mscclpp::Communicator>(std::move(bootstrap));

    // Exchange device ordinals
    ordinals_.resize(world_size_);
    check_cuda_error(cudaGetDevice(&ordinals_[rank_]));
    comm_->bootstrap()->allGather(ordinals_.data(), sizeof(int));

    // Prepare allocation properties & granularity
    alloc_prop_.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    alloc_prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    alloc_prop_.location.id   = rank_;
    CUDRVCHECK(cuMemGetAllocationGranularity(&alloc_granularity_, &alloc_prop_, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    // Prepare access descriptors
    alloc_access_descs_.resize(world_size_);
    for (int r = 0; r < world_size_; ++r) {
        alloc_access_descs_[r].location.id   = ordinals_[r];
        alloc_access_descs_[r].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        alloc_access_descs_[r].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
}

CustomComm::~CustomComm()
{
    Deregister(scratch_buff_);
    Deregister(packet_buff_);
    // device_semaphores_ is not registered

    Free(scratch_buff_);
    Free(packet_buff_);
    Free(device_semaphore_data_);

    for (const auto& [ptr, _] : registered_memories_) {
        TM_LOG_WARNING("[TM][COMM][%d] Buffer %p is not deregistered", rank_, ptr);
    }

    for (const auto& [ptr, alloc] : allocations_) {
        TM_LOG_WARNING("[TM][COMM][%d] Allocation (%p, %lu) is not freed", rank_, ptr, alloc.size);
    }

    check_cuda_error(cudaFreeAsync(device_syncer_, 0));
    check_cuda_error(cudaFreeAsync(device_semaphores_, 0));
    check_cuda_error(cudaStreamSynchronize(0));

    comm_.reset();
}

void CustomComm::Initialize()
{
    const int flags_size = 3 * sizeof(uint64_t) * kChannelsPerConn * (world_size_ - 1);
    uint64_t* flags      = (uint64_t*)Allocate(flags_size);
    check_cuda_error(cudaMemsetAsync(flags, 0, flags_size));
    device_semaphore_data_ = flags;

    std::vector<uint64_t*> all_flags(world_size_);
    all_flags[rank_] = flags;
    comm_->bootstrap()->allGather(all_flags.data(), sizeof(uint64_t*));

    const int peers = world_size_ - 1;

    std::vector<mscclpp::SmDevice2DeviceSemaphoreDeviceHandle> device_semaphores;
    for (int c = 0; c < kChannelsPerConn; ++c) {
        for (int r = 0; r < world_size_; ++r) {
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

    packet_buff_ = Allocate(kPacketBuffSize);
    check_cuda_error(cudaMemsetAsync(packet_buff_, 0, kPacketBuffSize));

    scratch_buff_ = Allocate(kScratchBuffSize);
    check_cuda_error(cudaMemsetAsync(scratch_buff_, 0, kScratchBuffSize));

    check_cuda_error(cudaMallocAsync(&device_syncer_, sizeof(mscclpp::DeviceSyncer), 0));
    check_cuda_error(cudaMemsetAsync(device_syncer_, 0, sizeof(mscclpp::DeviceSyncer)));

    check_cuda_error(cudaStreamSynchronize(0));

    Register(packet_buff_, kPacketBuffSize);
    Register(scratch_buff_, kScratchBuffSize);
}

void* CustomComm::Allocate(size_t size)
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

void CustomComm::Free(void* ptr)
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
        TM_LOG_WARNING("[TM][COMM][%d] Freeing %p which is not allocated by this module", rank_, ptr);
    }
}

void CustomComm::Register(void* ptr, size_t size)
{
    if (!registered_memories_.count(ptr)) {
        using Buffer = std::pair<void*, size_t>;

        std::vector<Buffer> buffers(world_size_);
        buffers[rank_] = {ptr, size};
        comm_->bootstrap()->allGather(buffers.data(), sizeof(Buffer));

        std::vector<Buffer> bufs;
        for (int i = 0; i < world_size_; ++i) {
            if (i != rank_) {
                bufs.push_back(buffers[i]);
            }
        }

        registered_memories_.emplace(ptr, std::move(bufs));
    }
    else {
        TM_LOG_WARNING("[TM][COMM][%d] Duplicated registration on (%p, %lu)", rank_, ptr, size);
    }
}

void CustomComm::Deregister(void* ptr)
{
    if (int erased = registered_memories_.erase(ptr); erased == 0) {
        TM_LOG_WARNING("[TM][COMM][%d] Deregistering non-registered address %p", rank_, ptr);
    }
}

int CustomComm::Query(QueryAttr attr) const noexcept
{
    if (attr == kHasAllGather2D) {
        return 1;
    }
    return 0;
}

Array<void*, kMaxNearPeers> CustomComm::get_near_impl(void* ptr)
{
    auto& memories = registered_memories_.at(ptr);
    FT_CHECK(memories.size() <= kMaxNearPeers);
    Array<void*, kMaxNearPeers> ret{};
    for (size_t i = 0; i < memories.size(); ++i) {
        ret[i] = memories[i].first;
    }
    return ret;
}

class LocalGroupId: public GroupId {
public:
    void Initialize() override
    {
        internal_ = std::make_shared<Internal>();
    }

    void Export(std::ostream& os) override
    {
        FT_CHECK((bool)internal_);  // `Initialize` must come befor `Export`

        const void* ptr = this;
        os.write((const char*)&ptr, sizeof(ptr));
    }

    void Import(std::istream& is) override
    {
        void* ptr{};
        is.read((char*)&ptr, sizeof(ptr));
        internal_ = reinterpret_cast<LocalGroupId*>(ptr)->internal_;

        FT_CHECK((bool)internal_);
    }

    std::unique_ptr<Comm> CreateCommunicator(int rank, int world_size) override
    {
        auto init_shared_state = [&] {  //
            internal_->state = std::make_shared<LocalBootstrap::State>(world_size);
        };

        FT_CHECK((bool)internal_);

        // One of the rank initialize the shared state
        std::call_once(internal_->flag, init_shared_state);

        FT_CHECK((bool)internal_->state);

        auto bootstrap = std::make_shared<LocalBootstrap>(world_size, rank, internal_->state);

        auto comm = std::make_unique<CustomComm>(bootstrap);

        comm->Initialize();

        return comm;
    }

private:
    struct Internal {
        std::once_flag                         flag;
        std::shared_ptr<LocalBootstrap::State> state;
    };

private:
    std::shared_ptr<Internal> internal_;
};

std::unique_ptr<GroupId> CreateCustomGroupId()
{
    return std::make_unique<LocalGroupId>();
}

}  // namespace turbomind::comm
