
#include <memory>
#include <mutex>

#include <cuda.h>

#include "src/turbomind/comm/custom/custom_comm.h"

#include "mscclpp/core.hpp"
#include "mscclpp/semaphore_device.hpp"

#include "src/turbomind/comm/custom/bootstrap.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

CustomComm::CustomComm(std::shared_ptr<mscclpp::Bootstrap> bootstrap):
    Comm{bootstrap->getNranks(), bootstrap->getRank()}
{
    comm_ = std::make_shared<mscclpp::Communicator>(std::move(bootstrap));
}

CustomComm::~CustomComm()
{
    Free(scratch_buff_);
    Free(packet_buff_);
    Free(device_syncer_);
    Free(device_semaphores_);

    // make destruction order explicit
    registered_channels_.clear();  // channels are constructed on memories
    registered_memories_.clear();
    semaphores_.clear();
    connections_.clear();
    comm_.reset();
}

void CustomComm::Initialize()
{
    FT_CHECK(comm_->bootstrap()->getNranks() == comm_->bootstrap()->getNranksPerNode());
    comm_->bootstrap()->barrier();
    {
        std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connections;
        for (int i = 0; i < world_size_; ++i) {
            if (i == rank_) {
                continue;
            }
            connections.push_back(comm_->connectOnSetup(i, 0, mscclpp::Transport::CudaIpc));
        }
        comm_->setup();
        for (auto& c : connections) {
            connections_.push_back(c.get());
        }
    }

    for (int c = 0; c < kChannelsPerConn; ++c) {
        for (size_t i = 0; i < connections_.size(); ++i) {
            semaphores_.push_back(std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*comm_, connections_[i]));
        }
    }

    comm_->setup();

    device_semaphores_ = (mscclpp::SmDevice2DeviceSemaphoreDeviceHandle*)Allocate(
        sizeof(mscclpp::SmDevice2DeviceSemaphoreDeviceHandle) * semaphores_.size());

    std::vector<mscclpp::SmDevice2DeviceSemaphoreDeviceHandle> device_semaphores;
    for (auto& s : semaphores_) {
        device_semaphores.push_back(s->deviceHandle());
    }
    check_cuda_error(cudaMemcpy(device_semaphores_,
                                device_semaphores.data(),
                                sizeof(mscclpp::SmDevice2DeviceSemaphoreDeviceHandle) * semaphores_.size(),
                                cudaMemcpyHostToDevice));

    device_syncer_ = (mscclpp::DeviceSyncer*)Allocate(sizeof(mscclpp::DeviceSyncer));
    check_cuda_error(cudaMemset(device_syncer_, 0, sizeof(mscclpp::DeviceSyncer)));

    packet_buff_ = Allocate(kPacketBuffSize);
    check_cuda_error(cudaMemset(packet_buff_, 0, kPacketBuffSize));

    scratch_buff_ = Allocate(kScratchBuffSize);
    check_cuda_error(cudaMemset(scratch_buff_, 0, kScratchBuffSize));

    Register(packet_buff_, kPacketBuffSize);
    Register(scratch_buff_, kScratchBuffSize);
}

void* CustomComm::Allocate(size_t size)
{
    void* ptr{};
    check_cuda_error(cudaMalloc(&ptr, size));
    return ptr;
}

void CustomComm::Free(void* ptr)
{
    check_cuda_error(cudaFree(ptr));
}

void CustomComm::Register(void* ptr, size_t size)
{
    FT_CHECK(registered_channels_.count(ptr) == 0);

    mscclpp::RegisteredMemory memory = comm_->registerMemory(ptr, size, mscclpp::Transport::CudaIpc);
    std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> futures;

    for (int i = 0; i < world_size_; ++i) {
        if (i == rank_) {
            continue;
        }
        futures.push_back(comm_->recvMemoryOnSetup(i, 0));
        comm_->sendMemoryOnSetup(memory, i, 0);
    }

    comm_->setup();

    std::vector<mscclpp::SmChannel>        channels;
    std::vector<mscclpp::RegisteredMemory> memories;

    for (size_t i = 0; i < connections_.size(); ++i) {
        mscclpp::RegisteredMemory remote_memory = futures[i].get();
        memories.push_back(remote_memory);
        channels.emplace_back(semaphores_[i], remote_memory, ptr, nullptr);
    }

    registered_memories_.emplace(ptr, std::move(memories));
    registered_channels_.emplace(ptr, std::move(channels));
}

void CustomComm::Deregister(void* ptr)
{
    registered_channels_.erase(ptr);
    registered_memories_.erase(ptr);
}

Array<void*, kMaxNearPeers> CustomComm::get_near_impl(void* ptr)
{
    auto& memories = registered_memories_.at(ptr);
    FT_CHECK(memories.size() <= kMaxNearPeers);
    Array<void*, kMaxNearPeers> ret{};
    for (size_t i = 0; i < memories.size(); ++i) {
        ret[i] = memories[i].data();
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

        // one of the rank initialize the shared state
        std::call_once(internal_->flag, init_shared_state);

        FT_CHECK((bool)internal_->state);

        auto bootstrap = std::make_shared<LocalBootstrap>(world_size, rank, internal_->state);

        std::vector<CUcontext> ctx(world_size);
        CUDRVCHECK(cuCtxGetCurrent(&ctx[rank]));

        bootstrap->allGather(ctx.data(), sizeof(CUcontext));

        for (int i = 0; i < world_size; ++i) {
            if (i != rank) {
                auto ec = cuCtxEnablePeerAccess(ctx[i], 0);
                FT_CHECK(ec == CUDA_SUCCESS || ec == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);
            }
        }

        bootstrap->barrier();

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