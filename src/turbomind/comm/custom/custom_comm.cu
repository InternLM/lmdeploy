

#include "mscclpp/core.hpp"
#include "src/turbomind/comm/custom/bootstrap.h"
#include "src/turbomind/comm/custom/custom_comm.h"

#include "mscclpp/semaphore_device.hpp"

namespace turbomind {

CustomComm::CustomComm(std::shared_ptr<mscclpp::Bootstrap> bootstrap):
    Comm{bootstrap->getNranks(), bootstrap->getRank()}
{
    comm_ = std::make_shared<mscclpp::Communicator>(std::move(bootstrap));
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

    cudaMallocAsync(
        &device_semaphores_, sizeof(mscclpp::SmDevice2DeviceSemaphoreDeviceHandle) * semaphores_.size(), {});
    std::vector<mscclpp::SmDevice2DeviceSemaphoreDeviceHandle> device_semaphores;
    for (auto& s : semaphores_) {
        device_semaphores.push_back(s->deviceHandle());
    }
    cudaMemcpyAsync(device_semaphores_,
                    device_semaphores.data(),
                    sizeof(mscclpp::SmDevice2DeviceSemaphoreDeviceHandle) * semaphores_.size(),
                    cudaMemcpyDefault,
                    {});

    cudaMallocAsync(&device_syncer_, sizeof(mscclpp::DeviceSyncer), {});
    cudaMemsetAsync(device_syncer_, 0, sizeof(mscclpp::DeviceSyncer), {});
    cudaStreamSynchronize({});

    cudaMalloc(&packet_buff_, kScratchBuffSize);
    cudaMemset(packet_buff_, 0, kScratchBuffSize);

    RegisterBuffer(packet_buff_, kScratchBuffSize);
    packet_chns_ = registered_channels_.at(packet_buff_);

    cudaMalloc(&scratch_buff_, kScratchBuffSize);
}

void CustomComm::RegisterBuffer(void* ptr, size_t size)
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

    registered_channels_.emplace(ptr, std::move(channels));
    registered_memories_.emplace(ptr, std::move(memories));
}

std::vector<std::unique_ptr<Comm>> CreateCustomComm(const std::vector<int>& devices)
{
    const int num   = devices.size();
    auto      state = std::make_shared<LocalBootstrap::State>(num);

    std::vector<std::unique_ptr<Comm>> comm;

    for (int i = 0; i < num; ++i) {
        auto bootstrap = std::make_shared<LocalBootstrap>(num, i, state);
        comm.push_back(std::make_unique<CustomComm>(std::static_pointer_cast<mscclpp::Bootstrap>(bootstrap)));
    }

    std::vector<std::thread> threads;

    for (const auto& c : comm) {
        threads.emplace_back([&] {
            cudaSetDevice(devices[c->rank()]);
            for (int i = 0; i < c->world_size(); ++i) {
                if (i != c->rank()) {
                    cudaDeviceEnablePeerAccess(devices[i], 0);
                }
            }
            ((CustomComm&)*c).Initialize();
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    return comm;
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

}  // namespace turbomind