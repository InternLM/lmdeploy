
#pragma once

#include <stdexcept>
#include <unordered_map>

#include "mscclpp/concurrency_device.hpp"
#include "mscclpp/core.hpp"
#include "mscclpp/semaphore.hpp"
#include "mscclpp/sm_channel.hpp"

#include "src/turbomind/comm/comm.h"
#include "src/turbomind/utils/Tensor.h"

namespace turbomind {

static inline size_t elem_size(DataType type)
{
    switch (type) {
        case DataType::TYPE_FP16:
        case DataType::TYPE_BF16:
        case DataType::TYPE_INT16:
            return 2;
        case DataType::TYPE_FP32:
            return 4;
        case DataType::TYPE_UINT8:
            return 1;
        default:
            throw std::runtime_error("not supported");
    }
}

class CustomComm: public Comm {
public:
    static constexpr int kScratchBuffSize = 64 << 20;
    static constexpr int kChannelsPerConn = 64;

    CustomComm(std::shared_ptr<mscclpp::Bootstrap> bootstrap);

    void Initialize();

    void RegisterBuffer(void* ptr, size_t size) override;

    void AllReduceSum(const void* sendbuff, void* recvbuff, size_t count, DataType type, cudaStream_t stream) override;

    void AllGather(const void* sendbuff, void* recvbuff, size_t sendcount, DataType type, cudaStream_t stream) override;

    void AllreduceResidualBiasRMSnorm(void*        hidden,
                                      void*        residual,
                                      const void*  bias,
                                      const void*  weights,
                                      float        eps,
                                      int          dim,
                                      int          token_num,
                                      DataType     dtype,
                                      cudaStream_t stream) override;

private:
    std::shared_ptr<mscclpp::Communicator>            comm_;
    std::vector<std::shared_ptr<mscclpp::Connection>> connections_;

    std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>>   semaphores_;
    std::unordered_map<void*, std::vector<mscclpp::SmChannel>>        registered_channels_;
    std::unordered_map<void*, std::vector<mscclpp::RegisteredMemory>> registered_memories_;

    void*                           packet_buff_{};
    std::vector<mscclpp::SmChannel> packet_chns_;
    void*                           scratch_buff_{};
    uint32_t                        flag_{1};

    mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* device_semaphores_;
    mscclpp::DeviceSyncer*                         device_syncer_{};
};

std::vector<std::unique_ptr<Comm>> CreateCustomComm(const std::vector<int>& devices);

}  // namespace turbomind