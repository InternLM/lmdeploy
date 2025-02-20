
#pragma once

#ifdef _CLANGD
#define MSCCLPP_DEVICE_COMPILE 1
#define MSCCLPP_DEVICE_CUDA 1

#ifdef MSCCLPP_HOST_DEVICE_INLINE
#undef MSCCLPP_HOST_DEVICE_INLINE
#endif

#define MSCCLPP_HOST_DEVICE_INLINE __host__ __device__ __inline__
#define MSCCLPP_DEVICE_INLINE __device__ __inline__
#endif

#include <unordered_map>

#include "mscclpp/concurrency_device.hpp"
#include "mscclpp/core.hpp"
#include "mscclpp/semaphore.hpp"
#include "mscclpp/sm_channel.hpp"

#include "src/turbomind/comm/comm.h"
#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

static constexpr int kMaxNearPeers = 7;

class CustomComm: public Comm {
public:
    static constexpr int kPacketBuffSize  = 16 << 20;
    static constexpr int kScratchBuffSize = 64 << 20;
    static constexpr int kChannelsPerConn = 64;

    ~CustomComm() override;

    CustomComm(std::shared_ptr<mscclpp::Bootstrap> bootstrap);

    void Initialize();

    void* Allocate(size_t size) override;

    void Free(void* ptr) override;

    void Register(void* ptr, size_t size) override;

    void Deregister(void* ptr) override;

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
    template<class T>
    inline Array<T*, kMaxNearPeers> get_near(T* ptr)
    {
        auto                     src = get_near_impl(ptr);
        Array<T*, kMaxNearPeers> dst;
        for (int i = 0; i < dst.size(); ++i) {
            dst[i] = static_cast<T*>(src[i]);
        }
        return dst;
    }

    Array<void*, kMaxNearPeers> get_near_impl(void* ptr);

private:
    std::shared_ptr<mscclpp::Communicator>            comm_;
    std::vector<std::shared_ptr<mscclpp::Connection>> connections_;

    std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>>   semaphores_;
    std::unordered_map<void*, std::vector<mscclpp::SmChannel>>        registered_channels_;
    std::unordered_map<void*, std::vector<mscclpp::RegisteredMemory>> registered_memories_;

    void*    packet_buff_{};
    void*    scratch_buff_{};
    uint32_t flag_{1};

    mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* device_semaphores_;
    mscclpp::DeviceSyncer*                         device_syncer_{};
};

std::vector<std::unique_ptr<Comm>> CreateCustomComm(const std::vector<int>& devices);

struct Rank {
    int            rank;
    int            peers;
    __device__ int get_next_peer(int i)
    {
        return i + rank < peers ? i + rank : i + rank - peers;
    }
    __device__ int get_prev_peer(int i)
    {
        return get_next_peer(peers - 1 - i);
    }
    __device__ int get_peer_rank(int p)  // rank of `p`
    {
        return p < rank ? p : p + 1;
    }
    __device__ int inverse_peer(int p)  // peer idx of `rank` on peer `p`
    {
        return p < rank ? rank - 1 : rank;
    }
};

}  // namespace turbomind::comm