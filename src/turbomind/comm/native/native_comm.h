// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <unordered_map>

#include "src/turbomind/comm/comm.h"
#include "src/turbomind/comm/host.h"
#include "src/turbomind/comm/native/mscclpp.h"

#include "src/turbomind/kernels/core/array.h"

#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

static constexpr int kMaxNearPeers = 7;

class NativeCommImpl: public DeviceCommImpl {
public:
    static constexpr int kPacketBuffSize  = 8 << 20;  // 8 MB
    static constexpr int kScratchBuffSize = 8 << 20;  // 8 MB
    static constexpr int kChannelsPerConn = 64;

    ~NativeCommImpl() override;

    explicit NativeCommImpl(HostComm h_comm);

    void Initialize();

    int n_ranks(int group) const override
    {
        return groups_.at(group).l2g.size();
    }

    int rank(int group) const override
    {
        return groups_.at(group).g2l.at(global_rank_);
    }

    void* Allocate(size_t size) override;

    void Free(void* ptr) override;

    void Register(void* ptr, size_t size) override;

    void Deregister(void* ptr) override;

    int Split(int color, int key, int group) override;

    int Query(QueryAttr attr) const noexcept override;

    void AllReduceSum(
        const void* sendbuff, void* recvbuff, size_t count, DataType type, int group, cudaStream_t stream) override;

    void AllGather(
        const void* sendbuff, void* recvbuff, size_t sendcount, DataType type, int group, cudaStream_t stream) override;

    void AllreduceResidualBiasRMSnorm(void*        hidden,
                                      void*        residual,
                                      const void*  bias,
                                      const void*  weights,
                                      float        eps,
                                      int          dim,
                                      int          token_num,
                                      DataType     dtype,
                                      int          group,
                                      cudaStream_t stream) override;

    void AllGather2D(const void*  sendbuff,
                     void*        recvbuff,
                     size_t       pitch,
                     size_t       stride,
                     int          width,
                     int          height,
                     DataType     type,
                     int2         flags,
                     int          group,
                     cudaStream_t stream) override;

private:
    uint64_t* create_semaphore_buffer();

    mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* init_semaphores(const std::vector<uint64_t*>& buffers, int group);

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
    HostComm h_comm_;

    int global_n_ranks_;
    int global_rank_;

    std::vector<int> ordinals_;

    std::unordered_map<void*, std::vector<std::pair<void*, size_t>>> registered_memories_;

    void*    packet_buff_{};
    void*    scratch_buff_{};
    uint32_t flag_{1};

    struct Allocation {
        CUmemGenericAllocationHandle handle;
        size_t                       size;
    };

    CUmemAllocationProp          alloc_prop_{};
    size_t                       alloc_granularity_{};
    std::vector<CUmemAccessDesc> alloc_access_descs_{};

    std::unordered_map<void*, Allocation> allocations_;

    struct Group {
        std::vector<int> l2g;
        std::vector<int> g2l;

        uint64_t* d2d_semaphore_data;

        mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* d2d_semaphores;
    };

    std::vector<Group> groups_;
};

struct Rank {
    int                     rank;
    int                     peers;
    __host__ __device__ int get_next_peer(int i)
    {
        return i + rank < peers ? i + rank : i + rank - peers;
    }
    __host__ __device__ int get_prev_peer(int i)
    {
        return get_next_peer(peers - 1 - i);
    }
    __host__ __device__ int get_peer_rank(int p)  // rank of `p`
    {
        return p < rank ? p : p + 1;
    }
    __host__ __device__ int inverse_peer(int p)  // peer idx of `rank` on peer `p`
    {
        return p < rank ? rank - 1 : rank;
    }
};

}  // namespace turbomind::comm
