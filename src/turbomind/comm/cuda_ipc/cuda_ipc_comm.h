// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda.h>
#include <set>

#include "src/turbomind/comm/cuda_ipc/common.h"
#include "src/turbomind/comm/cuda_ipc/semaphore.h"
#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/host_comm.h"

#include "src/turbomind/kernels/core/array.h"

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

class MaxCtas {
public:
    MaxCtas(int value = 0): is_set_{}, value_{value} {}

    void set_value(int value)
    {
        value_  = value;
        is_set_ = true;
    }

    int value()
    {
        return value_;
    }

    int apply(int _default)
    {
        if (!is_set_) {  // `value_` is max possible value in this case
            return std::min(_default, value_);
        }
        else {
            return value_;
        }
    }

private:
    bool is_set_;
    int  value_;
};

class CudaIpcCommImpl: public DeviceCommImpl {
    struct Allocation;
    struct Symmetric;

public:
    ~CudaIpcCommImpl() override;

    explicit CudaIpcCommImpl(HostComm h_comm);

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

    void Broadcast(const void*  sendbuff,
                   void*        recvbuff,
                   size_t       count,
                   DataType     type,
                   int          root,
                   int          group,
                   cudaStream_t stream) override;

    void
    Gather(const void* sendbuff, void* recvbuff, size_t count, DataType type, int root, int group, cudaStream_t stream);

    void Barrier(int group, cudaStream_t stream);

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

    void AllreduceResidualBiasRMSnormEx(void*        hidden,
                                        void*        residual,
                                        const void*  bias,
                                        const void*  weights,
                                        float        eps,
                                        int          dim,
                                        DataType     type,
                                        int          group0,
                                        int          group1,
                                        const int*   local_token_nums,
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
    template<class T>
    inline SymmetricPtr_V2<T> get_symmetric_v2(T* ptr, int group)
    {
        auto               tmp = get_symmetric_v2_impl(ptr, group);
        SymmetricPtr_V2<T> ret{};
        ret.mc = static_cast<T*>(tmp.mc);
        for (int i = 0; i < ret.uc.size(); ++i) {
            ret.uc[i] = static_cast<T*>(tmp.uc[i]);
        }
        return ret;
    }

    SymmetricPtr_V2<void> get_symmetric_v2_impl(void* ptr, int group);

    void Register(const Allocation& alloc, int group);

    void Deregister(Symmetric& s);

private:
    HostComm h_comm_;

    int global_n_ranks_;
    int global_rank_;

    std::vector<int> ordinals_;

    struct Symmetric {
        void*              uc_beg;
        void*              uc_end;
        size_t             size;
        std::vector<void*> uc_ptrs;  // peers
        void*              mc_ptr;

        CUmemGenericAllocationHandle mc_handle;

        friend bool operator<(const Symmetric& a, const Symmetric& b)
        {
            return (char*)a.uc_beg < (char*)b.uc_beg;
        }
        friend bool operator<(const Symmetric& a, void* b)
        {
            return (char*)a.uc_end <= (char*)b;
        }
        friend bool operator<(void* a, const Symmetric& b)
        {
            return (char*)a < (char*)b.uc_beg;
        }
    };

    void*    packet_buff_{};
    void*    scratch_buff_{};
    uint32_t flag_{1};

    struct Allocation {
        void*                        uc_beg;
        void*                        uc_end;
        size_t                       size;
        size_t                       alignment;
        std::vector<void*>           uc_ptrs;  // ranks
        CUmemGenericAllocationHandle handle;

        friend bool operator<(const Allocation& a, const Allocation& b)
        {
            return (char*)a.uc_beg < (char*)b.uc_beg;
        }
        friend bool operator<(const Allocation& a, void* b)
        {
            return (char*)a.uc_end <= (char*)b;
        }
        friend bool operator<(void* a, const Allocation& b)
        {
            return (char*)a < (char*)b.uc_beg;
        }
    };

    std::vector<CUmemAccessDesc> alloc_access_descs_{};

    int multicast_capability_{false};

    std::set<Allocation, std::less<>> allocation_;

    struct Group {
        std::vector<int> l2g;  // local -> global
        std::vector<int> g2l;  // global -> local

        SystemSemaphoreStorage semaphore;

        std::set<Symmetric, std::less<>> symmetric;
    };

    std::vector<Group> groups_;

    MaxCtas max_ctas_;
    size_t  copy_threshold_{INT64_MAX};
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
