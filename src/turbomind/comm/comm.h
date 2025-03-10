// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>

#include <ostream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "src/turbomind/comm/host.h"
#include "src/turbomind/utils/Tensor.h"

namespace turbomind::comm {

enum QueryAttr {
    kHasAllGather2D
};

class Comm {
public:
    virtual ~Comm() = default;

    Comm(int world_size, int rank): world_size_{world_size}, rank_{rank} {}

    int rank() const noexcept
    {
        return rank_;
    }

    int world_size() const noexcept
    {
        return world_size_;
    }

    virtual void* Allocate(size_t size) = 0;

    virtual void Free(void* ptr) = 0;

    virtual void Register(void* ptr, size_t size) = 0;

    virtual void Deregister(void* ptr) = 0;

    virtual int Split(int color, int key, int group)
    {
        return -1;
    }

    virtual int Query(QueryAttr attr) const noexcept = 0;

    template<class T>
    void AllReduceSum(const T* sendbuff, T* recvbuff, size_t count, cudaStream_t stream)
    {
        return AllReduceSum(sendbuff, recvbuff, count, getTensorType<T>(), stream);
    }

    virtual void
    AllReduceSum(const void* sendbuff, void* recvbuff, size_t count, DataType type, cudaStream_t stream) = 0;

    template<class T>
    void AllGather(const T* sendbuff, T* recvbuff, size_t sendcount, int group, cudaStream_t stream)
    {
        return AllGather(sendbuff, recvbuff, sendcount, getTensorType<T>(), group, stream);
    }

    virtual void AllGather(
        const void* sendbuff, void* recvbuff, size_t sendcount, DataType type, int group, cudaStream_t stream) = 0;

    template<class T>
    void ReduceScatter(const T* sendbuff, T* recvbuff, size_t recvcount, cudaStream_t stream)
    {
        return ReduceScatter(sendbuff, recvbuff, recvcount, getTensorType<T>(), stream);
    }

    virtual void
    ReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, DataType type, cudaStream_t stream)
    {
        throw std::runtime_error("not implemented");
    }

    virtual void AllreduceResidualBiasRMSnorm(void*        hidden,
                                              void*        residual,
                                              const void*  bias,
                                              const void*  weights,
                                              float        eps,
                                              int          dim,
                                              int          token_num,
                                              DataType     dtype,
                                              cudaStream_t stream)
    {
        throw std::runtime_error("not implemented");
    }

    virtual void AllreduceResidualBiasRMSnormEx(void*        hidden,  // offset by caller
                                                void*        residual,
                                                const void*  bias,
                                                const void*  weights,
                                                float        eps,
                                                int          dim,
                                                DataType     type,
                                                int          group0,
                                                int          group1,
                                                const int*   local_token_nums,
                                                cudaStream_t stream)
    {
        throw std::runtime_error("not implemented");
    }

    template<class T>
    void AllreduceResidualBiasRMSnorm(
        T* hidden, T* residual, const T* bias, const T* weights, float eps, int dim, int token_num, cudaStream_t stream)
    {
        AllreduceResidualBiasRMSnorm(hidden, residual, bias, weights, eps, dim, token_num, getTensorType<T>(), stream);
    }

    virtual void AllGather2D(const void*  sendbuff,
                             void*        recvbuff,
                             size_t       pitch,
                             size_t       stride,
                             int          width,
                             int          height,
                             DataType     type,
                             int2         flags,  // (is_first, is_last)
                             cudaStream_t stream)
    {
        throw std::runtime_error("not implemented");
    }

    template<class T>
    void AllGather2D(const T*     sendbuff,
                     T*           recvbuff,
                     size_t       pitch,
                     size_t       stride,
                     int          width,
                     int          height,
                     int2         flags,
                     cudaStream_t stream)
    {
        AllGather2D(sendbuff, recvbuff, pitch, stride, width, height, getTensorType<T>(), flags, stream);
    }

    virtual void
    AllGatherAsym(const void* sendbuff, void* recvbuff, const size_t* sendcount, DataType type, cudaStream_t stream)
    {
        throw std::runtime_error("not implemented");
    }

    virtual void
    ReduceScatterAsym(const void* sendbuff, void* recvbuff, const size_t* recvcount, DataType type, cudaStream_t stream)
    {
        throw std::runtime_error("not implemented");
    }

protected:
    int world_size_;
    int rank_;
};

std::unique_ptr<Comm>
CreateCommunicator(const std::string& backend, int rank, int n_ranks, std::shared_ptr<HostComm> host_comm);

struct Splits {
    std::unique_ptr<Comm> tp;
    int                   attn_tp_group;

    std::shared_ptr<HostComm> h_comm;
    int                       h_comm_tp_group;
    int                       h_comm_dp_group;
};

}  // namespace turbomind::comm
