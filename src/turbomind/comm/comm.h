#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "src/turbomind/utils/Tensor.h"

namespace turbomind {

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

    template<class T>
    void AllReduceSum(const T* sendbuff, T* recvbuff, size_t count, cudaStream_t stream)
    {
        return AllReduceSum(sendbuff, recvbuff, count, getTensorType<T>(), stream);
    }

    virtual void
    AllReduceSum(const void* sendbuff, void* recvbuff, size_t count, DataType type, cudaStream_t stream) = 0;

    template<class T>
    void AllGather(const T* sendbuff, T* recvbuff, size_t sendcount, cudaStream_t stream)
    {
        return AllGather(sendbuff, recvbuff, sendcount, getTensorType<T>(), stream);
    }

    virtual void
    AllGather(const void* sendbuff, void* recvbuff, size_t sendcount, DataType type, cudaStream_t stream) = 0;

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

protected:
    int world_size_;
    int rank_;
};

std::vector<std::unique_ptr<Comm>> CreateNcclComm(const std::vector<int>& devices);

}  // namespace turbomind