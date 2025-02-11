#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "src/turbomind/utils/Tensor.h"

namespace turbomind {

class Barrier {
public:
    explicit Barrier(int count): threshold_{count}, count_{count} {}

    void arrive_and_wait()
    {
        std::unique_lock lock{mutex_};
        auto             phase = phase_;
        if (--count_ == 0) {
            ++phase_;
            count_ = threshold_;
            cv_.notify_all();
        }
        else {
            cv_.wait(lock, [this, phase] { return phase_ != phase; });
        }
    }

private:
    std::mutex              mutex_;
    std::condition_variable cv_;

    int threshold_;
    int count_;

    uint32_t phase_{};
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

    virtual void RegisterBuffer(void* ptr, size_t size) {}

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

    template<class T>
    void AllreduceResidualBiasRMSnorm(
        T* hidden, T* residual, const T* bias, const T* weights, float eps, int dim, int token_num, cudaStream_t stream)
    {
        AllreduceResidualBiasRMSnorm(hidden, residual, bias, weights, eps, dim, token_num, getTensorType<T>(), stream);
    }

protected:
    int world_size_;
    int rank_;
};

std::vector<std::unique_ptr<Comm>> CreateNcclComm(const std::vector<int>& devices);

std::vector<std::unique_ptr<Comm>> CreateCustomComm(const std::vector<int>& devices);

}  // namespace turbomind