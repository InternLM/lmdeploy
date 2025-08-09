// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>

#include <stdexcept>

#include <cuda_runtime.h>

#include "src/turbomind/comm/host_comm.h"

namespace turbomind::comm {

enum QueryAttr
{
    kHasAllGather2D
};

class DeviceCommImpl {
public:
    virtual ~DeviceCommImpl();

    virtual int n_ranks(int group) const = 0;

    virtual int rank(int group) const = 0;

    virtual void* Allocate(size_t size) = 0;

    virtual void Free(void* ptr) = 0;

    virtual void Register(void* ptr, size_t size) = 0;

    virtual void Deregister(void* ptr) = 0;

    virtual int Split(int color, int key, int group)
    {
        throw std::runtime_error("not implemented");
    }

    virtual int Query(QueryAttr attr) const noexcept = 0;

    virtual void AllReduceSum(const void*  sendbuff,  //
                              void*        recvbuff,
                              size_t       count,
                              DataType     type,
                              int          group,
                              cudaStream_t stream) = 0;

    virtual void AllGather(const void*  sendbuff,  //
                           void*        recvbuff,
                           size_t       sendcount,
                           DataType     type,
                           int          group,
                           cudaStream_t stream) = 0;

    virtual void ReduceScatter(const void*  sendbuff,  //
                               void*        recvbuff,
                               size_t       recvcount,
                               DataType     type,
                               int          group,
                               cudaStream_t stream)
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
                                              int          group,
                                              cudaStream_t stream)
    {
        throw std::runtime_error("not implemented");
    }

    virtual void AllreduceResidualBiasRMSnormEx(void*        hidden,
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

    virtual void AllGather2D(const void*  sendbuff,
                             void*        recvbuff,
                             size_t       pitch,
                             size_t       stride,
                             int          width,
                             int          height,
                             DataType     type,
                             int2         flags,  // (is_first, is_last)
                             int          group,
                             cudaStream_t stream)
    {
        throw std::runtime_error("not implemented");
    }

    virtual void Broadcast(const void*  sendbuff,  //
                           void*        recvbuff,
                           size_t       count,
                           DataType     type,
                           int          root,
                           int          group,
                           cudaStream_t stream)
    {
        throw std::runtime_error("not implemented");
    }
};

class DeviceComm {
public:
    DeviceComm() = default;

    /* implicit */ DeviceComm(std::unique_ptr<DeviceCommImpl> impl): impl_{std::move(impl)} {}

    DeviceCommImpl* operator->() const noexcept
    {
        return impl_.get();
    }

    operator DeviceCommImpl*() const noexcept
    {
        return impl_.get();
    }

private:
    std::unique_ptr<DeviceCommImpl> impl_;
};

DeviceComm CreateDeviceCommunicator(const std::string& backend,  //
                                    int                n_ranks,
                                    int                rank,
                                    HostComm           h_comm);

}  // namespace turbomind::comm
