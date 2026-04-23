// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>

#include <stdexcept>

#include <cuda_runtime.h>

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/tensor.h"

namespace turbomind::comm {

struct EpConfig {
    int num_nodes;
    int num_experts;
    int hidden;
    int ll_max_tokens_per_rank;
};

enum class EpMode
{
    kNull,
    kHighThroughput,
    kLowLatency,
};

struct EpDispatchInput {
    EpMode&                 mode;
    core::Tensor&           x;
    core::Tensor_<float>&   topk_weights;
    core::Tensor_<int64_t>& topk_idx;
    int                     num_worst_tokens;
    bool                    use_fp8;
    bool                    output_scales;
    bool                    zero_copy{false};
};

struct EpDispatchOutput {
    core::Tensor        out_x;
    core::Tensor        out_x_scales;
    core::Tensor        out_topk_weights;
    core::Buffer_<int>& f2n;
    core::Buffer_<int>& f2E;
    core::Buffer_<int>& en2f;
    core::Buffer_<int>& offsets;

    std::vector<core::Tensor> handle;

    int out_expert_token_num;

    core::Tensor rdma;  // used for low-latency
};

struct EpCombineInput {
    EpMode&                     mode;
    core::Tensor&               x;
    std::vector<core::Tensor>&  handle;
    std::optional<core::Tensor> topk_weights;
    std::optional<core::Tensor> topk_idx;
    bool                        zero_copy{false};
};

struct EpCombineOutput {
    core::Tensor out_x;
};

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

    virtual void ReduceScatterV(const void*   sendbuff,  //
                                void*         recvbuff,
                                const size_t* counts,
                                DataType      type,
                                int           group,
                                cudaStream_t  stream)
    {
        throw std::runtime_error("not implemented");
    }

    virtual void AllGatherV(const void*   sendbuff,  //
                            void*         recvbuff,
                            const size_t* counts,
                            DataType      type,
                            int           group,
                            cudaStream_t  stream)
    {
        throw std::runtime_error("not implemented");
    }

    virtual void InitializeEp(const EpConfig& config)
    {
        throw std::runtime_error("ep not implemented");
    }

    virtual void Dispatch(const EpDispatchInput& input, EpDispatchOutput& output, int group)
    {
        throw std::runtime_error("not implemented");
    }

    virtual void Combine(const EpCombineInput& input, EpCombineOutput& output, int group)
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
