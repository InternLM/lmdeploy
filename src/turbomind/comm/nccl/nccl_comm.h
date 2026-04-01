// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <nccl.h>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/host_comm.h"

namespace deep_ep {
class Buffer;
}

namespace turbomind::comm {

class NcclCommImpl: public DeviceCommImpl {
public:
    NcclCommImpl(ncclComm_t comm, int n_ranks, int rank, HostComm h_comm);
    ~NcclCommImpl();

    int rank(int group) const override;
    int n_ranks(int group) const override;

    void* Allocate(size_t size) override;
    void  Free(void* ptr) override;

    void Register(void* ptr, size_t size) override;
    void Deregister(void* ptr) override;

    int Split(int color, int key, int group) override;
    int Query(QueryAttr attr) const noexcept override;

    void AllReduceSum(
        const void* sendbuff, void* recvbuff, size_t count, DataType type, int group, cudaStream_t stream) override;

    void AllGather(
        const void* sendbuff, void* recvbuff, size_t sendcount, DataType type, int group, cudaStream_t stream) override;

    void ReduceScatter(
        const void* sendbuff, void* recvbuff, size_t recvcount, DataType type, int group, cudaStream_t stream) override;

    void ReduceScatterV(const void*   sendbuff,
                        void*         recvbuff,
                        const size_t* counts,
                        DataType      type,
                        int           group,
                        cudaStream_t  stream) override;

    void AllGatherV(const void*   sendbuff,
                    void*         recvbuff,
                    const size_t* counts,
                    DataType      type,
                    int           group,
                    cudaStream_t  stream) override;

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

    void Broadcast(const void*  sendbuff,
                   void*        recvbuff,
                   size_t       count,
                   DataType     type,
                   int          root,
                   int          group,
                   cudaStream_t stream) override;

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
    void InitializeEp(const EpConfig& config) override;
    void Dispatch(const EpDispatchInput& input, EpDispatchOutput& output, int group) override;
    void Combine(const EpCombineInput& input, EpCombineOutput& output, int group) override;
#endif

private:
    void Register(int group, void* buff, size_t size);
    void Deregister(int group, void* buff);

    HostComm h_comm_;

    int global_n_ranks_;
    int global_rank_;

    std::vector<ncclComm_t> groups_;

    std::vector<std::unordered_map<void*, std::pair<void*, size_t>>> handles_;

    std::unordered_map<void*, size_t> buffers_;

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
    std::unique_ptr<deep_ep::Buffer> buffer_;
#endif
    EpConfig ep_config_;
};

DeviceComm CreateNcclCommunicator(int n_ranks, int rank, HostComm h_comm);

}  // namespace turbomind::comm
