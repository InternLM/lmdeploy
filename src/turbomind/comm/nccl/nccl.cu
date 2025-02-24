// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>
#include <memory>
#include <unordered_map>

#include <nccl.h>

#include "src/turbomind/comm/comm.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/string_utils.h"

#include "src/turbomind/kernels/norm/rms_norm.h"

#define NCCLCHECK(e)                                                                                                   \
    if (auto ec = e; ec != ncclSuccess) {                                                                              \
        auto msg = fmtstr("NCCL error %s:%d '%s'", __FILE__, __LINE__, ncclGetErrorString(ec));                        \
        throw std::runtime_error(msg.c_str());                                                                         \
    }

namespace turbomind::comm {

static inline ncclDataType_t getNcclDataType(DataType type)
{
    switch (type) {
        case DataType::TYPE_FP32:
            return ncclFloat;
        case DataType::TYPE_FP16:
            return ncclHalf;
        case DataType::TYPE_BF16:
            return ncclBfloat16;
        case DataType::TYPE_UINT8:
            return ncclUint8;
        default:
            throw std::runtime_error("not supported");
    }
}

class NcclComm: public Comm {
public:
    NcclComm(ncclComm_t comm, int world_size, int rank): Comm(world_size, rank), comm_{comm} {}

    ~NcclComm()
    {
        for (const auto& [ptr, _] : handles_) {
            TM_LOG_WARNING("[TM][NCCL][%d] Buffer %p is not deregistered", rank_, ptr);
        }

        for (const auto& [ptr, size] : buffers_) {
            TM_LOG_WARNING("[TM][NCCL][%d] Allocation (%p, %lu) is not freed", rank_, ptr, size);
        }

        if (auto ec = ncclCommFinalize(comm_); ec != ncclSuccess) {
            TM_LOG_ERROR("[TM][NCCL][%d] Failed to finalize communicator: %s", rank_, ncclGetErrorString(ec));
        }

        if (auto ec = ncclCommDestroy(comm_); ec != ncclSuccess) {
            TM_LOG_ERROR("[TM][NCCL][%d] Failed to destroy communicator: %s", rank_, ncclGetErrorString(ec));
        }
    }

    void* Allocate(size_t size) override
    {
        void* ptr{};
        NCCLCHECK(ncclMemAlloc(&ptr, size));
        buffers_.emplace(ptr, size);
        return ptr;
    }

    void Free(void* ptr) override
    {
        if (auto it = buffers_.find(ptr); it != buffers_.end()) {
            NCCLCHECK(ncclMemFree(ptr));
            buffers_.erase(ptr);
        }
        else {
            TM_LOG_WARNING("[TM][NCCL][%d] Freeing %p which is not allocated by NcclComm", rank_, ptr);
        }
    }

    void Register(void* ptr, size_t size) override
    {
        if (!handles_.count(ptr)) {
            void* handle{};
            NCCLCHECK(ncclCommRegister(comm_, ptr, size, &handle));
            handles_.emplace(ptr, handle);
        }
        else {
            TM_LOG_WARNING("[TM][NCCL][%d] Duplicated registration on (%p, %lu)", rank_, ptr, size);
        }
    }

    void Deregister(void* ptr) override
    {
        if (auto it = handles_.find(ptr); it != handles_.end()) {
            NCCLCHECK(ncclCommDeregister(comm_, it->second));
            handles_.erase(it);
        }
        else {
            TM_LOG_WARNING("[TM][NCCL][%d] Deregistering non-registered address %p", rank_, ptr);
        }
    }

    int Query(QueryAttr attr) const noexcept override
    {
        return 0;
    }

    void AllReduceSum(const void* sendbuff, void* recvbuff, size_t count, DataType type, cudaStream_t stream) override
    {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, getNcclDataType(type), ncclSum, comm_, stream));
        NCCLCHECK(ncclGroupEnd());
    }

    void AllGather(const void* sendbuff, void* recvbuff, size_t sendcount, DataType type, cudaStream_t stream) override
    {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclAllGather(sendbuff, recvbuff, sendcount, getNcclDataType(type), comm_, stream));
        NCCLCHECK(ncclGroupEnd());
    }

    void
    ReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, DataType type, cudaStream_t stream) override
    {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclReduceScatter(sendbuff, recvbuff, recvcount, getNcclDataType(type), ncclSum, comm_, stream));
        NCCLCHECK(ncclGroupEnd());
    }

    void AllreduceResidualBiasRMSnorm(void*        hidden,
                                      void*        residual,
                                      const void*  bias,
                                      const void*  weights,
                                      float        eps,
                                      int          dim,
                                      int          token_num,
                                      DataType     dtype,
                                      cudaStream_t stream) override
    {
        const auto elem_size = get_elem_size(dtype);

        auto rms_norm = [&](int64_t first, int64_t count) {
            invokeResidualBiasRMSNorm((char*)hidden + elem_size * first * dim,
                                      (char*)residual + elem_size * first * dim,
                                      weights,
                                      bias,
                                      dtype,
                                      dim,
                                      count,
                                      eps,
                                      stream);
        };

        if (1) {
            AllReduceSum(hidden, hidden, token_num * dim, dtype, stream);
            rms_norm(0, token_num);
        }
        else {  // Only useful for large input size
            const int    slice     = (token_num + world_size_ - 1) / world_size_;
            const size_t recvcount = slice * dim;
            auto         sendbuff  = hidden;
            auto         recvbuff  = (char*)hidden + elem_size * rank() * recvcount;
            ReduceScatter(sendbuff, recvbuff, recvcount, dtype, stream);
            rms_norm(rank_ * slice, slice);
            AllGather(recvbuff, sendbuff, recvcount, dtype, stream);
        }
    }

private:
    ncclComm_t comm_;

    std::unordered_map<void*, void*>  handles_;
    std::unordered_map<void*, size_t> buffers_;
};

class NcclGroupId: public GroupId {
public:
    void Initialize() override
    {
        NCCLCHECK(ncclGetUniqueId(&uid_));
    }

    void Export(std::ostream& os) override
    {
        os.write((const char*)&uid_, sizeof(uid_));
    }

    void Import(std::istream& is) override
    {
        is.read((char*)&uid_, sizeof(uid_));
    }

    std::unique_ptr<Comm> CreateCommunicator(int rank, int world_size) override
    {
        ncclComm_t comm{};
        NCCLCHECK(ncclCommInitRank(&comm, world_size, uid_, rank));
        return std::make_unique<NcclComm>(comm, world_size, rank);
    }

private:
    ncclUniqueId uid_{};
};

std::unique_ptr<GroupId> CreateNcclGroupId()
{
    return std::make_unique<NcclGroupId>();
}

}  // namespace turbomind::comm
