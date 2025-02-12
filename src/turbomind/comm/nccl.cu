
#include <cstdint>
#include <memory>

#include <nccl.h>
#include <stdexcept>

#include "src/turbomind/comm/comm.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/string_utils.h"

#include "src/turbomind/kernels/norm/rms_norm.h"

#define NCCLCHECK(e)                                                                                                   \
    if (auto ec = e; ec != ncclSuccess) {                                                                              \
        auto msg = fmtstr("NCCL error %s:%d '%s'", __FILE__, __LINE__, ncclGetErrorString(ec));                        \
        throw std::runtime_error(msg.c_str());                                                                         \
    }

namespace turbomind {

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
        ncclCommDestroy(comm_);
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

    void RegisterBuffer(void* ptr, size_t size) override
    {
        // make no difference
        // void* handle{};
        // ncclCommRegister(comm_, ptr, size, &handle);
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
};

std::vector<std::unique_ptr<Comm>> CreateNcclComm(const std::vector<int>& devices)
{
    ncclUniqueId uid{};
    NCCLCHECK(ncclGetUniqueId(&uid));

    std::vector<std::unique_ptr<Comm>> ret(devices.size());

    int old_device{};
    // Note this will create ctx on dev 0 if CUDA is never called before
    cudaGetDevice(&old_device);

    // initialize the communicator clique
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < (int)ret.size(); ++i) {
        cudaSetDevice(devices[i]);
        ncclComm_t comm{};
        NCCLCHECK(ncclCommInitRank(&comm, ret.size(), uid, i));
        ret[i] = std::unique_ptr<Comm>{new NcclComm{comm, (int)ret.size(), i}};
    }
    NCCLCHECK(ncclGroupEnd());

    cudaSetDevice(old_device);

    return ret;
}

}  // namespace turbomind