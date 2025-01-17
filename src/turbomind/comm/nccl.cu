
#include <memory>

#include <nccl.h>
#include <stdexcept>

#include "src/turbomind/comm/comm.h"

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
        ncclGroupStart();
        ncclAllReduce(sendbuff, recvbuff, count, getNcclDataType(type), ncclSum, comm_, stream);
        ncclGroupEnd();

        // ncclGroupStart();
        // {
        //     const size_t recvcount = count / world_size();
        //     auto         sendbuff  = data;
        //     auto         recvbuff  = (char*)sendbuff + 2 * rank() * recvcount;
        //     ncclReduceScatter(sendbuff, recvbuff, recvcount, getNcclDataType(type), ncclSum, comm_, stream);
        //     ncclAllGather(recvbuff, sendbuff, recvcount, getNcclDataType(type), comm_, stream);
        // }
        // ncclGroupEnd();
    }

    void AllGather(const void* sendbuff, void* recvbuff, size_t sendcount, DataType type, cudaStream_t stream) override
    {
        ncclGroupStart();
        ncclAllGather(sendbuff, recvbuff, sendcount, getNcclDataType(type), comm_, stream);
        ncclGroupEnd();
    }

    void
    ReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, DataType type, cudaStream_t stream) override
    {
        ncclGroupStart();
        ncclReduceScatter(sendbuff, recvbuff, recvcount, getNcclDataType(type), ncclSum, comm_, stream);
        ncclGroupEnd();
    }

private:
    ncclComm_t comm_;
};

std::vector<std::unique_ptr<Comm>> CreateNcclComm(const std::vector<int>& devices)
{
    ncclUniqueId uid{};
    ncclGetUniqueId(&uid);

    std::vector<std::unique_ptr<Comm>> ret(devices.size());

    int old_device{};
    // Note this will create ctx on dev 0 if CUDA is never called before
    cudaGetDevice(&old_device);

    // initialize the communicator clique
    ncclGroupStart();
    for (int i = 0; i < (int)ret.size(); ++i) {
        cudaSetDevice(devices[i]);
        ncclComm_t comm{};
        ncclCommInitRank(&comm, ret.size(), uid, i);
        ret[i] = std::unique_ptr<Comm>{new NcclComm{comm, (int)ret.size(), i}};
    }
    ncclGroupEnd();

    cudaSetDevice(old_device);

    return ret;
}

}  // namespace turbomind