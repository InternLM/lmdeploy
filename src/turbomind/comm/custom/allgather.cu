
#include "src/turbomind/comm/custom/allgather.h"
#include "src/turbomind/comm/custom/custom_comm.h"

namespace turbomind {

__global__ void __launch_bounds__(1024, 1) local_allgather_kernel(SmChannels             channels,  //
                                                                  mscclpp::DeviceSyncer* device_syncer,
                                                                  int                    rank,
                                                                  int                    world_size,
                                                                  size_t                 size)
{
    local_allgather(channels, device_syncer, rank, world_size, size);
}

void CustomComm::AllGather(const void* sendbuff, void* recvbuff, size_t sendcount, DataType type, cudaStream_t stream)
{
    const size_t bytesize = elem_size(type) * sendcount;
    FT_CHECK((char*)sendbuff == (char*)recvbuff + bytesize * rank_);
    auto&      channels = registered_channels_.at(recvbuff);
    SmChannels chns{};
    for (size_t i = 0; i < channels.size(); ++i) {
        chns[i] = mscclpp::deviceHandle(channels[i]);
    }
    int threads = 1024;
    int blocks  = 48;
    local_allgather_kernel<<<blocks, threads, 0, stream>>>(chns,  //
                                                           device_syncer_,
                                                           rank_,
                                                           world_size_,
                                                           bytesize);
}

}  // namespace turbomind