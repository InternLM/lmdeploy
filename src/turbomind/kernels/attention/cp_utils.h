// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

struct CpPostContext {

    CpPostContext(comm::DeviceCommImpl* d_comm, int attn_cp_group, int cp_size);
    ~CpPostContext();

    CpPostContext(const CpPostContext&) = delete;
    CpPostContext& operator=(const CpPostContext&) = delete;

    comm::DeviceCommImpl* d_comm;
    int                   attn_cp_group;

    int          cp_rank;
    int          count;
    float*       partial_ML;
    cudaStream_t stream;

    // Dedicated stream serializing every AllGather of this CP group: concurrent
    // collectives on one communicator (e.g. prefill on an aux stream overlapping
    // decode on the main stream) are unordered and can pair inconsistently across
    // ranks. The events bracket each gather against the producing/consuming stream.
    // Only created when cp_size > 1.
    cudaStream_t cp_stream{};
    cudaEvent_t  produce_event{};
    cudaEvent_t  consume_event{};
};

void CpPost(void* context);

}  // namespace turbomind
