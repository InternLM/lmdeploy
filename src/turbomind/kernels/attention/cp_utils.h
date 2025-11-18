// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

struct CpPostContext {

    CpPostContext(comm::DeviceCommImpl* d_comm, int attn_cp_group): d_comm(d_comm), attn_cp_group(attn_cp_group) {}

    comm::DeviceCommImpl* d_comm;
    int                   attn_cp_group;

    int          cp_rank;
    int          count;
    float*       partial_ML;
    cudaStream_t stream;
};

void CpPost(void* context);

}  // namespace turbomind
