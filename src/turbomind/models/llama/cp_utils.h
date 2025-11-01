// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/kernels/attention/attention_params.h"

namespace turbomind {

struct CpPostContext {

    CpPostContext(comm::DeviceCommImpl* d_comm, int attn_cp_group): d_comm(d_comm), attn_cp_group(attn_cp_group) {}

    comm::DeviceCommImpl* d_comm;
    int                   attn_cp_group;

    float*   cp_ML;
    void*    attn_param;
    DataType attn_type;
};

void CpPost(void* context, int split_cnt);

}  // namespace turbomind
