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

// Fill an array of (M, L) pairs with (-inf, 0). Used to initialize this rank's
// slot in `partial_ML` before attention, so that reduce treats slots left
// untouched by early-exiting CTAs (e.g. finished sequences in async mode) as
// no-contribution rather than reading stale data from previous batches.
void invokeFillNegInfML(float* data, size_t n_pairs, cudaStream_t stream);

}  // namespace turbomind
