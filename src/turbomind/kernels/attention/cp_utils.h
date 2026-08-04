// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

struct CpPostContext {

    CpPostContext(comm::DeviceCommImpl* d_comm, int attn_cp_group, int cp_size);
    ~CpPostContext();

    CpPostContext(const CpPostContext&)            = delete;
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

// Fill an array of (M, L) pairs with (-inf, 0). Used to initialize this rank's
// slot in `partial_ML` before attention, so that reduce treats slots left
// untouched by early-exiting CTAs (e.g. finished sequences in async mode) as
// no-contribution rather than reading stale data from previous batches.
void invokeFillNegInfML(float* data, size_t n_pairs, cudaStream_t stream);

}  // namespace turbomind
