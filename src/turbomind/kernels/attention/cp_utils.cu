// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/cp_utils.h"

namespace turbomind {

CpPostContext::CpPostContext(comm::DeviceCommImpl* d_comm, int attn_cp_group, int cp_size):
    d_comm(d_comm), attn_cp_group(attn_cp_group)
{
    if (cp_size > 1) {
        TM_CUDA_CHECK(cudaStreamCreateWithFlags(&cp_stream, cudaStreamNonBlocking));
        TM_CUDA_CHECK(cudaEventCreateWithFlags(&produce_event, cudaEventDisableTiming));
        TM_CUDA_CHECK(cudaEventCreateWithFlags(&consume_event, cudaEventDisableTiming));
    }
}

CpPostContext::~CpPostContext()
{
    if (cp_stream) {
        TM_CUDA_CHECK(cudaEventDestroy(produce_event));
        TM_CUDA_CHECK(cudaEventDestroy(consume_event));
        TM_CUDA_CHECK(cudaStreamDestroy(cp_stream));
    }
}

void CpPost(void* context)
{
    auto ctx = reinterpret_cast<CpPostContext*>(context);

    // Only reachable when cp_size > 1 (`cp_fn` is unset otherwise), in which case
    // the constructor has created the stream and events.
    TM_CHECK(ctx->cp_stream != nullptr);

    // Wait for the attention kernel that produced this rank's partials, then run
    // the gather on the dedicated stream so gathers from overlapping prefill and
    // decode passes stay serialized in host call order on every rank.
    TM_CUDA_CHECK(cudaEventRecord(ctx->produce_event, ctx->stream));
    TM_CUDA_CHECK(cudaStreamWaitEvent(ctx->cp_stream, ctx->produce_event, 0));

    ctx->d_comm->AllGather(ctx->partial_ML + ctx->cp_rank * ctx->count,  //
                           ctx->partial_ML,
                           ctx->count,
                           DataType::kFloat,
                           ctx->attn_cp_group,
                           ctx->cp_stream);

    // Make the consuming reduce kernel on `ctx->stream` wait for the gather.
    TM_CUDA_CHECK(cudaEventRecord(ctx->consume_event, ctx->cp_stream));
    TM_CUDA_CHECK(cudaStreamWaitEvent(ctx->stream, ctx->consume_event, 0));
}

}  // namespace turbomind
