// Copyright (c) OpenMMLab. All rights reserved.

#include <math_constants.h>

#include "src/turbomind/kernels/attention/cp_utils.h"

namespace turbomind {

void CpPost(void* context)
{
    auto ctx = reinterpret_cast<CpPostContext*>(context);

    ctx->d_comm->AllGather(ctx->partial_ML + ctx->cp_rank * ctx->count,  //
                           ctx->partial_ML,
                           ctx->count,
                           DataType::kFloat,
                           ctx->attn_cp_group,
                           ctx->stream);
}

__global__ void FillNegInfMLKernel(float4* data, size_t n_quads)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_quads) {
        data[idx] = make_float4(-CUDART_INF_F, 0.f, -CUDART_INF_F, 0.f);
    }
}

void invokeFillNegInfML(float* data, size_t n_pairs, cudaStream_t stream)
{
    if (n_pairs == 0) {
        return;
    }
    constexpr int block   = 256;
    const size_t  n_quads = n_pairs >> 1;
    const size_t  grid    = (n_quads + block - 1) / block;
    FillNegInfMLKernel<<<grid, block, 0, stream>>>(reinterpret_cast<float4*>(data), n_quads);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
