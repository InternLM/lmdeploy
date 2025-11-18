// Copyright (c) OpenMMLab. All rights reserved.

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
    sync_check_cuda_error();
}

}  // namespace turbomind
