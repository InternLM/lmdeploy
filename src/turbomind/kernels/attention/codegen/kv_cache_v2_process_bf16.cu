// Copyright (c) OpenMMLab. All rights reserved.

#include "../kv_cache_utils_v2_impl.cuh"

namespace turbomind {

#if ENABLE_BF16
template void invokeProcessKV_v2(char**                 blocks,
                                 const nv_bfloat16*     k,
                                 const nv_bfloat16*     v,
                                 const nv_bfloat16*     k_bias,
                                 const nv_bfloat16*     v_bias,
                                 const int*             cu_q_len,
                                 const int*             cu_k_len,
                                 const int*             cu_block_num,
                                 const RopeKernelParam& rope_param,
                                 int64_t                stride_b,
                                 int64_t                stride_c,
                                 int64_t                stride_h,
                                 int64_t                stride_s,
                                 int                    block_seq_len,
                                 int                    layer_id,
                                 int                    cp_rank,
                                 cutlass::FastDivmod    cp_size,
                                 int                    max_q_len,
                                 int                    head_num,
                                 int                    head_dim,
                                 int                    batch_size,
                                 int                    quant_policy,
                                 cudaStream_t           stream);
#endif

}  // namespace turbomind
