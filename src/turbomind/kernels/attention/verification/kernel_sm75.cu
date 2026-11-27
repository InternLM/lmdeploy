// Copyright (c) OpenMMLab. All rights reserved.
//
// SM75 (Turing) verification-attention entry points. Instantiated ONLY for
// compute_75 / sm_75; no other architecture includes this translation unit's
// device code. Shares the generic mainloop in kernel_sm80.cuh and the
// arch-neutral reduce in reduce.cuh.
#include "src/turbomind/kernels/attention/verification/kernel_sm80.cuh"
#include "src/turbomind/kernels/attention/verification/policy_sm75.cuh"
#include "src/turbomind/kernels/attention/verification/reduce.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace turbomind::verification_attention::sm75 {

namespace {

using Sm75Policy128 = Sm75Policy<128, 64>;
using Sm75Policy256 = Sm75Policy<256, 64>;

template<class T, class Policy, class Tkv>
void Launch(const Arguments& arguments)
{
    const int     m_count  = arguments.max_query_length * arguments.query_group_size;
    const int     m_slices = (m_count + Policy::MTile - 1) / Policy::MTile;
    const dim3    grid(arguments.request_count, arguments.kv_head_count * m_slices, arguments.split_count);
    constexpr int smem_bytes = sizeof(SharedStorage<T, Policy>);
    if (arguments.split_count == 1) {
        auto kernel = VerificationAttentionKernel<T, Policy, false, Tkv>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        kernel<<<grid, Policy::Threads, smem_bytes, arguments.stream>>>(arguments);
    }
    else {
        auto kernel = VerificationAttentionKernel<T, Policy, true, Tkv>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        kernel<<<grid, Policy::Threads, smem_bytes, arguments.stream>>>(arguments);
    }
}

template<class T, int HeadDim, class Tkv>
void DispatchM(const Arguments& arguments)
{
    if (CtaM(arguments) == 64) {
        Launch<T, Sm75Policy<HeadDim, 64>, Tkv>(arguments);
    }
    else {
        Launch<T, Sm75Policy<HeadDim, 32>, Tkv>(arguments);
    }
}

template<class T, class Tkv>
void DispatchHeadDimension(const Arguments& arguments)
{
    if (arguments.head_dim == 128) {
        DispatchM<T, 128, Tkv>(arguments);
    }
    else {
        DispatchM<T, 256, Tkv>(arguments);
    }
}

}  // namespace

bool supports(const Capability& c)
{
    const bool dtype_supported     = c.data_type == DataType::kHalf;  // SM75 has no bf16 tensor cores
    const bool dimension_supported = c.head_dim == 128 || c.head_dim == 256;
    const int  required_smem       = c.head_dim == 128 ? int(sizeof(SharedStorage<cutlass::half_t, Sm75Policy128>)) :
                                                         int(sizeof(SharedStorage<cutlass::half_t, Sm75Policy256>));
    return c.arch >= 75 && c.arch < 80 && dtype_supported && dimension_supported && c.max_query_length <= 16
           && (c.quant_policy == 0 || c.quant_policy == 4) && c.cp_size == 1 && !c.is_mla && !c.has_attention_sinks
           && c.max_dynamic_smem_bytes >= required_smem;
}

int choose_split_count(int query_count,
                       int base_cta_count,
                       int max_key_length,
                       int key_tile,
                       int partial_capacity,
                       int requested_max_splits,
                       int sm_count)
{
    const int key_tiles        = (max_key_length + key_tile - 1) / key_tile;
    const int capacity_limit   = std::max(1, partial_capacity / query_count);
    const int useful_limit     = std::min(key_tiles, requested_max_splits);
    const int occupancy_target = std::max(1, sm_count * 2);
    const int occupancy_splits = (occupancy_target + base_cta_count - 1) / base_cta_count;
    return std::min(128, std::min(capacity_limit, std::min(useful_limit, std::max(1, occupancy_splits))));
}

void run(const Arguments& arguments)
{
    // fp16 only (SM75); supports() rejects bf16. quant_policy 4 = int4 KV.
    if (arguments.quant_policy == 4) {
        DispatchHeadDimension<cutlass::half_t, uint4_t>(arguments);
    }
    else {
        DispatchHeadDimension<cutlass::half_t, cutlass::half_t>(arguments);
    }
    if (arguments.split_count > 1) {
        Reduce(arguments);
    }
}

}  // namespace turbomind::verification_attention::sm75
