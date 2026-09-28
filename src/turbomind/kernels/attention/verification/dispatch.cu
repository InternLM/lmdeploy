// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/verification/kernel_sm80.cuh"
#include "src/turbomind/kernels/attention/verification/kernel_sm90_wgmma.cuh"
#include "src/turbomind/kernels/attention/verification/kernel_sm90_wgmma_rs.cuh"
#include "src/turbomind/kernels/attention/verification/kernel_sm90_wgmma_rs_ws.cuh"

#include <algorithm>

namespace turbomind::verification_attention {

bool supports(const Capability& c)
{
    const bool dtype_supported =
        c.data_type == DataType::kHalf || c.data_type == DataType::kBfloat16;
    const bool dimension_supported = c.head_dim == 128 || c.head_dim == 256;
    const int required_dynamic_smem_bytes = c.head_dim == 128 ? 112 * 1024 : 104 * 1024;
    return c.arch == 90 && dtype_supported && dimension_supported
           && c.max_query_length <= 16 && c.quant_policy == 0 && c.cp_size == 1
           && !c.is_mla && !c.has_attention_sinks
           && c.max_dynamic_smem_bytes >= required_dynamic_smem_bytes;
}

int choose_split_count(int query_count,
                       int base_cta_count,
                       int max_key_length,
                       int key_tile,
                       int partial_capacity,
                       int requested_max_splits,
                       int sm_count)
{
    const int key_tiles = (max_key_length + key_tile - 1) / key_tile;
    const int capacity_limit = std::max(1, partial_capacity / query_count);
    const int useful_limit = std::min(key_tiles, requested_max_splits);
    const int occupancy_target = std::max(1, sm_count * 2);
    const int occupancy_splits =
        (occupancy_target + base_cta_count - 1) / base_cta_count;
    return std::min(128,
                    std::min(capacity_limit,
                             std::min(useful_limit, std::max(1, occupancy_splits))));
}

template<class T, class Policy>
void Launch(const Arguments& arguments)
{
    const int m_count = arguments.max_query_length * arguments.query_group_size;
    const int m_slices = (m_count + Policy::MTile - 1) / Policy::MTile;
    const dim3 grid(arguments.request_count,
                    arguments.kv_head_count * m_slices,
                    arguments.split_count);
    constexpr int smem_bytes = sizeof(SharedStorage<T, Policy>);
    if (arguments.split_count == 1) {
        auto kernel = VerificationAttentionKernel<T, Policy, false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        kernel<<<grid, Policy::Threads, smem_bytes, arguments.stream>>>(arguments);
    }
    else {
        auto kernel = VerificationAttentionKernel<T, Policy, true>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        kernel<<<grid, Policy::Threads, smem_bytes, arguments.stream>>>(arguments);
    }
}

template<class T, int HeadDim>
void DispatchM(const Arguments& arguments)
{
    if (CtaM(arguments) == 64) {
        Launch<T, Sm90Policy<HeadDim, 64>>(arguments);
    }
    else {
        Launch<T, Sm90Policy<HeadDim, 128>>(arguments);
    }
}

template<class T>
void DispatchHeadDimension(const Arguments& arguments)
{
    if (arguments.head_dim == 128) {
        DispatchM<T, 128>(arguments);
    }
    else {
        DispatchM<T, 256>(arguments);
    }
}

void run(const Arguments& arguments)
{
    if (UsesM128(arguments)) {
        LaunchWgmmaRsWs<cutlass::bfloat16_t>(arguments);
    }
    else if (UsesM64(arguments)) {
        LaunchWgmmaRs<cutlass::bfloat16_t>(arguments);
    }
    else if (UsesM32(arguments)) {
        LaunchWgmma<cutlass::bfloat16_t>(arguments);
    }
    else if (arguments.data_type == DataType::kHalf) {
        DispatchHeadDimension<cutlass::half_t>(arguments);
    }
    else {
        DispatchHeadDimension<cutlass::bfloat16_t>(arguments);
    }
    if (arguments.split_count > 1) {
        Reduce(arguments);
    }
}

}  // namespace turbomind::verification_attention
