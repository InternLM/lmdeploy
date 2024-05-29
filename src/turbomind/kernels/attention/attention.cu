// Copyright (c) OpenMMLab. All rights reserved.

#include "attention.h"
#include "attention_config.h"
#include "src/turbomind/kernels/attention/arch.h"
#include "src/turbomind/models/llama/llama_utils.h"

namespace turbomind {

template<class Kernel>
void invokeAttention(const typename Kernel::ParamType& params);

template<class T>
void dispatchAttention(const AttentionParams<T>& params)
{
    using namespace attention;
    if (params.size_per_head == 128) {

        if (params.arch >= 80) {
            if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
                using Config = AttentionConfig<arch::Sm80, T, 128, CacheType::kBlock>;
                return invokeAttention<typename Config::Kernel>(params);
            }
            else {
                using Config = AttentionConfig<arch::Sm80, T, 128, CacheType::kLinear>;
                return invokeAttention<typename Config::Kernel>(params);
            }
        }

        if constexpr (!std::is_same_v<T, nv_bfloat16>) {
            if (params.arch == 75) {
                return invokeAttention<typename AttentionConfig<arch::Sm75, T, 128, CacheType::kLinear>::Kernel>(
                    params);
            }
            else if (params.arch >= 70) {
                return invokeAttention<typename AttentionConfig<arch::Sm70, T, 128, CacheType::kLinear>::Kernel>(
                    params);
            }
        }
    }
    FT_CHECK(0);
}

template void dispatchAttention(const AttentionParams<half>& params);
#if ENABLE_BF16
template void dispatchAttention(const AttentionParams<nv_bfloat16>& params);
#endif

}  // namespace turbomind
