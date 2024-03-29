// Copyright (c) OpenMMLab. All rights reserved.

#include "attention.h"
#include "attention_config.h"
#include "src/turbomind/kernels/attention/arch.h"

namespace turbomind {

template<class Kernel>
void invokeAttention(const typename Kernel::ParamType& params);

template<class T>
void dispatchAttention(const AttentionParams<T>& params)
{
    using namespace attention;
    if (params.size_per_head == 128) {
        if (0) {}
        else if (params.arch >= 80) {
            using Config = AttentionConfig<arch::Sm80, T, 128, CacheType::kLinear>;
            invokeAttention<typename Config::Kernel>(params);
        }
        // else if (params.arch == 75) {
        //     using Config = AttentionConfig<arch::Sm75, T, T, 1, 128>;
        //     invokeAttention<typename Config::Kernel>(params);
        // }
        // else if (params.arch == 70) {
        //     using Config = AttentionConfig<arch::Sm70, T, T, 1, 128>;
        //     invokeAttention<typename Config::Kernel>(params);
        // }
    }
}

#if ENABLE_BF16
template<>
void dispatchAttention(const AttentionParams<nv_bfloat16>& params)
{
    using namespace attention;
    if (params.size_per_head == 128) {
        if (0) {}
        // else if (params.arch >= 80) {
        //     using Config = AttentionConfig<arch::Sm80, nv_bfloat16, nv_bfloat16, 1, 128>;
        //     invokeAttention<typename Config::Kernel>(params);
        // }
    }
}
#endif

template void dispatchAttention(const AttentionParams<half>& params);
// #if ENABLE_BF16
// template void dispatchAttention(const AttentionParams<nv_bfloat16>& params);
// #endif

}  // namespace turbomind
