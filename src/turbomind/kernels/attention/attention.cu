// Copyright (c) OpenMMLab. All rights reserved.

#include "attention.h"
#include "attention_config.h"

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
            // using Config = AttentionConfig<arch::Sm80, T, T, std::integral_constant<int, 128>, 128>;
            using Config = AttentionConfig<arch::Sm80, T, T, int, 128>;
            invokeAttention<typename Config::Kernel>(params);
        }
    }
}

template void dispatchAttention(const AttentionParams<half>& params);

}  // namespace turbomind