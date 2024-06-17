// Copyright (c) OpenMMLab. All rights reserved.

#include "../attention_config.h"
#include "../attention_template.h"

namespace turbomind {

using namespace attention;

template void invokeAttention<typename AttentionConfig<arch::Sm80, half, 128, CacheType::kLinear>::Kernel>(
    const AttentionParams<half>& params);

template void invokeAttention<typename AttentionConfig<arch::Sm80, half, 128, CacheType::kBlock>::Kernel>(
    const AttentionParams<half>& params);

}  // namespace turbomind
