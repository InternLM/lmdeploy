// Copyright (c) OpenMMLab. All rights reserved.

#include "attention_config.h"
#include "attention_template.h"

namespace turbomind {

using Kernel_linear =
    typename attention::AttentionConfig<arch::Sm80, half, half, 1, 128, attention::CacheType::kLinear>::Kernel;
template void invokeAttention<Kernel_linear>(const typename Kernel_linear::ParamType& params);

using Kernel_block =
    typename attention::AttentionConfig<arch::Sm80, half, half, 1, 128, attention::CacheType::kBlock>::Kernel;
template void invokeAttention<Kernel_block>(const typename Kernel_block::ParamType& params);

}  // namespace turbomind
