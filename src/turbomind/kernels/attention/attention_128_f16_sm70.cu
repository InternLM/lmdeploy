// Copyright (c) OpenMMLab. All rights reserved.

#include "attention_config.h"
#include "attention_template.h"

namespace turbomind {

using Kernel = typename attention::AttentionConfig<arch::Sm70, half, half, 1, 128>::Kernel;

template void invokeAttention<Kernel>(const typename Kernel::ParamType& params);

}  // namespace turbomind
