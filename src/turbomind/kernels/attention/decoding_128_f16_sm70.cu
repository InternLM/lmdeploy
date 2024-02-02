// Copyright (c) OpenMMLab. All rights reserved.

#include "decoding_config.h"
#include "decoding_template.h"

namespace turbomind {

// using Kernel = typename attention::DecodingConfig<half, half, std::integral_constant<int, 128>, 128>::Kernel;
// using Kernel = typename attention::DecodingConfig<arch::Sm70, half, half, int, 128>::Kernel;

// template void invokeDecoding<Kernel>(const typename Kernel::ParamType& params);

}  // namespace turbomind