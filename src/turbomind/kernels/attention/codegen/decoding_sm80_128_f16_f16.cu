// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template void invokeDecoding<Decoding<arch::Sm80, half, half, 1, 128>>(const AttentionParams<half>& params);

// using sm80_f16_f16_g2_d128 = Decoding<arch::Sm80, half, half, 2, 128>;
// template void invokeDecoding<sm80_f16_f16_g2_d128>(const typename sm80_f16_f16_g2_d128::ParamType& params);

// using sm80_f16_f16_g4_d128 = Decoding<arch::Sm80, half, half, 4, 128>;
// template void invokeDecoding<sm80_f16_f16_g4_d128>(const typename sm80_f16_f16_g4_d128::ParamType& params);

template void invokeDecoding<Decoding<arch::Sm80, half, half, 8, 128>>(const AttentionParams<half>& params);

}  // namespace turbomind
