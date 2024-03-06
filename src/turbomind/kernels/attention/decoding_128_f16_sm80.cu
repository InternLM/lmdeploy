// Copyright (c) OpenMMLab. All rights reserved.

#include "decoding_config.h"
#include "decoding_template.h"

namespace turbomind {

using namespace attention;

using sm80_f16_f16_g1_d128 = Decoding<arch::Sm80, half, half, 1, 128>;
template void invokeDecoding<sm80_f16_f16_g1_d128>(const typename sm80_f16_f16_g1_d128::ParamType& params);

using sm80_f16_f16_g4_d128 = Decoding<arch::Sm80, half, half, 4, 128>;
template void invokeDecoding<sm80_f16_f16_g4_d128>(const typename sm80_f16_f16_g4_d128::ParamType& params);

using sm80_f16_f16_g5_d128 = Decoding<arch::Sm80, half, half, 5, 128>;
template void invokeDecoding<sm80_f16_f16_g5_d128>(const typename sm80_f16_f16_g4_d128::ParamType& params);

using sm80_f16_f16_g6_d128 = Decoding<arch::Sm80, half, half, 6, 128>;
template void invokeDecoding<sm80_f16_f16_g6_d128>(const typename sm80_f16_f16_g4_d128::ParamType& params);

using sm80_f16_f16_g8_d128 = Decoding<arch::Sm80, half, half, 8, 128>;
template void invokeDecoding<sm80_f16_f16_g8_d128>(const typename sm80_f16_f16_g8_d128::ParamType& params);

using sm80_f16_s8_g1_d128 = Decoding<arch::Sm80, half, int8_t, 1, 128>;
template void invokeDecoding<sm80_f16_s8_g1_d128>(const typename sm80_f16_s8_g1_d128::ParamType& params);

using sm80_f16_s8_g2_d128 = Decoding<arch::Sm80, half, int8_t, 2, 128>;
template void invokeDecoding<sm80_f16_s8_g2_d128>(const typename sm80_f16_s8_g2_d128::ParamType& params);

}  // namespace turbomind
