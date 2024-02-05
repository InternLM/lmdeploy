// Copyright (c) OpenMMLab. All rights reserved.

#include "decoding_config.h"
#include "decoding_template.h"

namespace turbomind {

using namespace attention;

using sm70_f16_f16_g1_d128 = Decoding<arch::Sm70, half, half, 1, 128>;
template void invokeDecoding<sm70_f16_f16_g1_d128>(const typename sm70_f16_f16_g1_d128::ParamType& params);

using sm70_f16_f16_g2_d128 = Decoding<arch::Sm70, half, half, 2, 128>;
template void invokeDecoding<sm70_f16_f16_g2_d128>(const typename sm70_f16_f16_g2_d128::ParamType& params);

using sm70_f16_f16_g4_d128 = Decoding<arch::Sm70, half, half, 4, 128>;
template void invokeDecoding<sm70_f16_f16_g4_d128>(const typename sm70_f16_f16_g2_d128::ParamType& params);

using sm70_f16_s8_g1_d128 = Decoding<arch::Sm70, half, int8_t, 1, 128>;
template void invokeDecoding<sm70_f16_s8_g1_d128>(const typename sm70_f16_s8_g1_d128::ParamType& params);

using sm70_f16_s8_g2_d128 = Decoding<arch::Sm70, half, int8_t, 2, 128>;
template void invokeDecoding<sm70_f16_s8_g2_d128>(const typename sm70_f16_s8_g2_d128::ParamType& params);

using sm70_f16_s8_g4_d128 = Decoding<arch::Sm70, half, int8_t, 4, 128>;
template void invokeDecoding<sm70_f16_s8_g4_d128>(const typename sm70_f16_s8_g4_d128::ParamType& params);

}  // namespace turbomind