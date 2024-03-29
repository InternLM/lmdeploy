// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

using sm80_f16_u8_g1_d128 = Decoding<arch::Sm80, half, uint8_t, 1, 128>;
template void invokeDecoding<sm80_f16_u8_g1_d128>(const typename sm80_f16_u8_g1_d128::ParamType& params);

using sm80_f16_u8_g8_d128 = Decoding<arch::Sm80, half, uint8_t, 8, 128>;
template void invokeDecoding<sm80_f16_u8_g8_d128>(const typename sm80_f16_u8_g8_d128::ParamType& params);

}  // namespace turbomind
