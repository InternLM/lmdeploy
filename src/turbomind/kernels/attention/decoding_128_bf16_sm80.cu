// Copyright (c) OpenMMLab. All rights reserved.

#include "decoding_config.h"
#include "decoding_template.h"

namespace turbomind {

using namespace attention;

using sm80_bf16_bf16_g1_d128 = Decoding<arch::Sm80, nv_bfloat16, nv_bfloat16, 1, 128>;
template void invokeDecoding<sm80_bf16_bf16_g1_d128>(const typename sm80_bf16_bf16_g1_d128::ParamType& params);

using sm80_bf16_f16_g2_d128 = Decoding<arch::Sm80, nv_bfloat16, nv_bfloat16, 2, 128>;
template void invokeDecoding<sm80_bf16_f16_g2_d128>(const typename sm80_bf16_f16_g2_d128::ParamType& params);

using sm80_bf16_bf16_g4_d128 = Decoding<arch::Sm80, nv_bfloat16, nv_bfloat16, 4, 128>;
template void invokeDecoding<sm80_bf16_bf16_g4_d128>(const typename sm80_bf16_bf16_g4_d128::ParamType& params);

using sm80_bf16_bf16_g8_d128 = Decoding<arch::Sm80, nv_bfloat16, nv_bfloat16, 8, 128>;
template void invokeDecoding<sm80_bf16_bf16_g8_d128>(const typename sm80_bf16_bf16_g8_d128::ParamType& params);

using sm80_bf16_s8_g1_d128 = Decoding<arch::Sm80, nv_bfloat16, int8_t, 1, 128>;
template void invokeDecoding<sm80_bf16_s8_g1_d128>(const typename sm80_bf16_s8_g1_d128::ParamType& params);

using sm80_bf16_s8_g2_d128 = Decoding<arch::Sm80, nv_bfloat16, int8_t, 2, 128>;
template void invokeDecoding<sm80_bf16_s8_g2_d128>(const typename sm80_bf16_s8_g2_d128::ParamType& params);

}  // namespace turbomind