// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template bool
invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, nv_bfloat16, 1, 64>>(const AttentionParams<nv_bfloat16>& params);

template bool
invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, nv_bfloat16, 2, 64>>(const AttentionParams<nv_bfloat16>& params);

template bool
invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, nv_bfloat16, 8, 64>>(const AttentionParams<nv_bfloat16>& params);

template bool
invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, nv_bfloat16, 16, 64>>(const AttentionParams<nv_bfloat16>& params);

}  // namespace turbomind
