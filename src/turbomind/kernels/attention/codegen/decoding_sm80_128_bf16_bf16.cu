// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template bool
invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, nv_bfloat16, 1, 128>>(const AttentionParams<nv_bfloat16>& params);

template bool
invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, nv_bfloat16, 2, 128>>(const AttentionParams<nv_bfloat16>& params);

template bool
invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, nv_bfloat16, 8, 128>>(const AttentionParams<nv_bfloat16>& params);

template bool
invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, nv_bfloat16, 16, 128>>(const AttentionParams<nv_bfloat16>& params);

}  // namespace turbomind
