// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template bool
invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, nv_bfloat16, 1, 192>>(const AttentionParams<nv_bfloat16>& params);

template bool invokeDecoding<Decoding<arch::Sm80, half, half, 1, 192>>(const AttentionParams<half>& params);

template bool
invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, uint8_t, 1, 192>>(const AttentionParams<nv_bfloat16>& params);

template bool invokeDecoding<Decoding<arch::Sm80, half, uint8_t, 1, 192>>(const AttentionParams<half>& params);

}  // namespace turbomind
