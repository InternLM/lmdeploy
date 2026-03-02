// Copyright (c) OpenMMLab. All rights reserved.

#include "../attention_params.h"
#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template bool invokeDecoding<Decoding<arch::Sm75, half, uint8_t, 8, 576>>(const AttentionParams<half>& params);

template bool invokeDecoding<Decoding<arch::Sm75, half, uint8_t, 16, 576>>(const AttentionParams<half>& params);

}  // namespace turbomind
