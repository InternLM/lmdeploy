// Copyright (c) OpenMMLab. All rights reserved.

#include "../attention_params.h"
#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template bool invokeDecoding<Decoding<arch::Sm70, half, uint4_t, 1, 576>>(const AttentionParams<half>& params);

template bool invokeDecoding<Decoding<arch::Sm70, half, uint4_t, 2, 576>>(const AttentionParams<half>& params);

template bool invokeDecoding<Decoding<arch::Sm70, half, uint4_t, 3, 576>>(const AttentionParams<half>& params);

}  // namespace turbomind
