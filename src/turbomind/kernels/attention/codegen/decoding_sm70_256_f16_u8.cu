// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template bool invokeDecoding<Decoding<arch::Sm70, half, uint8_t, 1, 256>>(const AttentionParams<half>& params);

template bool invokeDecoding<Decoding<arch::Sm70, half, uint8_t, 2, 256>>(const AttentionParams<half>& params);

template bool invokeDecoding<Decoding<arch::Sm70, half, uint8_t, 3, 256>>(const AttentionParams<half>& params);

}  // namespace turbomind
