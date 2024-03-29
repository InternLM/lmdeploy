// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template void invokeDecoding<Decoding<arch::Sm70, half, uint8_t, 1, 128>>(const AttentionParams<half>& params);

}  // namespace turbomind
