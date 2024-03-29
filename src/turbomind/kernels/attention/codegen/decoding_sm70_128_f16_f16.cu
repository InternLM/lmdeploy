// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template void invokeDecoding<Decoding<arch::Sm70, half, half, 1, 128>>(const AttentionParams<half>& params);

template void invokeDecoding<Decoding<arch::Sm70, half, half, 2, 128>>(const AttentionParams<half>& params);

template void invokeDecoding<Decoding<arch::Sm70, half, half, 4, 128>>(const AttentionParams<half>& params);

}  // namespace turbomind
