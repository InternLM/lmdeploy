// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template bool invokeDecoding<Decoding<arch::Sm70, half, half, 1, 128>>(const AttentionParams<half>& params);

template bool invokeDecoding<Decoding<arch::Sm70, half, half, 2, 128>>(const AttentionParams<half>& params);

template bool invokeDecoding<Decoding<arch::Sm70, half, half, 3, 128>>(const AttentionParams<half>& params);

}  // namespace turbomind
