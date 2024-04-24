// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template bool invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, uint8_t, 8, 128>>(const AttentionParams<nv_bfloat16>&);

}  // namespace turbomind
