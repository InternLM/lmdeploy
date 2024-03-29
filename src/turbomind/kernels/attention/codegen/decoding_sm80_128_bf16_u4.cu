// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

template void
invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, uint4_t, 8, 128>>(const AttentionParams<nv_bfloat16>& params);

}  // namespace turbomind
