// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "decoder_multihead_attention_params.h"

namespace turbomind {

template<typename T>
void DispatchDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<T>& params);

}