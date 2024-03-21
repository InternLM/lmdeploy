// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "attention_params.h"

namespace turbomind {

template<class T>
void dispatchDecoding(const AttentionParams<T>& params);

}
