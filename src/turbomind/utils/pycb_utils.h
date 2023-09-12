// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdio>

namespace turbomind {

void set_batch_info(int current, int total);

int is_first_in_batch();

int is_last_in_batch();

}  // namespace turbomind
