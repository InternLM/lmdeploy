// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/verification/attention.h"

namespace turbomind::verification_attention {

bool supports(const Capability&)
{
    return false;
}

int choose_split_count(int, int, int, int, int, int, int)
{
    return 1;
}

void run(const Arguments&)
{
}

}  // namespace turbomind::verification_attention
