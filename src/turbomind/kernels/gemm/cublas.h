// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind::gemm {

class Collector;

inline bool always_available(int)
{
    return true;
}

void add_cublas(Collector& collector, bool (*available)(int) = always_available);

}  // namespace turbomind::gemm
