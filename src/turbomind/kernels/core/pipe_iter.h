// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

template<int Stages, int Step = 1>
struct PipeIter {
    static constexpr int kMaxStep = Stages * Step;

    int r = 0;
    int w = kMaxStep - Step;

    __inline__ __device__ PipeIter& operator++()
    {
        w = r;
        r += Step;
        if (r == kMaxStep) {
            r -= kMaxStep;
        }
        return *this;
    }
};

}  // namespace turbomind