// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind::arch {

// tags for dispatching & conditional codegen

template<int Begin, int End = -1>
struct Arch {
    static constexpr bool is_compatible(int arch)
    {
        return Begin <= arch && (End == -1 || arch < End);
    }
};

struct Sm70: Arch<700, 750> {
};

struct Sm75: Arch<750, 800> {
};

struct Sm80: Arch<800> {
};

}  // namespace turbomind::arch
