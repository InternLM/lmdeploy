// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind::gemm {

// tags for dispatching & conditional codegen

template<int Begin, int End = -1>
struct Arch {
    static constexpr bool is_compatible(int arch)
    {
        return Begin <= arch && (End == -1 || arch < End);
    }
};

struct Sm70: Arch<700, 750> {
    static constexpr int value = 70;
};

struct Sm75: Arch<750, 800> {
    static constexpr int value = 75;
};

struct Sm80: Arch<800> {
    static constexpr int value = 80;
};

inline bool is_arch_compatible(int karch, int darch)
{
    switch (karch) {
        case 70:
            return Sm70::is_compatible(darch);
        case 75:
            return Sm75::is_compatible(darch);
        case 80:
            return Sm80::is_compatible(darch);
        default:
            return false;
    }
}

}  // namespace turbomind::gemm
