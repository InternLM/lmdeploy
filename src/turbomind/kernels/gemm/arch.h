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
    static constexpr int value = 700;
};

struct Sm75: Arch<750, 800> {
    static constexpr int value = 750;
};

struct Sm80: Arch<800, 900> {
    static constexpr int value = 800;
};

struct Sm90: Arch<900> {
    static constexpr int value = 900;
};

inline bool is_arch_compatible(int karch, int darch)
{
    switch (karch) {
        case 700:
            return Sm70::is_compatible(darch);
        case 750:
            return Sm75::is_compatible(darch);
        case 800:
            return Sm80::is_compatible(darch);
        case 900:
            return Sm90::is_compatible(darch);
        default:
            return false;
    }
}

}  // namespace turbomind::gemm
