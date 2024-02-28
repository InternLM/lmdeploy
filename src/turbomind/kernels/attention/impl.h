// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

namespace attention {

template<int Begin, int End = -1>
struct Arch {
    static constexpr bool is_compatible(int x)
    {
        return Begin <= x && (End == -1 || x <= End);
    }
};

struct Sm80_16816: Arch<80> {};

struct Sm80_81616: Arch<80> {};

struct Sm75_1688: Arch<75, 80> {};

struct Sm70_884: Arch<70, 75> {};

struct Sm70_Simt: Arch<70> {};

template<class Tag,
         class T,
         class Tkv,
         int CTA_H,
         int CTA_Q,
         int CTA_S,
         int WARP_H,
         int WARP_Q,
         int WARP_S,
         int HeadDim,
         int Stages = 2>
struct Impl {};

}  // namespace attention

}  // namespace turbomind