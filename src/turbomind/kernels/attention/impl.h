// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

namespace attention {

struct MMA_16816 {
};

struct MMA_81616 {
};  // MMA_16816 transposed

struct MMA_1688 {
};

struct MMA_884 {
};

struct MMA_SIMT {
};

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
struct Impl {
};

}  // namespace attention

}  // namespace turbomind
