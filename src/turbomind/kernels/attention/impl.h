// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

class TensorOp {};
class Simt {};

namespace attention {

struct Sm80_16816 {};
struct Sm75_1688 {};
struct Sm70_884 {};
struct Sm70_Simt {};
struct Sm80_16816_Decoding {};

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