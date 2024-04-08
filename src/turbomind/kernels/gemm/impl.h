#pragma once

namespace turbomind::gemm {

struct MMA_81616 {};

template<class Tag,
         class T,
         class Tx,
         class Tw,
         int CTA_M,
         int CTA_N,
         int CTA_K,
         int WARP_M,
         int WARP_N,
         int WARP_K,
         int Stages>
struct Impl {};

}  // namespace turbomind::gemm