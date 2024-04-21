#pragma once

namespace turbomind::gemm {

struct MMA_81616 {};

template<class Tag,
         class T,
         class Tb,
         class Tq,
         int CTA_M,
         int CTA_N,
         int CTA_K,
         int WARP_M,
         int WARP_N,
         int WARP_K,
         int Stages,
         int Flag>
struct Impl {};

}  // namespace turbomind::gemm