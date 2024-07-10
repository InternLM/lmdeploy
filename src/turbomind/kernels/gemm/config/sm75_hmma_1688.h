#include "src/turbomind/kernels/gemm/config/sm80_hmma_16816.h"
#include "src/turbomind/kernels/gemm/iterator_sm70.h"

namespace turbomind::gemm {

namespace sm75_hmma_1688 {

using namespace sm80_hmma_16816;

template<class A, class TransformA, class U, class B, class TransformB, class V, Order order_c, class Tc>
struct SM75_HMMA_1688_F32 {
    template<int  CTA_M,
             int  CTA_N,
             int  CTA_K,
             int  WARP_CNT_M,
             int  WARP_CNT_N,
             int  WARP_CNT_K,
             int  Stages,
             bool SplitK,
             int  GroupSizeU = 1,
             int  GroupSizeV = 1>
    struct Type {
        using Partition = Raked<WARP_CNT_M, WARP_CNT_N, kColMajor>;
        using MMA_Map   = MMA_Map<CTA_M, CTA_N, CTA_K, 16, 16, 16, Partition, WARP_CNT_K>;
        using MMA       = Tiled_MMA_v2<SM80_MMA_16x8x16_F32_F16_F16_F32_TN, MMA_Map>;

        using Mainloop = MainloopSm80_v2<CTA_M,
                                         CTA_N,
                                         CTA_K,
                                         MMA,
                                         IteratorSm70,
                                         A,
                                         TransformA,
                                         U,
                                         GroupSizeU,
                                         B,
                                         TransformB,
                                         V,
                                         GroupSizeV,
                                         Stages,
                                         false>;  // FusePrefetch_

        using Epilogue = gemm::Epilogue_<Tc,
                                         CTA_M,
                                         CTA_N,
                                         CTA_M,
                                         CTA_N,
                                         MMA::kThreadCount,
                                         typename MMA::Rearrange,
                                         Operand_C<float, order_c>,
                                         SplitK>;

        using Kernel = GemmUniversal<Sm80, Mainloop, Epilogue, CtaMap>;
    };
};

}  // namespace sm75_hmma_1688

}  // namespace turbomind::gemm