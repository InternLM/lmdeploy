#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/cp_async.h"
#include "src/turbomind/kernels/gemm/iterator_sm70.h"

namespace turbomind::gemm {

namespace sm75_s16816 {

using namespace sm80_s16816;

template<class A, class TransformA, class U, class B, class TransformB, class V, Order order_c, class Tc>
struct Sm75_s16816 {

    static_assert(A::SmemCopyAtom::K == B::SmemCopyAtom::K);

    static constexpr int SMEM_M = A::SmemCopyAtom::M / A::SmemCopyAtom::kFragNum;
    static constexpr int SMEM_N = B::SmemCopyAtom::M / B::SmemCopyAtom::kFragNum;
    static constexpr int SMEM_K = A::SmemCopyAtom::K;

    template<int  CTA_M,
             int  CTA_N,
             int  CTA_K,
             int  TG_M,
             int  TG_N,
             int  TG_K,
             int  Stages,
             bool SplitK,
             int  GroupSizeU = 1,
             int  GroupSizeV = 1>
    struct Type {
        // Raked partition dont support `Pack_M > 1`
        using Partition = Blocked<TG_M, TG_N, kColMajor>;
        using MMA_Map   = MMA_Map<CTA_M, CTA_N, CTA_K, SMEM_M, SMEM_N, SMEM_K, Partition, TG_K>;
        using MMA       = Tiled_MMA_v2<SM80_MMA_16x8x16_F32_F16_F16_F32_TN, MMA_Map>;

        using Mainloop = MainloopSm80_v2<MMA,
                                         A,
                                         IteratorSm70<cache_policy::Default>,
                                         TransformA,
                                         U,
                                         GroupSizeU,
                                         B,
                                         IteratorSm70<cache_policy::Default>,
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
                                         Rearrange<MMA>,
                                         Operand_C<float, order_c>,
                                         SplitK>;

        using Kernel = GemmUniversal<Sm75, Mainloop, Epilogue, CtaMap>;
    };
};

}  // namespace sm75_s16816

}  // namespace turbomind::gemm
