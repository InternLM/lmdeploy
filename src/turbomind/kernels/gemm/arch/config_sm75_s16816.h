// Copyright (c) OpenMMLab. All rights reserved.

#include <numeric>

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/mma_sm80.h"
#include "src/turbomind/kernels/gemm/arch/operand_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/epilogue.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/kernel/config.h"
#include "src/turbomind/kernels/gemm/kernel_impl.h"
#include "src/turbomind/kernels/gemm/iterator_sm70.h"
#include "src/turbomind/kernels/gemm/mainloop_sm70.h"
#include "src/turbomind/kernels/gemm/scheduler_sm70.cuh"
#include "src/turbomind/kernels/gemm/thread_group_map.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

namespace sm75_s16816 {

using namespace sm80_s16816;

template<Order mma_iter_order,
         class A,
         class TransformA,
         class U,
         class B,
         class TransformB,
         class V,
         Order order_C,
         class Tc,
         int   group_axis>
struct Sm75_s16816 {

    static_assert(A::SmemCopyAtom::K == B::SmemCopyAtom::K);

    static constexpr int SMEM_M = A::SmemCopyAtom::M / A::SmemCopyAtom::kFragNum;
    static constexpr int SMEM_N = B::SmemCopyAtom::M / B::SmemCopyAtom::kFragNum;
    static constexpr int SMEM_K = A::SmemCopyAtom::K;

    static constexpr auto MODE_ = group_axis >= 0 ? Striding::kBlocked : Striding::kFlat;

    static constexpr auto MODE_A = group_axis == 0 ? Striding::kIndexed : MODE_;
    static constexpr auto MODE_B = group_axis == 1 ? Striding::kIndexed : MODE_;
    static constexpr auto MODE_C = MODE_;

    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int GroupSizeU = 1, int GroupSizeV = 1, int EpiM = -1, int EpiN = -1>
    struct Type {
        using Tile = typename Config_::Tile;
        using Groups = typename Config_::Groups;

        static constexpr int CTA_M = Tile::M;
        static constexpr int CTA_N = Tile::N;
        static constexpr int CTA_K = Tile::K;

        // Raked partition dont support `Pack_M > 1`
        using Partition = Blocked<Groups::M, Groups::N, kColMajor>;
        using MMA_Map   = gemm::MMA_Map<CTA_M, CTA_N, CTA_K, SMEM_M, SMEM_N, SMEM_K, Partition, Groups::K>;
        using MMA       = Tiled_MMA_v2<SM80_MMA_16x8x16_F32_F16_F16_F32_TN<half>, MMA_Map, mma_iter_order>;

        using Mainloop = MainloopSm70<MMA,
                                      A,
                                      IteratorSm70<MODE_A, PolicyA>,
                                      TransformA,
                                      U,
                                      GroupSizeU,
                                      B,
                                      IteratorSm70<MODE_B, PolicyB>,
                                      TransformB,
                                      V,
                                      GroupSizeV,
                                      Stages,
                                      true>;  // FusePrefetch_

        static constexpr int CHUNK_K = std::lcm(std::lcm(GroupSizeU, GroupSizeV), CTA_K);

        using Scheduler = SchedulerSm70<Raster, CTA_M, CTA_N, CTA_K, CHUNK_K, SplitK, group_axis>;

        static constexpr int TILE_C_M = EpiM == -1 ? CTA_M : EpiM;
        static constexpr int TILE_C_N = EpiN == -1 ? CTA_N : EpiN;

        using Epilogue = gemm::Epilogue_<Tc,
                                         CTA_M,
                                         CTA_N,
                                         TILE_C_M,
                                         TILE_C_N,
                                         MMA::kThreadCount,
                                         Rearrange<MMA>,
                                         Operand_C<float, order_C>,
                                         MODE_C,
                                         SplitK>;

        using Kernel = KernelImpl<GemmUniversal<Sm75, Mainloop, Epilogue, Scheduler>>;
    };
};

// mma_iter_order has no effect yet

template<int GroupSize>
struct Config_U4_d {
    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1>
    using Type = typename Sm75_s16816<kColMajor, Operand_A<half, kRowMajor>, Transform_Default, VoidOperand, Operand_B_Pack<uint4_t, kColMajor, 2>, Transform_HMMA_16816<1, 0>, Operand_UV_Pack<uint32_t, true>, kRowMajor, half, -1>::template Type<Config_, Stages, Raster, PolicyA, PolicyB, SplitK, 1, GroupSize, EpiM, EpiN>::Kernel;
};

template<int GroupSize>
struct Config_U4_g {
    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1>
    using Type = typename Sm75_s16816<kColMajor, Operand_A<half, kRowMajor>, Transform_Default, VoidOperand, Operand_B_Pack<uint4_t, kRowMajor, 2>, Transform_HMMA_16816<1, 0>, Operand_UV_Pack<uint32_t, true>, kRowMajor, half, 0>::template Type<Config_, Stages, Raster, PolicyA, PolicyB, SplitK, 1, GroupSize, EpiM, EpiN>::Kernel;
};

struct Config_MXF4 {
    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, int GroupAxis = -1>
    using Type = typename Sm75_s16816<kColMajor, Operand_A_Pack<fp4_e2m1_t, kColMajor, 1>, Transform_HMMA_16816<0, 1>, Operand_UV_Pack<uint8_t, false>, Operand_B<half_t, kRowMajor>, Transform_Default, VoidOperand, kColMajor, half_t, GroupAxis>::template Type<Config_, Stages, Raster, PolicyA, PolicyB, SplitK, 32, 1, EpiM, EpiN>::Kernel;
};

struct Config_E4M3 {
    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, int GroupAxis = -1>
    using Type = typename Sm75_s16816<kColMajor, Operand_A_Pack<fp8_e4m3_t, kColMajor, 1>, Transform_HMMA_16816<0, 1>, Operand_UV_Pack<uint16_t, false>, Operand_B<half_t, kRowMajor>, Transform_Default, VoidOperand, kColMajor, half_t, GroupAxis>::template Type<Config_, Stages, Raster, PolicyA, PolicyB, SplitK, 128, 1, EpiM, EpiN>::Kernel;
};

struct Config_F16 {
    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, int GroupAxis = -1>
    using Type = typename Sm75_s16816<kColMajor, Operand_A<half, kRowMajor>, Transform_Default, VoidOperand, Operand_B_Pack<half, kRowMajor, 1>, Transform_Default, VoidOperand, kRowMajor, half, GroupAxis>::template Type<Config_, Stages, Raster, PolicyA, PolicyB, SplitK, 1, 1, EpiM, EpiN>::Kernel;
};

}  // namespace sm75_s16816

}  // namespace turbomind::gemm
