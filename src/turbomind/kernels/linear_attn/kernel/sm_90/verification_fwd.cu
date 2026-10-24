#include "src/turbomind/kernels/linear_attn/kernel/sm_90/internal.h"

#include "src/turbomind/kernels/linear_attn/kernel/plan.h"
#include "src/turbomind/kernels/linear_attn/kernel/sm_90/common.h"
#include "src/turbomind/kernels/linear_attn/registrar.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

namespace turbomind::linear_attn::delta_rule {
namespace {

using namespace cute;

template<class Operator>
__global__ __launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
void Sm90GdrVerifyCommitDeviceKernel(const __grid_constant__ typename Operator::Params parameters)
{
    extern __shared__ __align__(Operator::SharedMemoryAlignment) unsigned char shared_bytes[];
    auto& shared_storage = *reinterpret_cast<typename Operator::SharedStorage*>(shared_bytes);
    Operator{}(parameters, shared_storage);
}

// verify:
//
//            WG-0                         WG-1                         WG-2
//      TC gram=_gram_(K)             TC P=_p_(Q,K)            prefix=_prefix_(gate)
//        AG=_ag_(gram)              TC O0=_o_sq_(S,Q)            TC U=_u_(S,K)
//     _ag_scale_(prefix)                                         W=_w_(V,U,prefix)
//                                 TC AGW=_agw_(W,AG)
//                              O1=_o_scale_(O0,prefix)
//                                 _p_scale_(P,prefix)
//                              TC O2=_o_final_(O1,AGW,P)
//                                     _store_(O2)
//
// commit:
//
//            WG-0                         WG-1                         WG-2
//      TC gram=_gram_(K)                                      prefix=_prefix_(gate)
//        AG=_ag_(gram)                                          TC U=_u_(S,K)
//     _ag_scale_(prefix)                                        W=_w_(V,U,prefix)
//                                 TC AGW=_agw_(W,AG)
//                              Kc=_k_scale_(K,prefix,L)
//                               TC S'=_commit_(S,AGW,Kc)
//                                     _store_(S')
//

template<int Capacity, DataType StateType = kFloat32, GdrMode Mode = GdrMode::kVerify>
class Sm90GdrVerifyCommitKernel final: public GdrKernel {
private:
    static_assert(Mode == GdrMode::kVerify || Mode == GdrMode::kCommit);
    static constexpr bool kCommit = Mode == GdrMode::kCommit;
    // Unqualified bfloat16_t and gemm resolve against TurboMind declarations;
    // qualify only these compiler-ambiguous CuTe symbols.
    using Element = cute::bfloat16_t;
    using MmaAtom             = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;

    static constexpr int   kHeadDim        = 128;
    static constexpr int   kBlockDv        = 128;
    static constexpr int   kHalfDv         = 64;
    static constexpr int   kMmaM           = 16;
    static constexpr int   kMmaN           = 8;
    static constexpr int   kWarpThreads    = 32;
    static constexpr int   kComputeWarpGroups = 3;
    static constexpr int   kWarpGroups     = kComputeWarpGroups + 1;
    static constexpr int   kTmaGlobalAddressAlignment = 16;
    static constexpr int   kTmaWgmmaSmemAlignment = 1024;
    static constexpr int   kBarrierAlignment = 16;
    static constexpr int   kValueHalves    = kBlockDv / kHalfDv;
    static constexpr int   kCapacityNTiles = Capacity / kMmaN;
    static constexpr float kHeadScale      = 0.08838834764831845f;
    static constexpr int   kStateElementsPerHead = kHeadDim * kBlockDv;
    static constexpr int   kWStageBytes = kBlockDv * kMmaM * sizeof(Element);
    static constexpr int   kRemovedQKVTailSharedBytesPerStage =
        (kMmaM - Capacity) * (2 * kHeadDim + kBlockDv) * sizeof(Element);
    static constexpr int   kRemovedGateExpDifferenceTailSharedBytesPerStage =
        (kMmaM * kMmaM - Capacity * Capacity) * sizeof(float);

    struct Fp32StateTraits {
        using State = float;
        using SmemSwizzle = Swizzle<3, 4, 3>;

        static constexpr DataType  kDataType         = kFloat32;
        static constexpr int       kStages           = 2;
        static constexpr int       kStateOperandOffsetBytes = kStages * kStateElementsPerHead * sizeof(State);
        static constexpr int       kStateOperandStageBytes = 0;
        static constexpr int       kStateStorageBytes =
            kStateOperandOffsetBytes + kStateElementsPerHead * sizeof(Element);
        static constexpr int       kPersistentSharedBytes =
            203776 + 3 * kTmaWgmmaSmemAlignment + 2 * kWStageBytes
            - kRemovedQKVTailSharedBytesPerStage * kStages
            - kRemovedGateExpDifferenceTailSharedBytesPerStage * (kStages + 1)
                  / kTmaWgmmaSmemAlignment * kTmaWgmmaSmemAlignment;
        static constexpr const char* kName8 =
            kCommit ? "sm90_delta_rule_commit8_f32_state" : "sm90_delta_rule_verify8_f32_state";
        static constexpr const char* kName16 =
            kCommit ? "sm90_delta_rule_commit16_f32_state" : "sm90_delta_rule_verify16_f32_state";
    };

    struct Bf16StateTraits {
        using State = Element;
        using SmemSwizzle = Swizzle<2, 4, 3>;

        static constexpr DataType  kDataType         = kBfloat16;
        static constexpr int       kStages           = 3;
        static constexpr int       kStateOperandOffsetBytes = 0;
        static constexpr int       kStateOperandStageBytes = kStateElementsPerHead * sizeof(State);
        static constexpr int       kStateStorageBytes = kStages * kStateOperandStageBytes;
        static constexpr int       kPersistentSharedBytes =
            159744 + kTmaWgmmaSmemAlignment + 2 * kWStageBytes
            - kRemovedQKVTailSharedBytesPerStage * kStages
            - kRemovedGateExpDifferenceTailSharedBytesPerStage * (kStages + 1)
                  / kTmaWgmmaSmemAlignment * kTmaWgmmaSmemAlignment;
        static constexpr const char* kName8 =
            kCommit ? "sm90_delta_rule_commit8_bf16_state" : "sm90_delta_rule_verify8_bf16_state";
        static constexpr const char* kName16 =
            kCommit ? "sm90_delta_rule_commit16_bf16_state" : "sm90_delta_rule_verify16_bf16_state";
    };

    static_assert(StateType == kFloat32 || StateType == kBfloat16);
    using StateTraits = std::conditional_t<StateType == kFloat32, Fp32StateTraits, Bf16StateTraits>;
    using StateT      = typename StateTraits::State;
    using State       = StateT;
    static constexpr int   kStages = StateTraits::kStages;
    static constexpr int   kHandoffStages = kStages + 1;
    static constexpr int   kInvalidRequest = -1;
    static constexpr unsigned kWarpMask = 0xffffffffu;
    static constexpr int   kMinimumTokens = kCommit ? 1 : 2;
    static constexpr int   kLowerTriangularBankPadding = kMmaN / 2;
    static constexpr int   kLowerTriangularRowStride = kMmaM + kLowerTriangularBankPadding;
    static constexpr int   kHardwareNamedBarrierCount = 16;

    enum class NamedBarrierId : int {
        PReady = 0,
        GramReady,
        WReady,
        AGReady = WReady + kHandoffStages,
        StateDv0Ready = AGReady + kHandoffStages,
        StateDv1Ready,
        GatePrefixReady,
        AGSolveReady,
    };

    enum class WarpGroupRole : int {
        GramPAG,
        Output,
        Update,
        Producer,
    };

    static_assert(Capacity == kMmaN || Capacity == kMmaM);
    static_assert(kWarpThreads % kMmaM == 0);
    static_assert(static_cast<int>(NamedBarrierId::AGSolveReady)
                  < kHardwareNamedBarrierCount);

    using HeadDim        = Int<kHeadDim>;
    using BlockDv        = Int<kBlockDv>;
    using HalfDv         = Int<kHalfDv>;
    using MmaM           = Int<kMmaM>;
    using MmaN           = Int<kMmaN>;
    using WarpThreads    = Int<kWarpThreads>;
    using AGBlock        = _4;
    using AGBlocks       = Int<Capacity / AGBlock::value>;
    using QKVector       = HalfDv;
    using QKVectors      = Int<kHeadDim / kHalfDv>;
    using StateDvBoxes   = Int<kBlockDv / kWarpThreads>;
    using StateDkPanels  = Int<kHeadDim / kMmaM>;
    using GateLaneRows   = Int<kWarpThreads / kMmaM>;
    using GateLaneLayout = Layout<Shape<MmaM, GateLaneRows>>;
    using ValueDvCoordinateLayout = Layout<Shape<HalfDv, Int<kValueHalves>>>;

    using SmemLayoutQKTile = decltype(
        tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},
                      Shape<Int<Capacity>, HeadDim>{},
                      Step<_1, _2>{}));
    using SmemLayoutQKMmaOperandTile = decltype(
        tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},
                      Shape<MmaM, HeadDim>{},
                      Step<_1, _2>{}));
    using SmemBf16PointerFlagBits = smem_ptr_flag_bits<sizeof_bits<Element>::value>;
    using SmemQKSwizzle = decltype(get_swizzle_portion(SmemLayoutQKTile{}));
    using SmemLayoutQKTileLinear = decltype(get_nonswizzle_portion(SmemLayoutQKTile{}));
    using SmemLayoutQKTmaTileLinear = decltype(select<1, 3, 2>(
        flatten(zipped_divide(SmemLayoutQKTileLinear{}, Tile<_1, QKVector>{}))));
    using SmemLayoutQKTmaTile = decltype(composition(
        SmemQKSwizzle{}, SmemBf16PointerFlagBits{}, SmemLayoutQKTmaTileLinear{}));
    using SmemLayoutValueTma = decltype(
        tile_to_shape(
            GMMA::Layout_MN_SW128_Atom<Element>{}, Shape<HalfDv, Int<Capacity>>{}));
    using Stages = Int<kStages>;
    using HandoffStages = Int<kHandoffStages>;
    using SmemLayoutAG = decltype(tile_to_shape(
        GMMA::Layout_K_SW32_Atom<Element>{},
        Shape<MmaM, MmaM, HandoffStages>{},
        Step<_1, _2, _3>{}));
    using SmemLayoutP = decltype(tile_to_shape(
        GMMA::Layout_K_SW32_Atom<Element>{},
        Shape<MmaM, MmaM>{},
        Step<_1, _2>{}));
    using SmemLayoutW = decltype(tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<Element>{},
        Shape<BlockDv, MmaM, HandoffStages>{},
        Step<_1, _2, _3>{}));
    using SmemLayoutAGW = decltype(tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<Element>{},
        Shape<BlockDv, MmaM>{},
        Step<_1, _2>{}));
    using WgmmaBOperandTailCoordinates =
        Layout<Shape<Int<Capacity>, MmaN>, Stride<MmaN, _1>>;
    using LowerTriangularRowStride = Int<kLowerTriangularRowStride>;
    using SmemLayoutLowerTriangularTile =
        Layout<Shape<MmaM, MmaM>, Stride<LowerTriangularRowStride, _1>>;
    static constexpr int kLowerTriangularElements = cosize_v<SmemLayoutLowerTriangularTile>;
    using SmemLayoutLowerTriangular =
        Layout<Shape<MmaM, MmaM, Stages>,
               Stride<LowerTriangularRowStride, _1, Int<kLowerTriangularElements>>>;
    using SmemLayoutBetaCheckpoint =
        Layout<Shape<MmaM, Stages>, Stride<_1, MmaM>>;
    using SmemLayoutGateExp =
        Layout<Shape<MmaM, HandoffStages>, Stride<_1, MmaM>>;
    using SmemLayoutGateExpDifferenceStage = decltype(tile_to_shape(
        Layout<Shape<MmaN, MmaN>, Stride<MmaN, _1>>{},
        Shape<Int<Capacity>, Int<Capacity>>{},
        Step<_1, _2>{}));
    using SmemLayoutGateExpDifference = decltype(tile_to_shape(
        SmemLayoutGateExpDifferenceStage{},
        Shape<Int<Capacity>, Int<Capacity>, HandoffStages>{},
        Step<_1, _2, _3>{}));
    static_assert(kLowerTriangularElements >= kMmaM * kMmaM);

    static constexpr int kOutputN        = Capacity;
    static constexpr int kThreads        = kWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int kComputeThreads = kComputeWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int kSVConsumerThreads =
        kComputeThreads - cutlass::NumThreadsPerWarpGroup;
    static constexpr int kComputeWarps         = kComputeThreads / kWarpThreads;
    static constexpr int kProducerWarp         = kComputeWarps;
    static constexpr int kProducerThread       = kComputeThreads;
    static constexpr int kProducerRegisters    = 48;
    static constexpr int kGramPAGRegisters     = 144;
    static constexpr int kOutputRegisters      = 160;
    static constexpr int kUpdateRegisters      = 160;
    static constexpr int kRegisterBudgetPerThread = 128;
    static constexpr int kGramWarpBegin        = 0;
    static constexpr int kAGWarp               = kGramWarpBegin;
    static constexpr int kGramWarps             = kMmaM / kMmaN;
    static constexpr int kGramPAGWarps          = cutlass::NumWarpsPerWarpGroup;
    static constexpr int kOutputHeadCheckpointWarp = cutlass::NumWarpsPerWarpGroup + 2;
    static constexpr int kPrefixWarp           = kComputeWarps - 1;
    static constexpr int kPrefixRowsPerWarp    = Capacity / cutlass::NumWarpsPerWarpGroup;
    static constexpr int kPrefixColumnVectors = Capacity / _4::value;
    static constexpr int kQKBytes                    = Capacity * kHeadDim * sizeof(Element);
    static constexpr int kPayloadValueBytes          = Capacity * kBlockDv * sizeof(Element);
    static constexpr int kStateBytesPerHead          = kStateElementsPerHead * sizeof(State);
    static constexpr int kQKTransactionBytes         = (kCommit ? 1 : 2) * kQKBytes;
    static constexpr int kSVTransactionBytes = kPayloadValueBytes + kStateBytesPerHead;
    static_assert(Capacity % AGBlock::value == 0);
    static_assert(AGBlocks::value <= cutlass::NumWarpsPerWarpGroup);
    static_assert(kGramWarpBegin + kCapacityNTiles <= kGramWarps);
    static_assert(kAGWarp < kGramPAGWarps);
    static_assert(kCapacityNTiles <= cutlass::NumWarpsPerWarpGroup);
    static_assert(Capacity % cutlass::NumWarpsPerWarpGroup == 0);
    static_assert(Capacity % _4::value == 0);
    static_assert(kPrefixWarp / cutlass::NumWarpsPerWarpGroup
                  == static_cast<int>(WarpGroupRole::Update));
    static_assert(kProducerWarp / cutlass::NumWarpsPerWarpGroup
                  == static_cast<int>(WarpGroupRole::Producer));
    static_assert(kProducerRegisters + kGramPAGRegisters + kOutputRegisters + kUpdateRegisters
                  <= kWarpGroups * kRegisterBudgetPerThread);
    static_assert(kMmaM % kMmaN == 0);
    static_assert(kMmaN * kMmaN % kWarpThreads == 0);
    static_assert(kMmaM * kMmaM % kWarpThreads == 0);

    using PrefixWarpGroupThreadLayout =
        Layout<Shape<Int<cutlass::NumWarpsPerWarpGroup>, WarpThreads>,
               Stride<WarpThreads, _1>>;
    using PrefixOutputLaneLayout =
        Layout<Shape<Int<kPrefixRowsPerWarp>, Int<kPrefixColumnVectors>>,
               Stride<_1, Int<kPrefixRowsPerWarp>>>;
    using PrefixOutputRowLayout =
        Layout<Shape<Int<cutlass::NumWarpsPerWarpGroup>, Int<kPrefixRowsPerWarp>>,
               Stride<Int<kPrefixRowsPerWarp>, _1>>;
    using PrefixColumnVectorLayout =
        Layout<Shape<Int<kPrefixColumnVectors>, _4>, Stride<_4, _1>>;
    static_assert(size(PrefixOutputLaneLayout{}) <= kWarpThreads);
    using AGBlockTile = Tile<AGBlock, AGBlock>;
    using AGBlockTranspose = Layout<Shape<AGBlock, AGBlock>, Stride<AGBlock, _1>>;
    using AGBlockMma = TiledMMA<MMA_Atom<UniversalFMA<float>>,
                                Layout<Shape<AGBlock, AGBlock, _1>>>;
    using AGBlockThreads = Int<AGBlock::value * AGBlock::value>;
    static_assert(AGBlockThreads::value <= kWarpThreads);
    using AGScaleVectorElements =
        std::conditional_t<Capacity == kMmaN, _2, _4>;
    using AGScaleColumnVectors = Int<Capacity / AGScaleVectorElements::value>;
    using AGScaleRowsPerWarp = Int<Capacity / cutlass::NumWarpsPerWarpGroup>;
    using AGScaleLaneCoordinates =
        Layout<Shape<AGScaleRowsPerWarp, AGScaleColumnVectors>,
               Stride<_1, AGScaleRowsPerWarp>>;
    using AGScaleRowCoordinates =
        Layout<Shape<Int<cutlass::NumWarpsPerWarpGroup>, AGScaleRowsPerWarp>,
               Stride<AGScaleRowsPerWarp, _1>>;
    using AGScaleTile = Tile<_1, AGScaleVectorElements>;
    using AGScaleLoadAtom = Copy_Atom<
        UniversalCopy<uint_bit_t<sizeof_bits<float>::value
                                 * AGScaleVectorElements::value>>,
        float>;
    using AGScaleStoreAtom = Copy_Atom<
        UniversalCopy<uint_bit_t<sizeof_bits<Element>::value
                                 * AGScaleVectorElements::value>>,
        Element>;
    static_assert(size(AGScaleLaneCoordinates{}) <= kWarpThreads);
    static_assert(size(AGScaleRowCoordinates{}) == Capacity);
    using OutputN = Int<kOutputN>;
    using StateTmaTile = Tile<WarpThreads, MmaM, StateDvBoxes, Int<kValueHalves>, _1>;
    using GmemLayoutState = Layout<Shape<BlockDv, HeadDim, int>,
                                   Stride<_1, BlockDv, Int<kStateElementsPerHead>>>;
    using GmemLayoutStateTma = decltype(select<0, 1, 3, 4, 5>(
        flatten(zipped_divide(GmemLayoutState{}, Tile<WarpThreads, MmaM, _1>{}))));
    using SmemLayoutStateTmaTileLinear =
        Layout<Shape<WarpThreads, MmaM, StateDvBoxes, Int<kValueHalves>>>;
    using SmemLayoutStateTmaTile = decltype(composition(
        typename StateTraits::SmemSwizzle{},
        smem_ptr_flag_bits<sizeof_bits<State>::value>{},
        SmemLayoutStateTmaTileLinear{}));
    using SmemLayoutStatePanel = decltype(take<0, 3>(SmemLayoutStateTmaTile{}));
    using SmemLayoutStatePanels = decltype(append<4>(
        SmemLayoutStatePanel{},
        Layout<StateDkPanels, Int<cosize_v<SmemLayoutStatePanel>>>{}));
    using SmemLayoutState = decltype(append<5>(
        SmemLayoutStatePanels{},
        Layout<Int<kStages>, Int<kStateElementsPerHead>>{}));
    using StateMmaLayout =
        Layout<Shape<Shape<WarpThreads, StateDvBoxes>, Shape<MmaM, StateDkPanels>>,
               Stride<Stride<_1, Int<kWarpThreads * kMmaM>>,
                      Stride<WarpThreads, Int<kBlockDv * kMmaM>>>>;
    using SmemLayoutStateOperandStorage = decltype(composition(
        Swizzle<2, 4, 3>{},
        SmemBf16PointerFlagBits{},
        Layout<Shape<WarpThreads, MmaM, StateDvBoxes, StateDkPanels>>{}));
    using SmemLayoutStateOperand = decltype(
        SmemLayoutStateOperandStorage{}.compose(StateMmaLayout{}));
    using StateConversionThreads = Int<cutlass::NumThreadsPerWarpGroup>;
    using StateConversionVectorElements = _4;
    using StateConversionTile = Tile<HalfDv, HeadDim>;
    using StateConversionIterations =
        Int<size(StateConversionTile{})
            / (StateConversionThreads::value * StateConversionVectorElements::value)>;
    using StateConversionThreadValueLayout =
        Layout<Shape<StateConversionThreads,
                     Shape<StateConversionVectorElements, StateConversionIterations>>,
               Stride<StateConversionVectorElements,
                      Stride<_1,
                             Int<StateConversionThreads::value
                                 * StateConversionVectorElements::value>>>>;
    static_assert(cosize_v<SmemLayoutStateTmaTile> == size(StateTmaTile{}));
    static_assert(size(StateConversionThreadValueLayout{}) == size(StateConversionTile{}));

    using Policy = Sm90GdrVerifyCommitKernel;

    static constexpr const char* kName = Capacity == kMmaN ? StateTraits::kName8 : StateTraits::kName16;
    static constexpr GdrKernelSpec kSpec{
        "sm90", Mode, kBfloat16, StateTraits::kDataType, kHeadDim, Capacity};

    using QKTmaTile = Tile<QKVector, QKVectors, Int<Capacity>, _1, _1>;
    using ValueTmaTile = Tile<HalfDv, Int<Capacity>>;
    using PipelineShape = Shape<_1, _1, _1>;
    using QKMmaTileNK = Tile<MmaN, HeadDim>;
    static constexpr SM90_TMA_LOAD kTmaLoad{};
    using TmaState = decltype(make_tma_copy(
        kTmaLoad,
        make_tensor(make_gmem_ptr(static_cast<const StateT*>(nullptr)), GmemLayoutStateTma{}),
        SmemLayoutStateTmaTile{},
        StateTmaTile{},
        _1{}));
    using MmaAtomThrLayout = Layout<Shape<_1, _1, _1>>;
    using WgmmaAtomLayout = Layout<Shape<_1, _1, _1>>;
    using QKTiledMma = decltype(make_tiled_mma(MmaAtom{}, MmaAtomThrLayout{}, Tile<MmaM, MmaN, MmaM>{}));
    using TokenRows = std::conditional_t<
        Capacity == kMmaN,
        Coord<Underscore, _0>,
        Coord<Underscore, Underscore>>;
    using PAccumulatorStoreOperation = std::conditional_t<
        Capacity == kMmaN,
        SM90_U32x1_STSM_N,
        SM90_U32x2_STSM_N>;
    using SSWgmmaTiledMma = decltype(make_tiled_mma(
        GMMA::ss_op_selector<Element,
                             Element,
                             float,
                             Shape<HalfDv, OutputN, MmaM>,
                             GMMA::Major::MN,
                             GMMA::Major::K>(),
        WgmmaAtomLayout{}));
    using QKMmaAValidTVCoordinate =
        Coord<Underscore,
              Coord<Coord<Underscore, _0, Underscore>, Coord<Underscore, Underscore>>>;
    using QKMmaAValidTVLayout = decltype(
        QKTiledMma{}.get_layoutA_TV()(QKMmaAValidTVCoordinate{}));
    using QKCapacityMmaATVLayout = decltype(
        group<1, QKMmaAValidTVLayout::rank>(QKMmaAValidTVLayout{}));
    using QKCapacityTiledCopyA = decltype(make_tiled_copy_impl(
        Copy_Atom<SM75_U32x2_LDSM_N, Element>{},
        QKCapacityMmaATVLayout{},
        make_shape(MmaM{}, MmaM{})));
    using QKMmaFullTiledCopyA = decltype(make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, QKTiledMma{}));
    using QKTiledCopyA = std::conditional_t<Capacity == kMmaN,
                                             QKCapacityTiledCopyA,
                                             QKMmaFullTiledCopyA>;
    using QKCopyOperationA = std::conditional_t<Capacity == kMmaN,
                                                 SM75_U32x2_LDSM_N,
                                                 SM75_U32x4_LDSM_N>;
    using QKCopyAtomA = Copy_Atom<QKCopyOperationA, Element>;
    using QKMmaAFragmentMCoordinate = std::conditional_t<Capacity == kMmaN,
                                                          Coord<Underscore, _0, Underscore>,
                                                          Underscore>;
    using SmemCopyAtomBKMajor  = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
    using WgmmaAccumulatorStoreOperation =
        std::conditional_t<Capacity == kMmaN, SM90_U16x4_STSM_T, SM90_U16x8_STSM_T>;
    using WgmmaAccumulatorStoreAtom = Copy_Atom<WgmmaAccumulatorStoreOperation, Element>;
    using QKTiledCopyBKMajor = decltype(make_tiled_copy_B(SmemCopyAtomBKMajor{}, QKTiledMma{}));
    using OutputStoreAtom = Copy_Atom<UniversalCopy<uint16_t>, Element>;
    using OutputStoreVectorElements = Int<OutputStoreAtom::NumValSrc>;
    using OutputStoreTiledCopy =
        decltype(make_tiled_copy_C(OutputStoreAtom{}, SSWgmmaTiledMma{}));
    static_assert(OutputStoreAtom::NumValSrc == OutputStoreAtom::NumValDst);
    static_assert(size(SSWgmmaTiledMma{}) == cutlass::NumThreadsPerWarpGroup);

    using StateUpdateTiledMma = decltype(make_tiled_mma(
        GMMA::ss_op_selector<Element, Element, float,
                             Shape<HalfDv, HalfDv, MmaM>,
                             GMMA::Major::MN, GMMA::Major::MN>(),
        WgmmaAtomLayout{}));
    using StateUpdateTile = Tile<HalfDv, HalfDv>;
    using StateUpdateTiles = Layout<Shape<Int<kValueHalves>, Int<kHeadDim / kHalfDv>>>;
    using SmemLayoutCommitKey = decltype(tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<Element>{}, Shape<HeadDim, MmaM>{}));
    using CommitKeyCopyAtom = Copy_Atom<UniversalCopy<uint128_t>, Element>;
    using CommitKeyTiledCopy = decltype(make_tiled_copy(
        CommitKeyCopyAtom{},
        Layout<Shape<_16, _8>, Stride<_1, _16>>{},
        Layout<Shape<_8, _1>>{}));

    using QKPipeline = cutlass::PipelineTmaAsync<kStages>;
    using SVPipeline = cutlass::PipelineTmaAsync<kStages>;
    using QKPipelineState = typename QKPipeline::PipelineState;
    using SVPipelineState = typename SVPipeline::PipelineState;
    using QKPipelineStorage = typename QKPipeline::SharedStorage;
    using SVPipelineStorage = typename SVPipeline::SharedStorage;
    using SmemLayoutQK = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<Element>{},
        Shape<Int<Capacity>, HeadDim, Int<kStages>>{},
        Step<_1, _2, _3>{}));
    using SmemLayoutQKLinear = decltype(get_nonswizzle_portion(SmemLayoutQK{}));
    using SmemLayoutQKTmaLinear = decltype(select<1, 4, 2, 3, 0, 5>(
        flatten(zipped_divide(SmemLayoutQKLinear{}, Tile<_1, QKVector, _1>{}))));
    using SmemLayoutQKTma = decltype(composition(
        SmemQKSwizzle{}, SmemBf16PointerFlagBits{}, SmemLayoutQKTmaLinear{}));
    using SmemLayoutValue = decltype(tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<Element>{},
        Shape<HalfDv, Int<Capacity>, Int<kValueHalves>, Int<kStages>>{}));
    using SmemLayoutGate = Layout<Shape<MmaM, GateLaneRows, Int<kStages>>,
                                  Stride<_1, MmaM, WarpThreads>>;
    using SmemLayoutBeta = SmemLayoutGate;

    static_assert(SmemLayoutQK{}(_0{}, _0{}, _1{}) == Capacity * kHeadDim);
    static_assert(SmemLayoutValue{}(_0{}, _0{}, _0{}, _1{}) == Capacity * kBlockDv);
    static_assert(SmemLayoutState{}(_0{}, _0{}, _0{}, _0{}, _1{})
                  == kStateElementsPerHead);

    struct OutputHeadCoordinate {
        int request;
        int value_head;
        int commit_length;
    };

    struct alignas(kTmaWgmmaSmemAlignment) SharedStorage {
        using T = typename Policy::Element;
        array_aligned<unsigned char,
                      StateTraits::kStateStorageBytes,
                      Policy::kTmaWgmmaSmemAlignment> state;
        array_aligned<T,
                      cosize_v<SmemLayoutQK>,
                      Policy::kTmaWgmmaSmemAlignment> q;
        array_aligned<T,
                      cosize_v<SmemLayoutQK>,
                      Policy::kTmaWgmmaSmemAlignment> k;
        array_aligned<T,
                      cosize_v<SmemLayoutValue>,
                      Policy::kTmaWgmmaSmemAlignment> value;
        array_aligned<float,
                      cosize_v<SmemLayoutGate>,
                      Policy::kBarrierAlignment> gate;
        array_aligned<float,
                      cosize_v<SmemLayoutBeta>,
                      Policy::kBarrierAlignment> beta;
        array_aligned<float,
                      cosize_v<SmemLayoutBetaCheckpoint>,
                      Policy::kBarrierAlignment> beta_checkpoint;
        array_aligned<float,
                      cosize_v<SmemLayoutGateExp>,
                      Policy::kBarrierAlignment> gate_exp;
        array_aligned<float,
                      cosize_v<SmemLayoutGateExpDifference>,
                      Policy::kBarrierAlignment> gate_exp_difference;
        alignas(Policy::kBarrierAlignment) QKPipelineStorage qk_pipeline;
        alignas(Policy::kBarrierAlignment) SVPipelineStorage sv_pipeline;
        alignas(Policy::kBarrierAlignment) cutlass::arch::ClusterBarrier output_head_ready[kStages];
        alignas(Policy::kBarrierAlignment) uint64_t prefix_ready[kHandoffStages];
        array_aligned<OutputHeadCoordinate,
                      Policy::kStages,
                      Policy::kBarrierAlignment> output_heads;
        array_aligned<OutputHeadCoordinate,
                      Policy::kStages,
                      Policy::kBarrierAlignment> output_head_checkpoints;
        array_aligned<T,
                      cosize_v<SmemLayoutAG>,
                      Policy::kTmaWgmmaSmemAlignment> AG_bf16;
        array_aligned<T,
                      cosize_v<SmemLayoutP>,
                      Policy::kTmaWgmmaSmemAlignment> P_bf16;
        array_aligned<T,
                      cosize_v<SmemLayoutW>,
                      Policy::kTmaWgmmaSmemAlignment> W_bf16;
        array_aligned<T,
                      cosize_v<SmemLayoutAGW>,
                      Policy::kTmaWgmmaSmemAlignment> AGW_bf16;
        array_aligned<float,
                      cosize_v<SmemLayoutLowerTriangular>,
                      Policy::kBarrierAlignment> lower;
    };

    static constexpr size_t kSharedBytes = sizeof(SharedStorage);
    static_assert(kSharedBytes == StateTraits::kPersistentSharedBytes);
    // Commit reuses the unused query allocation for its final scaled-key operand.
    static_assert(cosize_v<SmemLayoutQK> >= cosize_v<SmemLayoutCommitKey>);

    template<NamedBarrierId BarrierId, int ParticipantWarps>
    struct NamedBarrier {
        static_assert(static_cast<int>(BarrierId) >= 0
                      && static_cast<int>(BarrierId) < kHardwareNamedBarrierCount);
        static_assert(ParticipantWarps > 0);

        static CUTE_DEVICE void arrive()
        {
            asm volatile(
                "barrier.arrive %0, %1;"
                :
                : "n"(static_cast<int>(BarrierId)), "n"(ParticipantWarps * kWarpThreads)
                : "memory");
        }

        static CUTE_DEVICE void sync()
        {
            asm volatile(
                "barrier.sync %0, %1;"
                :
                : "n"(static_cast<int>(BarrierId)), "n"(ParticipantWarps * kWarpThreads)
                : "memory");
        }
    };

    template<NamedBarrierId FirstBarrierId, int Stages, int ParticipantWarps>
    struct StagedNamedBarrier {
        static_assert(static_cast<int>(FirstBarrierId) >= 0);
        static_assert(static_cast<int>(FirstBarrierId) + Stages <= kHardwareNamedBarrierCount);
        static_assert(ParticipantWarps > 0);

        static CUTE_DEVICE void arrive(int stage)
        {
            const int barrier_id = static_cast<int>(FirstBarrierId) + stage;
            asm volatile(
                "barrier.arrive %0, %1;"
                :
                : "r"(barrier_id), "n"(ParticipantWarps * kWarpThreads)
                : "memory");
        }

        static CUTE_DEVICE void sync(int stage)
        {
            const int barrier_id = static_cast<int>(FirstBarrierId) + stage;
            asm volatile(
                "barrier.sync %0, %1;"
                :
                : "r"(barrier_id), "n"(ParticipantWarps * kWarpThreads)
                : "memory");
        }
    };

    using GramReady =
        NamedBarrier<NamedBarrierId::GramReady, kGramPAGWarps>;
    using WReady = StagedNamedBarrier<NamedBarrierId::WReady,
                                      kHandoffStages,
                                      2 * cutlass::NumWarpsPerWarpGroup>;
    using AGReady = StagedNamedBarrier<NamedBarrierId::AGReady,
                                       kHandoffStages,
                                       2 * cutlass::NumWarpsPerWarpGroup>;
    using StateDv0Ready =
        NamedBarrier<NamedBarrierId::StateDv0Ready,
                     2 * cutlass::NumWarpsPerWarpGroup>;
    using StateDv1Ready =
        NamedBarrier<NamedBarrierId::StateDv1Ready,
                     2 * cutlass::NumWarpsPerWarpGroup>;
    using PReady =
        NamedBarrier<NamedBarrierId::PReady, cutlass::NumWarpsPerWarpGroup>;
    using GatePrefixReady =
        NamedBarrier<NamedBarrierId::GatePrefixReady, cutlass::NumWarpsPerWarpGroup>;
    using AGSolveReady =
        NamedBarrier<NamedBarrierId::AGSolveReady, kGramPAGWarps>;

    template<class StateReady, class StateSource, class StateOperand, class StateDvHalf>
    static CUTE_DEVICE void ConvertAndReleaseStateDvHalf(Fp32StateTraits,
                                                         StateSource state_source,
                                                         StateOperand state_operand,
                                                         int state_conversion_thread,
                                                         StateDvHalf state_dv_half)
    {
        auto state_source_tile =
            local_tile(state_source,
                       StateConversionTile{},
                       make_coord(state_dv_half, _0{}));
        auto state_operand_tile =
            local_tile(state_operand,
                       StateConversionTile{},
                       make_coord(state_dv_half, _0{}));
        auto state_source_partition =
            state_source_tile.compose(StateConversionThreadValueLayout{});
        auto state_operand_partition =
            state_operand_tile.compose(StateConversionThreadValueLayout{});
        CUTE_UNROLL
        for (int iteration = 0; iteration < StateConversionIterations::value; ++iteration) {
            auto source_registers = make_tensor<float>(Shape<StateConversionVectorElements>{});
            copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                 state_source_partition(state_conversion_thread, make_coord(_, iteration)),
                 source_registers);
            auto converted_registers = make_tensor<Element>(Shape<StateConversionVectorElements>{});
            recast<cutlass::Array<Element, StateConversionVectorElements::value>>(
                converted_registers)(0) =
                cutlass::NumericArrayConverter<Element,
                                               float,
                                               StateConversionVectorElements::value>{}(
                    recast<cutlass::Array<float, StateConversionVectorElements::value>>(
                        source_registers)(0));
            copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                 converted_registers,
                 state_operand_partition(state_conversion_thread, make_coord(_, iteration)));
        }
        cutlass::arch::fence_view_async_shared();
        // This WG releases the fenced Dv half to the peer WG without waiting for it.
        StateReady::arrive();
    }

    template<class StateReady, class StateSource, class StateOperand, class StateDvHalf>
    static CUTE_DEVICE void ConvertAndReleaseStateDvHalf(Bf16StateTraits,
                                                         StateSource,
                                                         StateOperand,
                                                         int,
                                                         StateDvHalf)
    {
    }

    template<class StateReady>
    static CUTE_DEVICE void AcquireStateDvHalf(Fp32StateTraits)
    {
        // This WG acquires the peer WG's converted Dv half before its WGMMA reads it.
        StateReady::sync();
    }

    template<class StateReady>
    static CUTE_DEVICE void AcquireStateDvHalf(Bf16StateTraits)
    {
    }

    template<class QKTensor, class KeyTensor, class Accumulator>
    static CUTE_DEVICE void mma_qk_by_k_tile(QKTensor     qk,
                                              KeyTensor    key,
                                              Accumulator& accumulator,
                                              int          lane,
                                              int          k_tile_index)
    {
        auto mma             = QKTiledMma{};
        auto thread_mma      = mma.get_thread_slice(lane);
        auto key_tile        = local_tile(key, QKMmaTileNK{}, make_coord(k_tile_index, _0{}));
        // MMA A is [M=16,K=16]. K8 loads its valid [M=8,K=16] half with
        // LDSM.x2; the cleared upper half supplies zero rows to mma.sync.
        auto qk_mma_operand = make_tensor(qk.data(), SmemLayoutQKMmaOperandTile{});
        auto qk_fragment    = thread_mma.partition_fragment_A(qk_mma_operand);
        auto key_fragment    = thread_mma.partition_fragment_B(key_tile);
        auto copy_a          = QKTiledCopyA{};
        auto copy_b          = QKTiledCopyBKMajor{};
        auto thread_copy_a   = copy_a.get_thread_slice(lane);
        auto thread_copy_b   = copy_b.get_thread_slice(lane);
        auto qk_source       = thread_copy_a.partition_S(qk);
        auto key_source      = thread_copy_b.partition_S(key_tile);
        auto qk_valid_fragment = filter_zeros(
            qk_fragment(QKMmaAFragmentMCoordinate{}, _, _));
        auto qk_source_by_k_block = group_modes<0, decltype(rank(qk_source))::value - 1>(
            qk_source);
        auto qk_fragment_by_k_block =
            group_modes<0, decltype(rank(qk_valid_fragment))::value - 1>(
                qk_valid_fragment);
        auto key_view        = thread_copy_b.retile_D(key_fragment);
        clear(qk_fragment);
        CUTE_UNROLL
        // X[N,N] += QK[N,Dk] * K^T[Dk,N]
        for (int block = 0; block < size<2>(qk_fragment); ++block) {
            copy(QKCopyAtomA{},
                 qk_source_by_k_block(_, block),
                 qk_fragment_by_k_block(_, block));
            copy(copy_b, key_source(_, _, block), key_view(_, _, block));
            cute::gemm(thread_mma,
                       qk_fragment(_, _, block),
                       key_fragment(_, _, block),
                       accumulator);
        }
    }

    template<class LowerTensor>
    static CUTE_DEVICE auto ComputeAGInverse(LowerTensor lower, int lane, int warp)
    {
        if (warp < AGBlocks::value && lane == 0) {
            auto diagonal = local_tile(lower, AGBlockTile{}, make_coord(warp, warp));
            auto lower_block = make_tensor<float>(AGBlockTranspose{});
            auto inverse = make_tensor<float>(AGBlockTranspose{});
            CUTE_UNROLL
            for (int row = 0; row < AGBlock::value; ++row) {
                copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                     diagonal(row, _),
                     lower_block(row, _));
                CUTE_UNROLL
                for (int column = 0; column < AGBlock::value; ++column) {
                    inverse(row, column) = row == column ? 1.0f : 0.0f;
                }
            }
            CUTE_UNROLL
            for (int row = 1; row < AGBlock::value; ++row) {
                CUTE_UNROLL
                for (int column = 0; column < row; ++column) {
                    float value = 0.0f;
                    CUTE_UNROLL
                    for (int middle = 0; middle < row; ++middle) {
                        value -= lower_block(row, middle) * inverse(middle, column);
                    }
                    inverse(row, column) = value;
                }
            }
            copy(inverse, diagonal);
        }
        // WG0 publishes the inverted 4x4 diagonal blocks.
        AGSolveReady::sync();

        // Merge adjacent 4x4 inverses into 8x8, then merge the two 8x8
        // inverses into the complete 16x16 inverse.
        CUTE_UNROLL
        for (int half_blocks = 1; half_blocks < AGBlocks::value; half_blocks *= _2::value) {
            const int blocks_per_merge = _2::value * half_blocks;
            const int merges = AGBlocks::value / blocks_per_merge;
            auto merge_layout = make_layout(
                make_shape(half_blocks, half_blocks, merges));

            if (warp < size(merge_layout)) {
                const auto merge_coordinate = merge_layout.get_hier_coord(warp);
                const int row_in_half = int(get<0>(merge_coordinate));
                const int column_in_half = int(get<1>(merge_coordinate));
                const int merge = int(get<2>(merge_coordinate));
                const int first_block = merge * blocks_per_merge;
                const int block_row = first_block + half_blocks + row_in_half;
                const int block_column = first_block + column_in_half;

                if (lane < AGBlockThreads::value) {
                    auto mma = AGBlockMma{};
                    auto thread_mma = mma.get_thread_slice(lane);
                    auto identity = make_identity_tensor(
                        make_shape(AGBlock{}, AGBlock{}));
                    auto coordinates = thread_mma.partition_C(identity);
                    auto right_product = thread_mma.make_fragment_C(coordinates);
                    clear(right_product);
                    CUTE_UNROLL
                    for (int inner = 0; inner < half_blocks; ++inner) {
                        const int middle_block = first_block + half_blocks + inner;
                        auto right_inverse = local_tile(
                            lower, AGBlockTile{}, make_coord(block_row, middle_block));
                        auto cross_block = local_tile(
                            lower, AGBlockTile{}, make_coord(middle_block, block_column));
                        auto cross_block_transpose = cross_block.compose(AGBlockTranspose{});
                        cute::gemm(mma,
                                   thread_mma.partition_A(right_inverse),
                                   thread_mma.partition_B(cross_block_transpose),
                                   right_product);
                    }
                    auto right_product_scratch = local_tile(
                        lower, AGBlockTile{}, make_coord(block_column, block_row));
                    copy(right_product, thread_mma.partition_C(right_product_scratch));
                }
            }
            // The right-half products are stored in the unused upper blocks.
            AGSolveReady::sync();

            if (warp < size(merge_layout)) {
                const auto merge_coordinate = merge_layout.get_hier_coord(warp);
                const int row_in_half = int(get<0>(merge_coordinate));
                const int column_in_half = int(get<1>(merge_coordinate));
                const int merge = int(get<2>(merge_coordinate));
                const int first_block = merge * blocks_per_merge;
                const int block_row = first_block + half_blocks + row_in_half;
                const int block_column = first_block + column_in_half;

                if (lane < AGBlockThreads::value) {
                    auto mma = AGBlockMma{};
                    auto thread_mma = mma.get_thread_slice(lane);
                    auto identity = make_identity_tensor(
                        make_shape(AGBlock{}, AGBlock{}));
                    auto coordinates = thread_mma.partition_C(identity);
                    auto inverse = thread_mma.make_fragment_C(coordinates);
                    clear(inverse);
                    CUTE_UNROLL
                    for (int inner = 0; inner < half_blocks; ++inner) {
                        const int middle_block = first_block + inner;
                        auto right_product = local_tile(
                            lower, AGBlockTile{}, make_coord(middle_block, block_row));
                        auto left_inverse = local_tile(
                            lower, AGBlockTile{}, make_coord(middle_block, block_column));
                        auto left_inverse_transpose = left_inverse.compose(AGBlockTranspose{});
                        cute::gemm(mma,
                                   thread_mma.partition_A(right_product),
                                   thread_mma.partition_B(left_inverse_transpose),
                                   inverse);
                    }
                    auto inverse_values = coalesce(inverse);
                    CUTE_UNROLL
                    for (int index = 0; index < size(inverse_values); ++index) {
                        inverse_values(index) = -inverse_values(index);
                    }
                    auto inverse_block = local_tile(
                        lower, AGBlockTile{}, make_coord(block_row, block_column));
                    copy(inverse, thread_mma.partition_C(inverse_block));
                }
            }
            // All warps must finish reading the upper-block scratch before it is cleared.
            AGSolveReady::sync();
            if (warp < size(merge_layout)) {
                const auto merge_coordinate = merge_layout.get_hier_coord(warp);
                const int row_in_half = int(get<0>(merge_coordinate));
                const int column_in_half = int(get<1>(merge_coordinate));
                const int merge = int(get<2>(merge_coordinate));
                const int first_block = merge * blocks_per_merge;
                const int block_row = first_block + half_blocks + row_in_half;
                const int block_column = first_block + column_in_half;

                if (lane < AGBlockThreads::value) {
                    auto thread_mma = AGBlockMma{}.get_thread_slice(lane);
                    auto right_product_scratch = local_tile(
                        lower, AGBlockTile{}, make_coord(block_column, block_row));
                    clear(thread_mma.partition_C(right_product_scratch));
                }
            }
            // Publish a zero upper triangle for the next merge and final AG scaling.
            AGSolveReady::sync();
        }
        return lower;
    }

    struct HeadCoordinate {
        int request;
        int value_head;
        int query_head;
        int head_group;
        int local_head;
    };

    static CUTE_DEVICE HeadCoordinate DecodeHead(int flattened_head,
                                                 int batch,
                                                 int hq,
                                                 int hv,
                                                 int value_heads_per_query_head,
                                                 int num_head_groups,
                                                 int heads_per_block)
    {
        auto head_layout = make_layout(make_shape(hv, batch));
        const auto head_coordinate = head_layout.get_hier_coord(flattened_head);
        const int  value_head       = int(get<0>(head_coordinate));
        const int  request          = int(get<1>(head_coordinate));

        auto query_head_layout = make_layout(make_shape(value_heads_per_query_head, hq));
        const auto query_head_coordinate = query_head_layout.get_hier_coord(value_head);

        auto state_head_layout = make_layout(make_shape(heads_per_block, num_head_groups));
        const auto state_head_coordinate = state_head_layout.get_hier_coord(value_head);
        return {request,
                value_head,
                int(get<1>(query_head_coordinate)),
                int(get<1>(state_head_coordinate)),
                int(get<0>(state_head_coordinate))};
    }

    template<class OutputCopy,
             class OutputFragment,
             class OutputCoordinates,
             class OutputTile,
             class StoreVectorElements,
             class ValidCoordinate>
    static CUTE_DEVICE void StoreOutput(OutputCopy        output_copy,
                                        OutputFragment&   output_fragment,
                                        OutputCoordinates output_coordinates,
                                        OutputTile        output_tile,
                                        StoreVectorElements,
                                        ValidCoordinate,
                                        int               valid_positions,
                                        int               thread)
    {
        auto output_thread      = output_copy.get_thread_slice(thread);
        auto output_source      = output_thread.retile_S(output_fragment);
        auto output_destination = output_thread.partition_D(output_tile);
        auto output_coordinate  = output_thread.retile_S(output_coordinates);
        auto output_source_vectors =
            zipped_divide(output_source, make_tile(StoreVectorElements{}));
        auto output_destination_vectors =
            zipped_divide(output_destination, make_tile(StoreVectorElements{}));
        auto output_coordinate_vectors =
            zipped_divide(output_coordinate, make_tile(StoreVectorElements{}));
        CUTE_UNROLL
        for (int vector = 0; vector < size<1>(output_coordinate_vectors); ++vector) {
            const auto coordinate = output_coordinate_vectors(_0{}, vector);
            if (int(get<ValidCoordinate::value>(coordinate)) < valid_positions) {
                auto output_vector = make_tensor<Element>(shape<0>(output_source_vectors));
                copy(output_source_vectors(_, vector), output_vector);
                copy(output_copy,
                     output_vector,
                     output_destination_vectors(_, vector));
            }
        }
    }

    static CUTE_DEVICE void ProcessGramPAGHead(SharedStorage& shared_storage,
                                              int            read_stage,
                                              int            handoff_stage,
                                              QKPipeline&    qk_pipeline,
                                              QKPipelineState qk_pipe_release,
                                              int            lane,
                                              int            warp,
                                              int            handoff_phase)
    {
        const int valid_positions =
            kCommit ? shared_storage.output_heads[read_stage].commit_length : Capacity;
        auto key_smem = make_tensor(
            make_smem_ptr(shared_storage.k.data()), SmemLayoutQK{});
        auto key = key_smem(_, _, read_stage);
        auto gate_exp_difference_storage = make_tensor(
            make_smem_ptr(shared_storage.gate_exp_difference.data()),
            SmemLayoutGateExpDifference{});
        auto gate_exp_difference = gate_exp_difference_storage(_, _, handoff_stage);

        auto AG_storage = make_tensor(
            make_smem_ptr(shared_storage.AG_bf16.data()), SmemLayoutAG{});
        auto AG = AG_storage(_, _, handoff_stage);
        auto beta_smem = make_tensor(
            make_smem_ptr(shared_storage.beta.data()), SmemLayoutBeta{});
        auto beta = beta_smem(_, _0{}, read_stage);
        auto beta_checkpoint_storage = make_tensor(
            make_smem_ptr(shared_storage.beta_checkpoint.data()),
            SmemLayoutBetaCheckpoint{});
        auto beta_checkpoint = beta_checkpoint_storage(_, read_stage);
        auto lower_storage = make_tensor(
            make_smem_ptr(shared_storage.lower.data()), SmemLayoutLowerTriangular{});
        auto lower = lower_storage(_, _, read_stage);

        if (warp == kAGWarp && lane < kMmaM) {
            beta_checkpoint(lane) = beta(lane);
        }

        // _gram_: Gram[N,N] = diag(beta) * StrictLower(K[N,Dk] * K^T[Dk,N])
        const int gram_tile = warp - kGramWarpBegin;
        if (gram_tile < kCapacityNTiles) {
            const int col0     = gram_tile * kMmaN;
            auto      mma      = QKTiledMma{};
            auto      thr      = mma.get_thread_slice(lane);
            auto      identity = make_identity_tensor(make_shape(MmaM{}, MmaN{}));
            auto      coords   = thr.partition_C(identity);
            auto      rC       = thr.make_fragment_C(coords);
            clear(rC);
            mma_qk_by_k_tile(key, key, rC, lane, gram_tile);
            auto values      = coalesce(rC);
            auto coordinates = coalesce(coords);
            CUTE_UNROLL
            for (int index = 0; index < size(values); ++index) {
                const auto coordinate = coordinates(index);
                const int  row        = int(get<0>(coordinate));
                const int  column     = col0 + int(get<1>(coordinate));
                values(index) = column < row && (!kCommit || row < valid_positions) ? beta(row) * values(index) : 0.0f;
            }
            auto lower_tiled_copy = make_tiled_copy_C(
                Copy_Atom<UniversalCopy<uint64_t>, float>{}, mma);
            auto lower_thread_copy = lower_tiled_copy.get_thread_slice(lane);
            auto lower_source      = lower_thread_copy.retile_S(rC);
            auto lower_destination = lower_thread_copy.partition_D(lower);
            copy(lower_tiled_copy,
                 lower_source(_, _0{}, _0{}),
                 lower_destination(_, _0{}, gram_tile));
        }

        // The Gram-role warps publish the complete Gram matrix before the AG solver starts.
        GramReady::sync();
        // The Gram/AG warps release their Q/K-stage participation after Gram.
        qk_pipeline.consumer_release(qk_pipe_release);

        auto AG_inverse = ComputeAGInverse(lower, lane, warp);
        // _ag_: WG0 owns the final AG scaling.
        // WG0 acquires the completed prefix before _ag_scale_.
        wait_barrier(shared_storage.prefix_ready[handoff_stage], handoff_phase);

        // _ag_scale_: AG[row,column] *= exp(prefix[row] - prefix[column]) * beta[column]
        if (lane < size(AGScaleLaneCoordinates{})) {
            const auto coordinate = AGScaleLaneCoordinates{}.get_hier_coord(lane);
            const int row = int(AGScaleRowCoordinates{}(warp, get<0>(coordinate)));
            const int column_vector = int(get<1>(coordinate));
            auto inverse = make_tensor<float>(Shape<AGScaleVectorElements>{});
            auto gate_difference = make_tensor<float>(Shape<AGScaleVectorElements>{});
            auto beta_vector = make_tensor<float>(Shape<AGScaleVectorElements>{});
            copy(AGScaleLoadAtom{},
                 coalesce(local_tile(
                     AG_inverse, AGScaleTile{}, make_coord(row, column_vector))),
                 inverse);
            copy(AGScaleLoadAtom{},
                 coalesce(local_tile(
                     gate_exp_difference,
                     AGScaleTile{},
                     make_coord(row, column_vector))),
                 gate_difference);
            copy(AGScaleLoadAtom{},
                 local_tile(beta_checkpoint,
                            Tile<AGScaleVectorElements>{},
                            make_coord(column_vector)),
                 beta_vector);
            auto transformed = make_tensor<Element>(Shape<AGScaleVectorElements>{});
            CUTE_UNROLL
            for (int element = 0; element < AGScaleVectorElements::value; ++element) {
                const int column = column_vector * AGScaleVectorElements::value + element;
                transformed(element) = kCommit && (row >= valid_positions || column > row) ? Element{} :
                    static_cast<Element>(inverse(element) * gate_difference(element) * beta_vector(element));
            }
            copy(AGScaleStoreAtom{},
                 transformed,
                 coalesce(local_tile(
                     AG, AGScaleTile{}, make_coord(row, column_vector))));
        }
        // WG0 releases its complete AG shared-memory tile to WG1.
        AGReady::arrive(handoff_stage);
    }

    static CUTE_DEVICE void ProcessFinalHead(SharedStorage&          shared_storage,
                                              int                     read_stage,
                                              int                     handoff_stage,
                                              QKPipeline&             qk_pipeline,
                                              QKPipelineState         qk_pipe_release,
                                              SVPipeline&             sv_pipeline,
                                              SVPipelineState         sv_pipe_read,
                                              SVPipelineState         sv_pipe_release,
                                              __nv_bfloat16*          out,
                                              int                     valid_positions,
                                              int                     batch,
                                              int                     hv,
                                              int64_t                 out_batch_stride,
                                              int64_t                 out_token_stride,
                                              int64_t                 out_head_stride,
                                              int                     lane,
                                              int                     warp,
                                              int                     warp_group_thread,
                                              State*                  commit_state,
                                              int                     commit_length)
    {
        auto query_smem = make_tensor(
            make_smem_ptr(shared_storage.q.data()), SmemLayoutQK{});
        auto key_smem = make_tensor(
            make_smem_ptr(shared_storage.k.data()), SmemLayoutQK{});
        auto query = query_smem(_, _, read_stage);
        auto key   = key_smem(_, _, read_stage);

        // _p_: P[N,N] = Q[N,Dk] * K^T[Dk,N]
        const int P_tile = warp_group_thread / kWarpThreads;
        auto      P_mma = QKTiledMma{};
        auto      P_thread = P_mma.get_thread_slice(lane);
        auto      P_identity = make_identity_tensor(make_shape(MmaM{}, MmaN{}));
        auto      P_coordinates = P_thread.partition_C(P_identity);
        auto      P_accumulator = P_thread.make_fragment_C(P_coordinates);
        clear(P_accumulator);
        if (!kCommit && P_tile < kCapacityNTiles) {
            mma_qk_by_k_tile(query, key, P_accumulator, lane, P_tile);
        }

        if (!kCommit && warp == kOutputHeadCheckpointWarp && lane == 0) {
            shared_storage.output_head_checkpoints[read_stage] =
                shared_storage.output_heads[read_stage];
        }
        auto output_query = inner_partition(query, Tile<OutputN, HeadDim>{}, _0{});
        auto gate_exp_storage = make_tensor(
            make_smem_ptr(shared_storage.gate_exp.data()), SmemLayoutGateExp{});
        auto gate_exp = gate_exp_storage(_, handoff_stage);
        auto gate_exp_difference_storage = make_tensor(
            make_smem_ptr(shared_storage.gate_exp_difference.data()),
            SmemLayoutGateExpDifference{});
        auto gate_exp_difference = gate_exp_difference_storage(_, _, handoff_stage);

        // WG1 acquires state and value before converting the state operand.
        sv_pipeline.consumer_wait(sv_pipe_read);

        auto mma                = SSWgmmaTiledMma{};
        auto thread_mma         = mma.get_thread_slice(warp_group_thread);
        auto output_identity    = make_identity_tensor(make_shape(BlockDv{}, OutputN{}));
        auto output_coordinates = thread_mma.partition_C(output_identity);
        auto output             = thread_mma.make_fragment_C(output_coordinates);
        clear(output);

        auto state = make_tensor(
            make_smem_ptr(reinterpret_cast<State*>(shared_storage.state.data())),
            SmemLayoutState{});
        auto state_source = state(_, _, _, _, read_stage).compose(StateMmaLayout{});
        auto state_operand = make_tensor(
            make_smem_ptr(reinterpret_cast<Element*>(
                shared_storage.state.data() + StateTraits::kStateOperandOffsetBytes
                + read_stage * StateTraits::kStateOperandStageBytes)),
            SmemLayoutStateOperand{});
        auto query_fragment = thread_mma.make_fragment_B(
            thread_mma.partition_B(output_query));
        auto output_dv_halves = zipped_divide(
            output, make_tile(shape<0>(output), _1{}, shape<2>(output)));

        // _o_sq_: O0[Dv,N] = S[Dv,Dk] * Q^T[Dk,N]
        if constexpr (!kCommit) {
            warpgroup_fence_operand(output);
        }
        auto mma_state_dv_half = [&](auto state_dv_half) {
            auto state_tile = local_tile(
                state_operand, StateConversionTile{}, make_coord(state_dv_half, _0{}));
            auto state_fragment = thread_mma.make_fragment_A(
                thread_mma.partition_A(state_tile));
            auto output_half = output_dv_halves(
                make_coord(_, _, _), make_coord(_0{}, state_dv_half, _0{}));
            warpgroup_arrive();
            cute::gemm(mma, state_fragment, query_fragment, output_half);
            warpgroup_commit_batch();
        };
        // _state_conversion_: WG1 converts S'[Dv=0:64,Dk].
        ConvertAndReleaseStateDvHalf<StateDv0Ready>(StateTraits{},
                                                    state_source,
                                                    state_operand,
                                                    warp_group_thread,
                                                    _0{});
        if constexpr (!kCommit) {
            mma_state_dv_half(_0{});
        }
        AcquireStateDvHalf<StateDv1Ready>(StateTraits{});
        if constexpr (!kCommit) {
            mma_state_dv_half(_1{});
            warpgroup_wait<0>();
            warpgroup_fence_operand(output);
            // Verification releases its inputs after the state/query product.
            qk_pipeline.consumer_release(qk_pipe_release);
            sv_pipeline.consumer_release(sv_pipe_release);
        }

        // WG1 acquires WG2's W tile and WG0's AG tile before _agw_.
        WReady::sync(handoff_stage);
        AGReady::sync(handoff_stage);
        auto W_storage = make_tensor(
            make_smem_ptr(shared_storage.W_bf16.data()), SmemLayoutW{});
        auto AG_storage = make_tensor(
            make_smem_ptr(shared_storage.AG_bf16.data()), SmemLayoutAG{});
        auto W_smem = W_storage(_, _, handoff_stage);
        auto AG = AG_storage(_, _, handoff_stage);
        auto output_AG = inner_partition(AG, Tile<OutputN, MmaM>{}, _0{});
        auto W_fragment = thread_mma.make_fragment_A(
            thread_mma.partition_A(W_smem));
        auto AG_fragment = thread_mma.make_fragment_B(
            thread_mma.partition_B(output_AG));
        auto AGW = thread_mma.make_fragment_C(output_coordinates);
        clear(AGW);

        // _agw_: AGW[Dv,N] = W[Dv,N] * AG[N,N]
        cutlass::arch::fence_view_async_shared();
        warpgroup_fence_operand(AGW);
        warpgroup_arrive();
        cute::gemm(mma, W_fragment, AG_fragment, AGW);
        warpgroup_commit_batch();

        if constexpr (!kCommit) {
            // _o_scale_: O1[Dv,N] = scale*exp(gate) * O0[Dv,N]
            auto output_values = coalesce(output);
            auto output_coords = coalesce(output_coordinates);
            CUTE_UNROLL
            for (int index = 0; index < size(output_values); ++index) {
                const int row = int(get<1>(output_coords(index)));
                const float scale = kHeadScale * gate_exp(row);
                output_values(index) *= scale;
            }

            auto P_storage = make_tensor(
                make_smem_ptr(shared_storage.P_bf16.data()), SmemLayoutP{});
            auto P = P_storage;
            // _p_scale_: P'[row,column] = scale * P[row,column] * exp(prefix[row] - prefix[column])
            if (P_tile < kCapacityNTiles) {
                auto P_packed = make_fragment_like<Element>(P_accumulator);
                auto P_values = coalesce(
                    P_accumulator(TokenRows{}, _, _));
                auto P_packed_values = coalesce(
                    P_packed(TokenRows{}, _, _));
                auto P_coords = coalesce(
                    P_coordinates(TokenRows{}, _, _));
                auto P_value_pairs       = zipped_divide(P_values, Tile<_2>{});
                auto P_packed_pairs      = zipped_divide(P_packed_values, Tile<_2>{});
                auto P_coordinate_pairs  = zipped_divide(P_coords, Tile<_2>{});
                CUTE_UNROLL
                for (int pair = 0; pair < size<1>(P_value_pairs); ++pair) {
                    const auto coordinate = P_coordinate_pairs(_0{}, pair);
                    const int  row        = int(get<0>(coordinate));
                    const int  column     = P_tile * kMmaN + int(get<1>(coordinate));
                    auto gate_difference = make_tensor<float>(Shape<_2>{});
                    copy(Copy_Atom<UniversalCopy<uint64_t>, float>{},
                         coalesce(local_tile(gate_exp_difference,
                                             Tile<_1, _2>{},
                                             make_coord(row, column / _2::value))),
                         gate_difference);
                    CUTE_UNROLL
                    for (int element = 0; element < size<0>(P_value_pairs); ++element) {
                        const float P_value = kHeadScale * P_value_pairs(element, pair)
                                              * gate_difference(element);
                        P_packed_pairs(element, pair) = static_cast<Element>(P_value);
                    }
                }
                auto P_tiled_copy = make_tiled_copy_C(
                    Copy_Atom<PAccumulatorStoreOperation, Element>{}, P_mma);
                auto P_thread_copy = P_tiled_copy.get_thread_slice(lane);
                auto P_destination = P_thread_copy.partition_D(
                    as_position_independent_swizzle_tensor(P));
                auto P_source = P_thread_copy.retile_S(P_packed);
                copy(P_tiled_copy,
                     P_source(_, _0{}, _0{}),
                     P_destination(_, _0{}, P_tile));
            }
        }
        warpgroup_wait<0>();
        warpgroup_fence_operand(AGW);
        if constexpr (!kCommit) {
            // Every WG1 warp acquires the complete scaled P matrix.
            PReady::sync();
        }

        auto AGW_packed = make_fragment_like<Element>(AGW);
        CUTE_UNROLL
        for (int index = 0; index < size(AGW); ++index) {
            AGW_packed(index) = static_cast<Element>(AGW(index));
        }
        auto AGW_storage = make_tensor(
            make_smem_ptr(shared_storage.AGW_bf16.data()), SmemLayoutAGW{});
        auto AGW_smem = AGW_storage;
        auto output_AGW_smem = inner_partition(
            AGW_smem, Tile<BlockDv, OutputN>{}, _0{});
        auto AGW_tiled_copy = make_tiled_copy_C(
            WgmmaAccumulatorStoreAtom{}, mma);
        auto AGW_thread_copy = AGW_tiled_copy.get_thread_slice(warp_group_thread);
        auto AGW_destination = AGW_thread_copy.partition_D(
            as_position_independent_swizzle_tensor(output_AGW_smem));
        auto AGW_source = AGW_thread_copy.retile_S(AGW_packed);
        copy(AGW_tiled_copy, AGW_source, AGW_destination);

        if constexpr (kCommit) {
            // AGW[Dv,t] is the solved delta for transition t. The final state is
            // exp(prefix[L-1])*S0 + AGW * (exp(prefix[L-1]-prefix[t])*K[t,Dk]).
            // Both update operands have reduction extent 16, including K8's zero tail.
            auto commit_key = make_tensor(
                make_smem_ptr(shared_storage.q.data()), SmemLayoutCommitKey{});
            auto key_transpose = make_tensor(key.data(), select<1, 0>(key.layout()));
            auto key_copy = CommitKeyTiledCopy{};
            auto key_thread_copy = key_copy.get_thread_slice(warp_group_thread);
            auto key_source = key_thread_copy.partition_S(key_transpose);
            auto key_destination = key_thread_copy.partition_D(commit_key);
            auto key_coordinates = key_thread_copy.partition_D(
                make_identity_tensor(Shape<HeadDim, MmaM>{}));
            // Each lane owns eight adjacent Dk elements. The SW128 layout
            // permutes the eight 16-byte vectors within each 64-element box.
            CUTE_UNROLL
            for (int tile = 0; tile < size<2>(key_destination); ++tile) {
                const int position = int(get<1>(key_coordinates(_0{}, _0{}, tile)));
                auto values = make_fragment_like(key_destination(_, _0{}, tile));
                clear(values);
                if (position < commit_length) {
                    copy(CommitKeyCopyAtom{}, key_source(_, _0{}, tile), values);
                    const float decay = gate_exp_difference(commit_length - 1, position);
                    CUTE_UNROLL
                    for (int index = 0; index < size(values); ++index) {
                        values(index) = static_cast<Element>(float(values(index)) * decay);
                    }
                }
                copy(CommitKeyCopyAtom{}, values, key_destination(_, _0{}, tile));
            }
            // The final WG owns both scratch operands through the state WGMMA wait.
            PReady::sync();
            cutlass::arch::fence_view_async_shared();

            auto update_mma = StateUpdateTiledMma{};
            auto update_thread = update_mma.get_thread_slice(warp_group_thread);
            auto state_destination = make_tensor(
                make_gmem_ptr(commit_state), Layout<Shape<BlockDv, HeadDim>, Stride<_1, BlockDv>>{});
            const float state_decay = gate_exp(commit_length - 1);
            // One 64x64 tile keeps only 32 FP32 accumulator values live per lane.
            CUTLASS_PRAGMA_NO_UNROLL
            for (int tile = 0; tile < size(StateUpdateTiles{}); ++tile) {
                const auto coordinate = StateUpdateTiles{}.get_hier_coord(tile);
                auto state_tile = local_tile(state_source, StateUpdateTile{}, coordinate);
                auto state_input = update_thread.partition_C(state_tile);
                auto accumulator = update_thread.make_fragment_C(state_input);
                CUTE_UNROLL
                for (int index = 0; index < size(accumulator); ++index) {
                    // Keep the original FP32 state for the residual; round BF16 only at the final store.
                    accumulator(index) = state_decay * float(state_input(index));
                }
                auto delta_tile = local_tile(
                    AGW_smem, Tile<HalfDv, MmaM>{}, make_coord(get<0>(coordinate), _0{}));
                auto key_tile = local_tile(
                    commit_key, Tile<HalfDv, MmaM>{}, make_coord(get<1>(coordinate), _0{}));
                auto delta_fragment = update_thread.make_fragment_A(update_thread.partition_A(delta_tile));
                auto update_key = update_thread.make_fragment_B(update_thread.partition_B(key_tile));
                warpgroup_fence_operand(accumulator);
                warpgroup_arrive();
                cute::gemm(update_mma, delta_fragment, update_key, accumulator);
                warpgroup_commit_batch();
                warpgroup_wait<0>();
                warpgroup_fence_operand(accumulator);

                auto destination_tile = local_tile(
                    state_destination, StateUpdateTile{}, coordinate);
                auto destination = update_thread.partition_C(destination_tile);
                CUTE_UNROLL
                for (int index = 0; index < size(accumulator); ++index) {
                    destination(index) = static_cast<State>(accumulator(index));
                }
            }
            // Commit retains K and the unconverted entry state until the final update completes.
            qk_pipeline.consumer_release(qk_pipe_release);
            sv_pipeline.consumer_release(sv_pipe_release);
        }

        if constexpr (!kCommit) {
            auto P = make_tensor(make_smem_ptr(shared_storage.P_bf16.data()), SmemLayoutP{});
            auto output_P = inner_partition(P, Tile<OutputN, MmaM>{}, _0{});
            auto AGW_fragment = thread_mma.make_fragment_A(
                thread_mma.partition_A(AGW_smem));
            auto P_fragment = thread_mma.make_fragment_B(
                thread_mma.partition_B(output_P));

            // _o_final_: O2[Dv,N] = O1[Dv,N] + AGW[Dv,N] * P[N,N]
            cutlass::arch::fence_view_async_shared();
            warpgroup_fence_operand(output);
            warpgroup_arrive();
            cute::gemm(mma, AGW_fragment, P_fragment, output);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(output);

            auto output_tensor = make_tensor(
                make_gmem_ptr(reinterpret_cast<Element*>(out)),
                make_layout(make_shape(batch, hv, BlockDv{}, OutputN{}),
                            make_stride(out_batch_stride,
                                        out_head_stride,
                                        _1{},
                                        out_token_stride)));
            const auto output_head = shared_storage.output_head_checkpoints[read_stage];
            auto output_tile = output_tensor(output_head.request, output_head.value_head, _, _);
            StoreOutput(OutputStoreTiledCopy{},
                        output,
                        output_coordinates,
                        output_tile,
                        OutputStoreVectorElements{},
                        _1{},
                        valid_positions,
                        warp_group_thread);
        }
    }

    static CUTE_DEVICE void ProcessUpdateHead(SharedStorage&          shared_storage,
                                              int                     read_stage,
                                              int                     handoff_stage,
                                              QKPipeline&             qk_pipeline,
                                              QKPipelineState         qk_pipe_release,
                                              SVPipeline&             sv_pipeline,
                                              SVPipelineState         sv_pipe_read,
                                              SVPipelineState         sv_pipe_release,
                                              int                     lane,
                                              int                     warp,
                                              int                     warp_group_thread,
                                              int                     handoff_phase)
    {
        const int commit_length = kCommit ? shared_storage.output_heads[read_stage].commit_length : Capacity;
        auto key_smem = make_tensor(
            make_smem_ptr(shared_storage.k.data()), SmemLayoutQK{});
        auto key = key_smem(_, _, read_stage);
        auto output_key = inner_partition(key, Tile<OutputN, HeadDim>{}, _0{});
        auto gate_smem = make_tensor(
            make_smem_ptr(shared_storage.gate.data()), SmemLayoutGate{});
        auto gate       = gate_smem(_, _0{}, read_stage);
        auto gate_exp_storage = make_tensor(
            make_smem_ptr(shared_storage.gate_exp.data()), SmemLayoutGateExp{});
        auto gate_exp = gate_exp_storage(_, handoff_stage);
        auto gate_exp_difference_storage = make_tensor(
            make_smem_ptr(shared_storage.gate_exp_difference.data()),
            SmemLayoutGateExpDifference{});
        auto gate_exp_difference = gate_exp_difference_storage(_, _, handoff_stage);

        // _prefix_: Prefix sum of gate.
        auto gate_prefix = make_tensor(gate.data(), Layout<Int<Capacity>>{});
        if (warp == kPrefixWarp) {
            float prefix = lane < Capacity ? gate_prefix(lane) : 0.0f;
            CUTE_UNROLL
            for (int delta = 1; delta < Capacity; delta <<= 1) {
                const float other = __shfl_up_sync(kWarpMask, prefix, delta);
                if (lane >= delta) {
                    prefix += other;
                }
            }
            if (lane < Capacity) {
                gate_prefix(lane) = prefix;
                gate_exp(lane) = FastExp(prefix);
            }
        }

        // WG2 acquires the stored prefix vector before cooperatively materializing its differences.
        GatePrefixReady::sync();
        const auto prefix_thread_coordinate =
            PrefixWarpGroupThreadLayout{}.get_hier_coord(warp_group_thread);
        const int prefix_warp = int(get<0>(prefix_thread_coordinate));
        if (lane < size(PrefixOutputLaneLayout{})) {
            const auto prefix_output_coordinate =
                PrefixOutputLaneLayout{}.get_hier_coord(lane);
            const int row = int(PrefixOutputRowLayout{}(
                prefix_warp, get<0>(prefix_output_coordinate)));
            const int column_vector = int(get<1>(prefix_output_coordinate));
            auto prefix_columns = make_tensor<float>(Shape<_4>{});
            copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                 local_tile(gate_prefix, Tile<_4>{}, make_coord(column_vector)),
                 prefix_columns);
            const float prefix_row = gate_prefix(row);
            auto gate_exp_difference_vector = make_tensor<float>(Shape<_4>{});
            CUTE_UNROLL
            for (int element = 0; element < size(gate_exp_difference_vector); ++element) {
                const int column = int(PrefixColumnVectorLayout{}(column_vector, element));
                gate_exp_difference_vector(element) =
                    column <= row ? FastExp(prefix_row - prefix_columns(element)) : 0.0f;
            }
            auto gate_exp_difference_destination = coalesce(local_tile(
                gate_exp_difference,
                Tile<_1, _4>{},
                make_coord(row, column_vector)));
            copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                 gate_exp_difference_vector,
                 gate_exp_difference_destination);
        }
        // Every WG2 warp releases its completed columns to the prefix consumers.
        arrive_barrier(shared_storage.prefix_ready[handoff_stage]);

        // WG2 acquires state and value after completing the gate-only prefix phase.
        sv_pipeline.consumer_wait(sv_pipe_read);

        auto state_mma          = SSWgmmaTiledMma{};
        auto state_thread_mma   = state_mma.get_thread_slice(warp_group_thread);
        auto output_identity    = make_identity_tensor(make_shape(BlockDv{}, OutputN{}));
        auto output_coordinates = state_thread_mma.partition_C(output_identity);
        auto U                  = state_thread_mma.make_fragment_C(output_coordinates);
        clear(U);

        auto state = make_tensor(
            make_smem_ptr(reinterpret_cast<State*>(shared_storage.state.data())),
            SmemLayoutState{});
        auto state_source = state(_, _, _, _, read_stage).compose(StateMmaLayout{});
        auto state_operand = make_tensor(
            make_smem_ptr(reinterpret_cast<Element*>(
                shared_storage.state.data() + StateTraits::kStateOperandOffsetBytes
                + read_stage * StateTraits::kStateOperandStageBytes)),
            SmemLayoutStateOperand{});
        auto key_fragment = state_thread_mma.make_fragment_B(
            state_thread_mma.partition_B(output_key));
        auto U_dv_halves = zipped_divide(
            U, make_tile(shape<0>(U), _1{}, shape<2>(U)));

        // _u_: U[Dv,N] = S[Dv,Dk] * K^T[Dk,N]
        warpgroup_fence_operand(U);
        auto mma_state_dv_half = [&](auto state_dv_half) {
            auto state_tile = local_tile(
                state_operand, StateConversionTile{}, make_coord(state_dv_half, _0{}));
            auto state_fragment = state_thread_mma.make_fragment_A(
                state_thread_mma.partition_A(state_tile));
            auto U_half = U_dv_halves(
                make_coord(_, _, _), make_coord(_0{}, state_dv_half, _0{}));
            warpgroup_arrive();
            cute::gemm(state_mma, state_fragment, key_fragment, U_half);
            warpgroup_commit_batch();
        };
        // _state_conversion_: WG2 converts S'[Dv=64:128,Dk].
        ConvertAndReleaseStateDvHalf<StateDv1Ready>(StateTraits{},
                                                    state_source,
                                                    state_operand,
                                                    warp_group_thread,
                                                    _1{});
        mma_state_dv_half(_1{});
        AcquireStateDvHalf<StateDv0Ready>(StateTraits{});
        mma_state_dv_half(_0{});
        warpgroup_wait<0>();
        warpgroup_fence_operand(U);

        // WG2 releases Q/K after the final key WGMMA read.
        qk_pipeline.consumer_release(qk_pipe_release);

        // WG2 independently acquires the completed prefix before _w_.
        wait_barrier(shared_storage.prefix_ready[handoff_stage], handoff_phase);

        // _w_: W[Dv,N] = V^T[Dv,N] - exp(g)*U[Dv,N]
        auto W = make_fragment_like<Element>(U);
        auto W_values      = coalesce(W);
        auto U_values      = coalesce(U);
        auto output_coords = coalesce(output_coordinates);
        auto input_values = make_tensor(
            make_smem_ptr(shared_storage.value.data()), SmemLayoutValue{});
        CUTE_UNROLL
        for (int index = 0; index < size(W_values); ++index) {
            const auto coordinate = output_coords(index);
            const int  dv         = int(get<0>(coordinate));
            const int  row        = int(get<1>(coordinate));
            const float gate_scale = gate_exp(row);
            const auto dv_coordinate = ValueDvCoordinateLayout{}.get_hier_coord(dv);
            W_values(index) = kCommit && row >= commit_length ? Element{} : static_cast<Element>(
                float(input_values(get<0>(dv_coordinate),
                                   row,
                                   get<1>(dv_coordinate),
                                   read_stage))
                - gate_scale * U_values(index));
        }
        // WG2 releases state/value after its final value read.
        sv_pipeline.consumer_release(sv_pipe_release);

        auto W_storage = make_tensor(
            make_smem_ptr(shared_storage.W_bf16.data()), SmemLayoutW{});
        auto W_smem = W_storage(_, _, handoff_stage);
        auto output_W_smem = inner_partition(W_smem, Tile<BlockDv, OutputN>{}, _0{});
        auto W_tiled_copy = make_tiled_copy_C(
            WgmmaAccumulatorStoreAtom{}, state_thread_mma);
        auto W_thread_copy = W_tiled_copy.get_thread_slice(warp_group_thread);
        auto W_destination = W_thread_copy.partition_D(
            as_position_independent_swizzle_tensor(output_W_smem));
        auto W_source = W_thread_copy.retile_S(W);
        copy(W_tiled_copy, W_source, W_destination);
        // WG2 releases its complete W shared-memory tile to WG1.
        WReady::arrive(handoff_stage);
    }

public:
    template<class TmaQ, class TmaK, class TmaV, class TmaState>
    struct DeviceOperator {
        using SharedStorage = typename Policy::SharedStorage;

        static constexpr int MaxThreadsPerBlock         = Policy::kThreads;
        static constexpr int MinBlocksPerMultiprocessor = 1;
        static constexpr int SharedMemoryAlignment      = alignof(SharedStorage);

        struct KernelParams {
            const float*         g;
            const float*         beta;
            const TmaDescriptor* state_tma_descs;
            __nv_bfloat16*       out;
            int                  valid_positions;
            int                  batch;
            int                  hq;
            int                  hv;
            int                  value_heads_per_query_head;
            int                  num_head_groups;
            int                  heads_per_block;
            int64_t              g_batch_stride;
            int64_t              g_token_stride;
            int64_t              beta_batch_stride;
            int64_t              beta_token_stride;
            int64_t              out_batch_stride;
            int64_t              out_token_stride;
            int64_t              out_head_stride;
            int                  state_layer;
            void* const*         state_ptrs;
            const int*           commit_lengths;
            const bool*          finished;
            int64_t              state_request_stride;
            int64_t              state_group_stride;
        };

        struct Params {
            TmaQ         tma_q;
            TmaK         tma_k;
            TmaV         tma_v;
            TmaState     tma_state;
            KernelParams kernel;
        };

        CUTE_DEVICE void operator()(const Params& parameters_, SharedStorage& storage)
        {
            const int thread = static_cast<int>(threadIdx.x);
            const int lane   = thread % Policy::kWarpThreads;
            const int warp   = thread / Policy::kWarpThreads;
            const int warp_group        = thread / cutlass::NumThreadsPerWarpGroup;
            const int warp_group_thread = thread % cutlass::NumThreadsPerWarpGroup;
            const bool is_producer_group =
                warp_group == static_cast<int>(WarpGroupRole::Producer);
            const bool is_producer_warp = is_producer_group && warp == kProducerWarp;
            const bool is_sv_consumer =
                warp_group == static_cast<int>(WarpGroupRole::Output)
                || warp_group == static_cast<int>(WarpGroupRole::Update);

            const auto& parameters = parameters_.kernel;
            typename QKPipeline::Params qk_pipeline_params;
            qk_pipeline_params.transaction_bytes = kQKTransactionBytes;
            qk_pipeline_params.role = is_producer_warp ? QKPipeline::ThreadCategory::Producer :
                                      is_producer_group ? QKPipeline::ThreadCategory::NonParticipant :
                                                          QKPipeline::ThreadCategory::Consumer;
            qk_pipeline_params.is_leader         = thread == kProducerThread;
            qk_pipeline_params.num_consumers     = kComputeThreads;
            qk_pipeline_params.num_producers     = 1 + kWarpThreads;
            qk_pipeline_params.initializing_warp = 0;
            QKPipeline qk_pipeline(
                storage.qk_pipeline,
                qk_pipeline_params,
                PipelineShape{},
                bool_constant<true>{},
                bool_constant<true>{});

            typename SVPipeline::Params sv_pipeline_params;
            sv_pipeline_params.transaction_bytes = kSVTransactionBytes;
            sv_pipeline_params.role = is_producer_warp ? SVPipeline::ThreadCategory::Producer :
                                      is_sv_consumer ? SVPipeline::ThreadCategory::Consumer :
                                                       SVPipeline::ThreadCategory::NonParticipant;
            sv_pipeline_params.is_leader         = thread == kProducerThread;
            sv_pipeline_params.num_consumers     = kSVConsumerThreads;
            sv_pipeline_params.num_producers     = 1;
            sv_pipeline_params.initializing_warp = 0;
            SVPipeline sv_pipeline(
                storage.sv_pipeline,
                sv_pipeline_params,
                PipelineShape{},
                bool_constant<true>{},
                bool_constant<true>{});

            if (thread == 0) {
                CUTE_UNROLL
                for (int stage = 0; stage < kStages; ++stage) {
                    storage.output_head_ready[stage].init(1);
                }
                CUTE_UNROLL
                for (int stage = 0; stage < kHandoffStages; ++stage) {
                    initialize_barrier(storage.prefix_ready[stage], cutlass::NumThreadsPerWarpGroup);
                }
                cutlass::arch::fence_barrier_init();
            }

            // All producer and consumer threads acquire the initialized barrier state.
            __syncthreads();

            if (is_producer_group) {
                cutlass::arch::warpgroup_reg_dealloc<kProducerRegisters>();
                if (is_producer_warp) {
                    const auto& tma_q     = parameters_.tma_q;
                    const auto& tma_k     = parameters_.tma_k;
                    const auto& tma_v     = parameters_.tma_v;
                    const auto& tma_state = parameters_.tma_state;
                    // TMA and cp.async zero-fill runtime rows [valid_positions, Capacity);
                    // Q/K/V shared-memory stages contain no fixed [Capacity, MmaM) tail.
                    auto query_tensor = tma_q.get_tma_tensor(
                        make_shape(QKVector{},
                                   QKVectors{},
                                   parameters.valid_positions,
                                   parameters.hq,
                                   parameters.batch));
                    auto key_tensor = tma_k.get_tma_tensor(
                        make_shape(QKVector{},
                                   QKVectors{},
                                   parameters.valid_positions,
                                   parameters.hq,
                                   parameters.batch));
                    auto value_tensor = tma_v.get_tma_tensor(
                        make_shape(BlockDv{},
                                   parameters.valid_positions,
                                   make_shape(parameters.hv, parameters.batch)));
                    auto state_tensor = tma_state.get_tma_tensor(
                        make_shape(WarpThreads{},
                                   MmaM{},
                                   StateDvBoxes{},
                                   StateDkPanels{},
                                   parameters.heads_per_block * (parameters.state_layer + 1)));
                    auto state_descriptors = make_tensor(
                        make_gmem_ptr(parameters.state_tma_descs),
                        make_layout(make_shape(parameters.num_head_groups, parameters.batch),
                                    make_stride(_1{}, parameters.num_head_groups)));
                    auto state_head_layout = make_layout(
                        make_shape(parameters.heads_per_block, parameters.state_layer + 1));

                    auto query_smem = make_tensor(
                        make_smem_ptr(storage.q.data()),
                        SmemLayoutQKTma{});
                    auto key_smem = make_tensor(
                        make_smem_ptr(storage.k.data()),
                        SmemLayoutQKTma{});
                    auto value_smem = make_tensor(
                        make_smem_ptr(storage.value.data()),
                        SmemLayoutValue{});
                    auto state_smem = make_tensor(
                        make_smem_ptr(reinterpret_cast<State*>(storage.state.data())),
                        SmemLayoutState{});

                    auto query_tma = tma_q.get_slice(_0{});
                    auto key_tma   = tma_k.get_slice(_0{});
                    auto value_tma = tma_v.get_slice(_0{});
                    auto state_tma = tma_state.get_slice(_0{});

                    auto tQgQ = query_tma.partition_S(query_tensor);
                    auto tQsQ = query_tma.partition_D(query_smem);
                    auto tKgK = key_tma.partition_S(key_tensor);
                    auto tKsK = key_tma.partition_D(key_smem);
                    auto tVgV = value_tma.partition_S(value_tensor);
                    auto tVsV = value_tma.partition_D(value_smem);
                    auto tSgS = state_tma.partition_S(state_tensor);
                    auto tSsS = state_tma.partition_D(state_smem);

                    auto gate = make_tensor(
                        make_gmem_ptr(parameters.g),
                        make_layout(make_shape(parameters.batch, MmaM{}, parameters.hv),
                                    make_stride(parameters.g_batch_stride,
                                                parameters.g_token_stride,
                                                _1{})));
                    auto beta = make_tensor(
                        make_gmem_ptr(parameters.beta),
                        make_layout(make_shape(parameters.batch, MmaM{}, parameters.hv),
                                    make_stride(parameters.beta_batch_stride,
                                                parameters.beta_token_stride,
                                                _1{})));
                    auto gate_smem = make_tensor(
                        make_smem_ptr(storage.gate.data()), SmemLayoutGate{});
                    auto beta_smem = make_tensor(
                        make_smem_ptr(storage.beta.data()), SmemLayoutBeta{});

                    auto qk_pipe_write = cutlass::make_producer_start_state<QKPipeline>();
                    auto sv_pipe_write = cutlass::make_producer_start_state<SVPipeline>();
                    const int lane_predicate = elect_one_sync();
                    auto prefetch_head = [&](const HeadCoordinate& head) {
                        int commit_length = parameters.valid_positions;
                        if constexpr (kCommit) {
                            commit_length = parameters.commit_lengths[head.request];
                            if (commit_length == 0 || (parameters.finished && parameters.finished[head.request])) {
                                return;
                            }
                        }
                        // Every producer lane acquires the reusable Q/K stage before writing it.
                        qk_pipeline.producer_acquire(qk_pipe_write);
                        using QKBarrier = typename QKPipeline::ProducerBarrierType;
                        QKBarrier* qk_barrier = qk_pipeline.producer_get_barrier(qk_pipe_write);
                        const int write_stage = qk_pipe_write.index();
                        if (lane_predicate) {
                            storage.output_heads[write_stage] = {head.request, head.value_head, commit_length};
                            // The producer publishes the decoded output coordinate for this stage.
                            storage.output_head_ready[write_stage].arrive();
                            if constexpr (!kCommit) {
                                copy(tma_q.with(*qk_barrier),
                                     tQgQ(_, _, _, _0{}, head.query_head, head.request),
                                     tQsQ(_, _, _, _0{}, _0{}, _0{}, write_stage));
                            }
                            copy(tma_k.with(*qk_barrier),
                                 tKgK(_, _, _, _0{}, head.query_head, head.request),
                                 tKsK(_, _, _, _0{}, _0{}, _0{}, write_stage));
                        }
                        auto gate_source = gate(head.request, _, head.value_head);
                        auto beta_source = beta(head.request, _, head.value_head);
                        const bool valid = lane < commit_length;
                        const int source_token = valid ? lane : 0;
                        const auto gate_lane = GateLaneLayout{}.get_hier_coord(lane);
                        SM80_CP_ASYNC_CACHEALWAYS_ZFILL<float>::copy(
                            gate_source(source_token),
                            gate_smem(get<0>(gate_lane), get<1>(gate_lane), write_stage),
                            valid);
                        SM80_CP_ASYNC_CACHEALWAYS_ZFILL<float>::copy(
                            beta_source(source_token),
                            beta_smem(get<0>(gate_lane), get<1>(gate_lane), write_stage),
                            valid);
                        // Every producer lane commits its gate/beta cp.async completion to the Q/K mbarrier.
                        qk_pipeline.producer_commit(
                            qk_pipe_write, cutlass::arch::cpasync_barrier_arrive_noinc);
                        ++qk_pipe_write;

                        // Every producer lane acquires the reusable state/value stage before writing it.
                        sv_pipeline.producer_acquire(sv_pipe_write);
                        using SVBarrier = typename SVPipeline::ProducerBarrierType;
                        SVBarrier* sv_barrier = sv_pipeline.producer_get_barrier(sv_pipe_write);
                        if (lane_predicate) {
                            CUTE_UNROLL
                            for (int half = 0; half < kValueHalves; ++half) {
                                copy(tma_v.with(*sv_barrier),
                                     tVgV(_, half, _0{}, make_coord(head.value_head, head.request)),
                                     tVsV(_, _0{}, _0{}, half, sv_pipe_write.index()));
                            }
                            const TmaDescriptor* state_descriptor =
                                &state_descriptors(head.head_group, head.request);
                            CUTE_UNROLL
                            for (int state_copy = 0; state_copy < size<4>(tSgS); ++state_copy) {
                                copy(tma_state.with(state_descriptor, *sv_barrier),
                                     tSgS(_, _, _, _, state_copy,
                                          state_head_layout(head.local_head, parameters.state_layer)),
                                     tSsS(_, _, _, _, state_copy, sv_pipe_write.index()));
                            }
                        }
                        ++sv_pipe_write;
                    };

                    const int persistent_ctas = static_cast<int>(gridDim.x);
                    const int heads_for_cta = ceil_div(
                        parameters.hv * parameters.batch - static_cast<int>(blockIdx.x),
                        persistent_ctas);
                    auto cta_head_layout = make_layout(make_shape(persistent_ctas, heads_for_cta));

                    CUTLASS_PRAGMA_NO_UNROLL
                    for (int prefetched_head_in_cta = 0;
                         prefetched_head_in_cta < heads_for_cta;
                        ++prefetched_head_in_cta) {
                        const int prefetched_flattened_head =
                            int(cta_head_layout(int(blockIdx.x), prefetched_head_in_cta));
                        auto prefetched_head = DecodeHead(
                            prefetched_flattened_head,
                            parameters.batch,
                            parameters.hq,
                            parameters.hv,
                            parameters.value_heads_per_query_head,
                            parameters.num_head_groups,
                            parameters.heads_per_block);
                        prefetch_head(prefetched_head);
                    }
                    // Every producer lane acquires both reusable input stages before publishing termination.
                    qk_pipeline.producer_acquire(qk_pipe_write);
                    sv_pipeline.producer_acquire(sv_pipe_write);
                    if (lane_predicate) {
                        const int write_stage = qk_pipe_write.index();
                        storage.output_heads[write_stage] = {kInvalidRequest, 0, 0};
                        // The producer publishes the terminal coordinate after acquiring its reusable stage.
                        storage.output_head_ready[write_stage].arrive();
                        // The elected producer drains both input pipelines after the final head releases them.
                        qk_pipeline.producer_tail(qk_pipe_write);
                        sv_pipeline.producer_tail(sv_pipe_write);
                    }
                }
            }
            else if (warp_group == static_cast<int>(WarpGroupRole::GramPAG)) {
                cutlass::arch::warpgroup_reg_alloc<kGramPAGRegisters>();
                auto AG_storage = make_tensor(
                    make_smem_ptr(storage.AG_bf16.data()), SmemLayoutAG{});
                // K8 _agw_ reads AG[N,K] with WGMMA K=16, while each head writes only
                // [0:Capacity,0:Capacity]. Initialize the K tail in every physical stage
                // once; the loop is zero-trip for K16.
                CUTE_UNROLL
                for (int tail_tile = 0;
                     tail_tile < (kMmaM - Capacity) / kMmaN;
                     ++tail_tile) {
                    if (warp_group_thread < size(WgmmaBOperandTailCoordinates{})) {
                        const auto coordinate = WgmmaBOperandTailCoordinates{}.get_hier_coord(
                            warp_group_thread);
                        const int row = int(get<0>(coordinate));
                        const int column = Capacity + tail_tile * kMmaN
                                           + int(get<1>(coordinate));
                        CUTE_UNROLL
                        for (int stage = 0; stage < kHandoffStages; ++stage) {
                            AG_storage(row, column, stage) = Element{};
                        }
                    }
                }
                QKPipelineState qk_pipe_read;
                QKPipelineState qk_pipe_release;
                cutlass::PipelineState<kHandoffStages> handoff_stage;
                while (true) {
                    const int read_stage = qk_pipe_read.index();
                    // Every compute thread acquires the decoded coordinate before testing the terminal stage.
                    storage.output_head_ready[read_stage].wait(qk_pipe_read.phase());
                    if (storage.output_heads[read_stage].request == kInvalidRequest) {
                        break;
                    }
                    // WG0 acquires Q/K and gate/beta before Gram.
                    qk_pipeline.consumer_wait(qk_pipe_read);
                    ProcessGramPAGHead(storage,
                                      read_stage,
                                      handoff_stage.index(),
                                      qk_pipeline,
                                      qk_pipe_release,
                                      lane,
                                      warp,
                                      handoff_stage.phase());
                    ++qk_pipe_read;
                    ++qk_pipe_release;
                    ++handoff_stage;
                }
            }
            else if (warp_group == static_cast<int>(WarpGroupRole::Output)) {
                cutlass::arch::warpgroup_reg_alloc<kOutputRegisters>();
                auto P_storage = make_tensor(
                    make_smem_ptr(storage.P_bf16.data()), SmemLayoutP{});
                auto AGW_storage = make_tensor(
                    make_smem_ptr(storage.AGW_bf16.data()), SmemLayoutAGW{});
                // K8 _o_final_ reads P[N,K] with WGMMA K=16, while each head writes
                // only [0:Capacity,0:Capacity]. Initialize the persistent K tail once;
                // the loop is zero-trip for K16.
                CUTE_UNROLL
                for (int tail_tile = 0;
                     tail_tile < (kMmaM - Capacity) / kMmaN;
                     ++tail_tile) {
                    if (warp_group_thread < size(WgmmaBOperandTailCoordinates{})) {
                        const auto coordinate = WgmmaBOperandTailCoordinates{}.get_hier_coord(
                            warp_group_thread);
                        const int row = int(get<0>(coordinate));
                        const int column = Capacity + tail_tile * kMmaN
                                           + int(get<1>(coordinate));
                        if constexpr (!kCommit) {
                            P_storage(row, column) = Element{};
                        }
                    }
                }
                // K8 _o_final_ reads AGW[Dv,K] with WGMMA K=16, while each head
                // writes only columns [0,Capacity). Initialize the local K tail once;
                // it persists across heads and this loop is zero-trip for K16.
                CUTE_UNROLL
                for (int column = Capacity; column < kMmaM; ++column) {
                    AGW_storage(warp_group_thread, column) = Element{};
                }
                QKPipelineState qk_pipe_read;
                QKPipelineState qk_pipe_release;
                SVPipelineState sv_pipe_read;
                SVPipelineState sv_pipe_release;
                cutlass::PipelineState<kHandoffStages> handoff_stage;
                while (true) {
                    const int read_stage = qk_pipe_read.index();
                    // Every compute thread acquires the decoded coordinate before testing the terminal stage.
                    storage.output_head_ready[read_stage].wait(qk_pipe_read.phase());
                    if (storage.output_heads[read_stage].request == kInvalidRequest) {
                        break;
                    }
                    // WG1 acquires Q/K and gate/beta before P.
                    qk_pipeline.consumer_wait(qk_pipe_read);
                    State* commit_state = nullptr;
                    int commit_length = 0;
                    if constexpr (kCommit) {
                        const auto head = storage.output_heads[read_stage];
                        commit_length = head.commit_length;
                        auto state_ptrs = make_tensor(
                            make_gmem_ptr(parameters.state_ptrs),
                            make_layout(make_shape(parameters.num_head_groups, parameters.batch),
                                        make_stride(parameters.state_group_stride, parameters.state_request_stride)));
                        auto state_head_layout = make_layout(
                            make_shape(parameters.heads_per_block, parameters.num_head_groups));
                        const auto state_head = state_head_layout.get_hier_coord(head.value_head);
                        auto state_heads = make_tensor(
                            make_gmem_ptr(static_cast<State*>(state_ptrs(get<1>(state_head), head.request))),
                            GmemLayoutState{make_shape(BlockDv{}, HeadDim{},
                                                       parameters.heads_per_block * (parameters.state_layer + 1))});
                        auto layer_head_layout = make_layout(
                            make_shape(parameters.heads_per_block, parameters.state_layer + 1));
                        commit_state = &state_heads(_0{}, _0{},
                            layer_head_layout(get<0>(state_head), parameters.state_layer));
                    }
                    ProcessFinalHead(storage,
                                      read_stage,
                                      handoff_stage.index(),
                                      qk_pipeline,
                                      qk_pipe_release,
                                      sv_pipeline,
                                      sv_pipe_read,
                                      sv_pipe_release,
                                      parameters.out,
                                      parameters.valid_positions,
                                      parameters.batch,
                                      parameters.hv,
                                      parameters.out_batch_stride,
                                      parameters.out_token_stride,
                                      parameters.out_head_stride,
                                      lane,
                                      warp,
                                      warp_group_thread,
                                      commit_state,
                                      commit_length);
                    ++qk_pipe_read;
                    ++qk_pipe_release;
                    ++sv_pipe_read;
                    ++sv_pipe_release;
                    ++handoff_stage;
                }
            }
            else {
                cutlass::arch::warpgroup_reg_alloc<kUpdateRegisters>();
                auto W_storage = make_tensor(
                    make_smem_ptr(storage.W_bf16.data()), SmemLayoutW{});
                // K8 _agw_ reads W[Dv,K] with WGMMA K=16, while each head writes
                // only columns [0,Capacity). Initialize every stage's K tail once;
                // it persists across heads and this loop is zero-trip for K16.
                CUTE_UNROLL
                for (int stage = 0; stage < kHandoffStages; ++stage) {
                    CUTE_UNROLL
                    for (int column = Capacity; column < kMmaM; ++column) {
                        W_storage(warp_group_thread, column, stage) = Element{};
                    }
                }
                QKPipelineState qk_pipe_read;
                QKPipelineState qk_pipe_release;
                SVPipelineState sv_pipe_read;
                SVPipelineState sv_pipe_release;
                cutlass::PipelineState<kHandoffStages> handoff_stage;
                while (true) {
                    const int read_stage = qk_pipe_read.index();
                    // Every compute thread acquires the decoded coordinate before testing the terminal stage.
                    storage.output_head_ready[read_stage].wait(qk_pipe_read.phase());
                    if (storage.output_heads[read_stage].request == kInvalidRequest) {
                        break;
                    }
                    // WG2 acquires Q/K and gate/beta before prefix.
                    qk_pipeline.consumer_wait(qk_pipe_read);
                    ProcessUpdateHead(storage,
                                      read_stage,
                                      handoff_stage.index(),
                                      qk_pipeline,
                                      qk_pipe_release,
                                      sv_pipeline,
                                      sv_pipe_read,
                                      sv_pipe_release,
                                      lane,
                                      warp,
                                      warp_group_thread,
                                      handoff_stage.phase());
                    ++qk_pipe_read;
                    ++qk_pipe_release;
                    ++sv_pipe_read;
                    ++sv_pipe_release;
                    ++handoff_stage;
                }
            }
        }
    };

public:
    static void RegisterStateVariants(Collector& collector)
    {
        collector.add<Sm90GdrVerifyCommitKernel<Capacity, kFloat32>>();
        collector.add<Sm90GdrVerifyCommitKernel<Capacity, kBfloat16>>();
        collector.add<Sm90GdrVerifyCommitKernel<Capacity, kFloat32, GdrMode::kCommit>>();
        collector.add<Sm90GdrVerifyCommitKernel<Capacity, kBfloat16, GdrMode::kCommit>>();
    }

    const GdrKernelSpec& spec() const noexcept override
    {
        return kSpec;
    }

    const char* name() const noexcept override
    {
        return kName;
    }

    bool Match(const Operation& operation, const PlanningContext& context) const override
    {
        if (operation.mode == Mode
            && (context.token_slots < kMinimumTokens || context.token_slots > spec().chunk_size)) {
            return false;
        }
        Operation selected = operation;
        if (selected.mode == Mode && selected.chunk_size == kAutoGdrChunkSize) {
            selected.chunk_size = context.token_slots <= kMmaN ? kMmaN : kMmaM;
        }
        return detail::MatchesGdrSpec(spec(), selected, context);
    }

    bool Plan(const Operation& operation, const PlanningContext& context, delta_rule::Plan* plan) const override
    {
        return detail::PlanSm90Operation(spec(), operation, context, plan);
    }

    void PrepareState(const core::Tensor&     state_ptrs,
                      core::Tensor&           state_tma_descs,
                      int                     layer_groups,
                      int                     layers_per_block,
                      const delta_rule::Plan& plan,
                      cudaStream_t            stream) const override
    {
        auto* state_prototype_address = reinterpret_cast<StateT*>(
            reinterpret_cast<uintptr_t>(state_ptrs.raw_data())
            & ~uintptr_t(kTmaGlobalAddressAlignment - 1));
        auto state = make_tensor(
            make_gmem_ptr(state_prototype_address),
            GmemLayoutState{make_shape(BlockDv{},
                                       HeadDim{},
                                       plan.problem.heads_per_block * layers_per_block)});
        auto g_state = make_tensor(
            state.data(),
            select<0, 1, 3, 4, 5>(
                flatten(zipped_divide(state.layout(), Tile<WarpThreads, MmaM, _1>{}))));
        auto tma_state =
            make_tma_copy(kTmaLoad, g_state, SmemLayoutStateTmaTile{}, StateTmaTile{}, _1{});
        detail::PrepareSm90StateTmaDescriptors(state_ptrs,
                                               state_tma_descs,
                                               layer_groups,
                                               plan.problem.batch,
                                               plan.problem.num_head_groups,
                                               *tma_state.get_tma_descriptor(),
                                               stream);
    }

    void Run(const Arguments& args, const delta_rule::Plan& plan, cudaStream_t stream) const override
    {
        if constexpr (kCommit) {
            if (!args.commit_lengths || args.commit_lengths.dtype() != kInt32
                || args.commit_lengths.device().type != kDEVICE || args.commit_lengths.ndim() != 1
                || args.commit_lengths.shape(0) != plan.problem.batch || args.commit_lengths.stride(0) != 1) {
                throw std::invalid_argument("GDR commit_lengths must be a contiguous device int32 [batch] tensor");
            }
            if (!args.state_ptrs || (args.state_ptrs.dtype() != kInt64 && args.state_ptrs.dtype() != kPointer)
                || args.state_ptrs.device().type != kDEVICE
                || (args.state_ptrs.ndim() != 1 && args.state_ptrs.ndim() != 2)
                || args.state_ptrs.shape(0) != plan.problem.batch
                || (args.state_ptrs.ndim() == 1 && plan.problem.num_head_groups != 1)
                || (args.state_ptrs.ndim() == 2 && args.state_ptrs.shape(1) != plan.problem.num_head_groups)) {
                throw std::invalid_argument("GDR commit requires device state_ptrs [batch, num_head_groups]");
            }
            if (!args.state_tma_descs) {
                throw std::invalid_argument("GDR commit requires prepared state_tma_descs");
            }
            if (args.finished && (args.finished.dtype() != kBool || args.finished.device().type != kDEVICE
                                  || args.finished.ndim() != 1 || args.finished.shape(0) != plan.problem.batch
                                  || args.finished.stride(0) != 1)) {
                throw std::invalid_argument("GDR commit finished must be a contiguous device bool [batch] tensor");
            }
        }
        using T      = Element;
        const auto& query = kCommit ? args.k : args.q;
        auto g_q     = make_tensor(
            make_gmem_ptr(reinterpret_cast<const T*>(query.raw_data())),
            make_layout(
                make_shape(QKVector{},
                           QKVectors{},
                           plan.problem.token_num,
                           plan.problem.hq,
                           plan.problem.batch),
                make_stride(_1{},
                            QKVector{},
                            query.stride(1),
                            query.stride(2),
                            query.stride(0))));
        auto g_k = make_tensor(
            make_gmem_ptr(reinterpret_cast<const T*>(args.k.raw_data())),
            make_layout(
                make_shape(QKVector{},
                           QKVectors{},
                           plan.problem.token_num,
                           plan.problem.hq,
                           plan.problem.batch),
                make_stride(_1{},
                            QKVector{},
                            args.k.stride(1),
                            args.k.stride(2),
                            args.k.stride(0))));
        auto g_v = make_tensor(
            make_gmem_ptr(reinterpret_cast<const T*>(args.v.raw_data())),
            make_layout(
                make_shape(BlockDv{},
                           plan.problem.token_num,
                           make_shape(plan.problem.hv, plan.problem.batch)),
                make_stride(_1{},
                            args.v.stride(1),
                            make_stride(args.v.stride(2), args.v.stride(0)))));
        auto tma_k =
            make_tma_copy(kTmaLoad, g_k, SmemLayoutQKTmaTile{}, QKTmaTile{}, _1{});
        auto tma_q = tma_k;
        if constexpr (!kCommit) {
            tma_q = make_tma_copy(kTmaLoad, g_q, SmemLayoutQKTmaTile{}, QKTmaTile{}, _1{});
        }
        auto tma_v =
            make_tma_copy(kTmaLoad, g_v, SmemLayoutValueTma{}, ValueTmaTile{}, _1{});

        const int state_layer =
            static_cast<int>(args.state_layer_offset
                             / (int64_t(plan.problem.heads_per_block) * kHeadDim * kBlockDv));
        TmaState tma_state{};

        const int value_heads_per_query_head = plan.problem.hv / plan.problem.hq;

        using TmaQ                    = decltype(tma_q);
        using TmaK                    = decltype(tma_k);
        using TmaV                    = decltype(tma_v);
        using TmaState                = decltype(tma_state);
        using Operator = DeviceOperator<TmaQ, TmaK, TmaV, TmaState>;
        typename Operator::KernelParams parameters{args.g.data<float>(),
                                                   args.beta.data<float>(),
                                                   reinterpret_cast<const TmaDescriptor*>(
                                                       args.state_tma_descs.raw_data()),
                                                   kCommit ? nullptr : args.out->data<__nv_bfloat16>(),
                                                   plan.problem.token_num,
                                                   plan.problem.batch,
                                                   plan.problem.hq,
                                                   plan.problem.hv,
                                                   value_heads_per_query_head,
                                                   plan.problem.num_head_groups,
                                                   plan.problem.heads_per_block,
                                                   args.g.stride(0),
                                                   args.g.stride(1),
                                                   args.beta.stride(0),
                                                   args.beta.stride(1),
                                                   kCommit ? 0 : args.out->stride(0),
                                                   kCommit ? 0 : args.out->stride(1),
                                                   kCommit ? 0 : args.out->stride(2),
                                                   state_layer,
                                                   kCommit ? reinterpret_cast<void* const*>(args.state_ptrs.raw_data()) : nullptr,
                                                   kCommit ? args.commit_lengths.data<int>() : nullptr,
                                                   kCommit && args.finished ? args.finished.data<bool>() : nullptr,
                                                   kCommit ? args.state_ptrs.stride(0) : 0,
                                                   kCommit && args.state_ptrs.ndim() == 2 ? args.state_ptrs.stride(1) : 0};
        typename Operator::Params kernel_parameters{tma_q, tma_k, tma_v, tma_state, parameters};
        auto kernel = Sm90GdrVerifyCommitDeviceKernel<Operator>;
        TM_CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes)));
        TM_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));

        int active_ctas_per_sm = 0;
        TM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_ctas_per_sm, kernel, Policy::kThreads, kSharedBytes));
        TM_CHECK(active_ctas_per_sm > 0) << "SM90 GDR kernel has zero active CTAs";

        auto head_layout = make_layout(make_shape(plan.problem.hv, plan.problem.batch));
        const int total_heads = int(size(head_layout));
        const int persistent_ctas = std::min(total_heads, getSMCount() * active_ctas_per_sm);
        const dim3 grid(persistent_ctas, 1, 1);
        const dim3 block(Policy::kThreads);
        kernel<<<grid, block, kSharedBytes, stream>>>(kernel_parameters);
        TM_CUDA_CHECK(cudaGetLastError());
    }
};

Registrar gdr_reg([](Collector& c) {
    Sm90GdrVerifyCommitKernel<8>::RegisterStateVariants(c);
    Sm90GdrVerifyCommitKernel<16>::RegisterStateVariants(c);
});

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
