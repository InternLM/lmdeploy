// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_traits.h"
#include "src/turbomind/kernels/gemm/sm90_mxfp4_fp8_traits.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::gemm {
namespace {

__global__ __launch_bounds__(128) void pack_sm90_mxfp4_fp8_unfolded_weight_kernel(
    uint32_t* dst, const uint16_t* src, int output_dim, int input_dim, int total_records)
{
    using namespace cute;

    using PackTraits         = GmmaMxFp4Fp8TraitsBase<64, 128, 1, WG_1x1, 128>;
    using TiledMma           = typename PackTraits::TiledMma;
    constexpr int kFragmentM = PackTraits::kOpM;
    constexpr int kFragmentK = PackTraits::kOpK;

    const int out_fragments = output_dim / kFragmentM;
    const int k_fragments   = input_dim / kFragmentK;
    auto gSrc = make_tensor(make_gmem_ptr(src), make_shape(output_dim, input_dim), make_stride(input_dim, Int<1>{}));

    TiledMma tiled_mma;
    auto     packed_layout = PackTraits::packed_layout_a_mk();
    auto     gDst          = make_tensor(make_gmem_ptr(recast_ptr<cute::uint4_t>(dst)),
                            make_layout(packed_layout, make_layout(total_records, cosize(packed_layout))));
    auto     thr_mma       = tiled_mma.get_thread_slice(threadIdx.x);

    for (int record = int(blockIdx.x); record < total_records; record += int(gridDim.x)) {
        auto record_coord = idx2crd(record, make_shape(out_fragments, k_fragments));
        auto gSrcRecord   = local_tile(gSrc, make_tile(Int<kFragmentM>{}, Int<kFragmentK>{}), record_coord);
        auto gDstSlice    = gDst(_, record);
        auto gDstRecord   = make_tensor(gDstSlice.data(), packed_layout);
        auto tAgSrc       = thr_mma.partition_A(gSrcRecord);
        auto tAgDst       = thr_mma.partition_A(gDstRecord);
        auto tArSrc       = make_fragment_like<uint16_t>(tAgSrc);
        auto tArDst       = make_fragment_like<cute::uint4_t>(tAgDst);
        copy(tAgSrc, tArSrc);

        static_assert(size(tArSrc) == size(tArDst));
        CUTE_UNROLL
        for (int value = 0; value < size(tArDst); ++value) {
            tArDst(value) = cute::uint4_t(unsigned(uint16_t(tArSrc(value)) & 0xfu));
        }
        copy(tArDst, tAgDst);
    }
}

// Each word stores one four-nibble magnitude selector per WGMMA-owned row.
// Sign bits are cross-packed between the rows: row_lo signs occupy odd nibble
// sign positions (already byte MSBs), and row_hi signs occupy even nibble sign
// positions (one shift from byte MSBs). The representation remains exactly
// four bits per weight.
__global__ __launch_bounds__(128) void pack_sm90_mxfp4_fp8_folded_weight_kernel(
    uint32_t* dst, const uint16_t* src, int output_dim, int input_dim, int total_records)
{
    using namespace cute;

    using PackTraits              = GmmaMxFp4Fp8TraitsBase<64, 128, 1, WG_1x1, 128>;
    using TiledMma                = typename PackTraits::TiledMma;
    constexpr int kFragmentM      = PackTraits::kOpM;
    constexpr int kFragmentK      = PackTraits::kOpK;
    constexpr int kWordsPerLane   = 2;
    constexpr int kWordsPerRecord = 128 * kWordsPerLane;

    const int out_fragments = output_dim / kFragmentM;
    const int k_fragments   = input_dim / kFragmentK;
    auto gSrc = make_tensor(make_gmem_ptr(src), make_shape(output_dim, input_dim), make_stride(input_dim, Int<1>{}));

    TiledMma tiled_mma;
    auto     thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    for (int record = int(blockIdx.x); record < total_records; record += int(gridDim.x)) {
        auto record_coord = idx2crd(record, make_shape(out_fragments, k_fragments));
        auto gSrcRecord   = local_tile(gSrc, make_tile(Int<kFragmentM>{}, Int<kFragmentK>{}), record_coord);
        auto tAgSrc       = thr_mma.partition_A(gSrcRecord);
        auto tArSrc       = make_fragment_like<uint16_t>(tAgSrc);
        copy(tAgSrc, tArSrc);
        static_assert(size(tArSrc) == Int<16>{});

        CUTE_UNROLL
        for (int word = 0; word < kWordsPerLane; ++word) {
            uint32_t packed = 0;
            CUTE_UNROLL
            for (int j = 0; j < 8; ++j) {
                const uint16_t bits = tArSrc(word * 8 + j);
                packed |= uint32_t(bits & 0x7u) << (4 * j);
                const int sign_slot = j < 4 ? 2 * j + 1 : 2 * (j - 4);
                packed |= uint32_t(bits & 0x8u) << (4 * sign_slot);
            }
            dst[int64_t(record) * kWordsPerRecord + threadIdx.x * kWordsPerLane + word] = packed;
        }
    }
}

__global__ __launch_bounds__(128) void pack_sm90_mxfp4_fp8_folded_qparams_kernel(
    uint8_t* dst, const uint8_t* src, int output_dim, int total_records, Sm90MxFp4Fp8FoldedPackStats* stats)
{
    using namespace cute;

    using PackTraits               = GmmaMxFp4Fp8FoldedTraits<64, 128, 1, WG_1x1, 128>;
    using TiledMma                 = typename PackTraits::TiledMma;
    constexpr int kFragmentM       = PackTraits::kOpM;
    constexpr int kFragmentK       = PackTraits::kOpK;
    constexpr int kGroupsPerRecord = PackTraits::kKBlocksPerStage;
    constexpr int kShiftValues     = kFragmentM * kGroupsPerRecord;
    constexpr int kRecordValues    = Sm90MxFp4Fp8FoldedFormat::kQparamValuesFragment;
    static_assert(kShiftValues == 256);
    static_assert(kRecordValues == 272);

    static_assert(tile_size<0>(TiledMma{}) == Int<kFragmentM>{});
    static_assert(tile_size<2>(TiledMma{}) == Int<kFragmentK>{});
    static_assert(size(TiledMma{}) == Int<128>{});

    __shared__ int tile_foldable;
    __shared__ int record_emin;
    __shared__ int record_emax;
    __shared__ int record_finite;
    const int      out_fragments = output_dim / kFragmentM;
    const int      k128_groups   = total_records / out_fragments;

    auto gSrc = make_tensor(
        make_gmem_ptr(src), make_shape(output_dim, k128_groups * kGroupsPerRecord), make_stride(Int<1>{}, output_dim));
    auto gShift = make_tensor(make_gmem_ptr(dst), make_layout(make_shape(Int<kShiftValues>{}, total_records)));
    auto gBase  = make_tensor(make_gmem_ptr(dst + size_t(total_records) * kShiftValues),
                             make_layout(make_shape(Int<16>{}, total_records)));

    TiledMma tiled_mma;
    auto     thr_mma  = tiled_mma.get_thread_slice(threadIdx.x);
    auto     identity = make_identity_tensor(Shape<Int<kFragmentM>, Int<kFragmentK>>{});
    auto     tAcA     = thr_mma.partition_A(identity);
    auto     tAcAScan = coalesce(tAcA);

    for (int record = static_cast<int>(blockIdx.x); record < total_records; record += static_cast<int>(gridDim.x)) {
        if (threadIdx.x == 0) {
            tile_foldable = 1;
            record_emin   = 254;
            record_emax   = 0;
            record_finite = 1;
        }
        __syncthreads();

        auto record_coord = idx2crd(record, make_shape(out_fragments, k128_groups));
        auto gSrcRecord   = local_tile(gSrc, make_tile(Int<kFragmentM>{}, Int<kGroupsPerRecord>{}), record_coord);
        CUTE_UNROLL
        for (int value = 0; value < size(tAcAScan); ++value) {
            auto mk = tAcAScan(value);
            if (get<1>(mk) != 0) {
                continue;
            }
            const int row = get<0>(mk);
            CUTE_UNROLL
            for (int group = 0; group < kGroupsPerRecord; ++group) {
                const uint8_t exponent = gSrcRecord(row, group);
                if (exponent == 0xff) {
                    atomicAnd(&record_finite, 0);
                }
                else {
                    atomicMin(&record_emin, static_cast<int>(exponent));
                    atomicMax(&record_emax, static_cast<int>(exponent));
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            tile_foldable = record_finite && record_emax - record_emin <= 6;
        }
        __syncthreads();

        // Four threads differ only in the WGMMA N coordinate and own the
        // same two A rows.  Store those two row scales as one uint16_t in the
        // exact order consumed by local_tid / 4 in the mainloop.
        const int local_tid = threadIdx.x;
        if (local_tid % 4 == 0) {
            const int pair      = local_tid / 4;
            const int row_lo    = get<0>(tAcA(make_coord(Int<0>{}, Int<0>{}, Int<0>{}), Int<0>{}, Int<0>{}));
            const int row_hi    = get<0>(tAcA(make_coord(Int<0>{}, Int<1>{}, Int<0>{}), Int<0>{}, Int<0>{}));
            const int base_code = record_emin;
            CUTE_UNROLL
            for (int group = 0; group < kGroupsPerRecord; ++group) {
                const uint8_t exponent_lo = gSrcRecord(row_lo, group);
                const uint8_t exponent_hi = gSrcRecord(row_hi, group);
                const int     shift_lo    = exponent_lo == 0xff ? 0 : static_cast<int>(exponent_lo) - base_code;
                const int     shift_hi    = exponent_hi == 0xff ? 0 : static_cast<int>(exponent_hi) - base_code;
                // Store ready-to-add E4M3 exponent-field deltas. Packing them
                // together makes the mainloop scale fetch one aligned 16-bit
                // load for both WGMMA-owned rows.
                const uint16_t scale_pair =
                    static_cast<uint16_t>(shift_lo << 3) | static_cast<uint16_t>(shift_hi << 11);
                auto* scale_dst = reinterpret_cast<uint16_t*>(&gShift(group * kFragmentM + 2 * pair, record));
                *scale_dst      = scale_pair;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            const int base_exponent = record_emin - 127;
            assert(!record_finite || (-127 <= base_exponent && base_exponent <= 127));
            gBase(0, record) = record_finite ? static_cast<uint8_t>(static_cast<int8_t>(base_exponent)) : uint8_t{0x80};
            if (stats) {
                atomicAdd(&stats->total_records, 1ull);
                if (tile_foldable) {
                    atomicAdd(&stats->foldable_records, 1ull);
                }
            }
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(128) void pack_sm90_mxfp4_fp8_unfolded_qparams_kernel(uint8_t*       dst,
                                                                                   const uint8_t* src,
                                                                                   int            output_dim,
                                                                                   int            total_records)
{
    using namespace cute;

    using PackTraits               = GmmaMxFp4Fp8UnfoldedTraits<64, 128, 1, WG_1x1, 128>;
    using TiledMma                 = typename PackTraits::TiledMma;
    constexpr int kFragmentM       = PackTraits::kOpM;
    constexpr int kFragmentK       = PackTraits::kOpK;
    constexpr int kGroupsPerRecord = PackTraits::kKBlocksPerStage;
    constexpr int kQparamGroups    = PackTraits::kKBlocksPerStage;

    static_assert(tile_size<0>(TiledMma{}) == Int<kFragmentM>{});
    static_assert(tile_size<2>(TiledMma{}) == Int<kFragmentK>{});
    static_assert(size(TiledMma{}) == Int<128>{});

    const int out_fragments = output_dim / kFragmentM;
    const int k128_groups   = total_records / out_fragments;

    auto gSrc = make_tensor(
        make_gmem_ptr(src), make_shape(output_dim, k128_groups * kGroupsPerRecord), make_stride(Int<1>{}, output_dim));
    auto gDst =
        make_tensor(make_gmem_ptr(dst),
                    make_layout(make_shape(Int<kFragmentM>{}, Int<kQparamGroups>{}, total_records),
                                make_stride(Int<kQparamGroups>{}, Int<1>{}, Int<kFragmentM * kQparamGroups>{})));

    TiledMma tiled_mma;
    auto     thr_mma  = tiled_mma.get_thread_slice(threadIdx.x);
    auto     identity = make_identity_tensor(Shape<Int<kFragmentM>, Int<kFragmentK>>{});
    auto     tAcA     = coalesce(thr_mma.partition_A(identity));

    for (int record = static_cast<int>(blockIdx.x); record < total_records; record += static_cast<int>(gridDim.x)) {
        auto record_coord = idx2crd(record, make_shape(out_fragments, k128_groups));
        auto gSrcRecord   = local_tile(gSrc, make_tile(Int<kFragmentM>{}, Int<kQparamGroups>{}), record_coord);
        CUTE_UNROLL
        for (int value = 0; value < size(tAcA); ++value) {
            auto mk = tAcA(value);
            if (get<1>(mk) != 0) {
                continue;
            }
            const int row = get<0>(mk);
            CUTE_UNROLL
            for (int group = 0; group < kQparamGroups; ++group) {
                gDst(row, group, record) = gSrcRecord(row, group);
            }
        }
    }
}

template<bool ReorderMxFp4>
__global__ __launch_bounds__(256) void pack_sm90_u4_weight_kernel(
    uint32_t* dst, const uint16_t* src, int output_dim, int input_dim, int total_tiles)
{
    using namespace cute;

    using TiledMma = GmmaMixedPackTraits::TiledMma;
    static_assert(size(TiledMma{}) == Int<256>{});
    static_assert(tile_size<0>(TiledMma{}) == Int<kSm90MixedTileN>{});
    static_assert(tile_size<1>(TiledMma{}) == Int<GmmaMixedPackTraits::TILE_BATCH>{});
    static_assert(tile_size<2>(TiledMma{}) == Int<16>{});
    static_assert(GmmaMixedPackTraits::TILE_K == kSm90MixedTileK);
    static_assert(sizeof(bfloat16_t) == sizeof(uint16_t));
    static_assert(kSm90U4WordsPerTile == 1024);

    const int tiles_k     = input_dim / kSm90MixedTileK;
    const int fragments_n = output_dim / kSm90MixedFragmentN;

    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const int tile_n = tile / tiles_k;
        const int tile_k = tile % tiles_k;

        const uint16_t* tile_src =
            src + (size_t)tile_n * kSm90MixedTileN * input_dim + (size_t)tile_k * kSm90MixedTileK;
        auto gA = make_tensor(make_gmem_ptr(reinterpret_cast<const bfloat16_t*>(tile_src)),
                              make_shape(Int<kSm90MixedTileN>{}, Int<kSm90MixedTileK>{}),
                              make_stride(input_dim, Int<1>{}));

        TiledMma tiled_mma;
        auto     thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
        auto     tArA    = thr_mma.make_fragment_A(thr_mma.partition_A(gA));

        auto      tiled_copy = make_tiled_copy_A(Copy_Atom<AutoVectorizingCopy, bfloat16_t>{}, tiled_mma);
        auto      thr_copy   = tiled_copy.get_thread_slice(threadIdx.x);
        auto      tAgA       = thr_copy.partition_S(gA);
        auto      tArA_copy  = thr_copy.retile_D(tArA);
        const int warpgroup  = threadIdx.x / 128;
        const int fragment_n = tile_n * (kSm90MixedTileN / kSm90MixedFragmentN) + warpgroup;

        static_assert(rank(tArA) == Int<3>{});
        static_assert(size<0>(tArA) == Int<8>{});
        static_assert(size<1>(tArA) == Int<1>{});
        static_assert(size<2>(tArA) == Int<4>{});
        static_assert(rank(tAgA) == Int<3>{});
        static_assert(size<0>(tAgA) == Int<8>{});
        static_assert(size<1>(tAgA) == Int<1>{});
        static_assert(size<2>(tAgA) == Int<4>{});
        static_assert(rank(tArA_copy) == Int<3>{});
        static_assert(size<0>(tArA_copy) == Int<8>{});
        static_assert(size<1>(tArA_copy) == Int<1>{});
        static_assert(size<2>(tArA_copy) == Int<4>{});

        const int lane = threadIdx.x % 128;

        if (fragment_n < fragments_n) {
            copy(tiled_copy, tAgA, tArA_copy);
#pragma unroll
            for (int kb = 0; kb < 4; ++kb) {
                uint32_t packed = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint16_t bits = reinterpret_cast<const uint16_t&>(tArA(j, 0, kb));
                    if constexpr (ReorderMxFp4) {
                        // Magnitudes form the two contiguous PRMT lookup groups
                        // {j0,j1,j4,j5} and {j2,j3,j6,j7}. Signs retain the
                        // cross-packed positions consumed directly by BF16x2.
                        const int magnitude_pos = (j & 1) | ((j & 2) << 1) | ((j & 4) >> 1);
                        const int sign_pos      = (j >> 1) + ((j & 1) << 2);
                        packed |= uint32_t(bits & 0x7u) << (4 * magnitude_pos);
                        packed |= uint32_t(bits & 0x8u) << (4 * sign_pos);
                    }
                    else {
                        // lop3 unpacks nibble positions (0,4), (1,5), (2,6),
                        // (3,7) into four BF16x2 registers. Interleave the fragment
                        // values so those pairs recover (j0,j1), ..., (j6,j7).
                        const int nibble_pos = (j >> 1) + ((j & 1) << 2);
                        packed |= uint32_t(bits & 0xfu) << (4 * nibble_pos);
                    }
                }

                const int     fragment_k = tile_k * (kSm90MixedTileK / kSm90MixedFragmentK) + kb;
                const int64_t dst_idx    = ((int64_t)fragment_k * fragments_n + fragment_n) * 128 + lane;
                dst[dst_idx]             = packed;
            }
        }
    }
}

__global__ __launch_bounds__(256) void pack_sm90_fp8_e4m3_weight_kernel(
    uint32_t* dst, const uint16_t* src, int output_dim, int input_dim, int total_tiles)
{
    using namespace cute;

    using TiledMma = GmmaMixedPackTraits::TiledMma;
    static_assert(size(TiledMma{}) == Int<256>{});
    static_assert(tile_size<0>(TiledMma{}) == Int<kSm90MixedTileN>{});
    static_assert(tile_size<1>(TiledMma{}) == Int<GmmaMixedPackTraits::TILE_BATCH>{});
    static_assert(tile_size<2>(TiledMma{}) == Int<16>{});
    static_assert(GmmaMixedPackTraits::TILE_K == kSm90MixedTileK);
    static_assert(sizeof(bfloat16_t) == sizeof(uint16_t));
    static_assert(kSm90Fp8E4M3WordsPerTile == 2048);

    const int tiles_k     = input_dim / kSm90MixedTileK;
    const int fragments_n = output_dim / kSm90MixedFragmentN;

    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const int tile_n = tile / tiles_k;
        const int tile_k = tile % tiles_k;

        const uint16_t* tile_src =
            src + (size_t)tile_n * kSm90MixedTileN * input_dim + (size_t)tile_k * kSm90MixedTileK;
        auto gA = make_tensor(make_gmem_ptr(reinterpret_cast<const bfloat16_t*>(tile_src)),
                              make_shape(Int<kSm90MixedTileN>{}, Int<kSm90MixedTileK>{}),
                              make_stride(input_dim, Int<1>{}));

        TiledMma tiled_mma;
        auto     thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
        auto     tArA    = thr_mma.make_fragment_A(thr_mma.partition_A(gA));

        auto tiled_copy = make_tiled_copy_A(Copy_Atom<AutoVectorizingCopy, bfloat16_t>{}, tiled_mma);
        auto thr_copy   = tiled_copy.get_thread_slice(threadIdx.x);
        auto tAgA       = thr_copy.partition_S(gA);
        auto tArA_copy  = thr_copy.retile_D(tArA);

        static_assert(rank(tArA) == Int<3>{});
        static_assert(size<0>(tArA) == Int<8>{});
        static_assert(size<1>(tArA) == Int<1>{});
        static_assert(size<2>(tArA) == Int<4>{});
        static_assert(rank(tAgA) == Int<3>{});
        static_assert(size<0>(tAgA) == Int<8>{});
        static_assert(size<1>(tAgA) == Int<1>{});
        static_assert(size<2>(tAgA) == Int<4>{});
        static_assert(rank(tArA_copy) == Int<3>{});
        static_assert(size<0>(tArA_copy) == Int<8>{});
        static_assert(size<1>(tArA_copy) == Int<1>{});
        static_assert(size<2>(tArA_copy) == Int<4>{});

        copy(tiled_copy, tAgA, tArA_copy);

        const int warpgroup = threadIdx.x / 128;
        const int lane      = threadIdx.x % 128;

#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            uint32_t packed[2]{};
#pragma unroll
            for (int word = 0; word < 2; ++word) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const uint16_t direct  = reinterpret_cast<const uint16_t&>(tArA(2 * word + half, 0, kb));
                    const uint16_t rotated = reinterpret_cast<const uint16_t&>(tArA(2 * word + half + 4, 0, kb));
                    const int      base    = 16 * half;
                    packed[word] |= uint32_t(direct & 0x7fu) << (base + 4);
                    packed[word] |= uint32_t(direct & 0x80u) << (base + 8);
                    packed[word] |= uint32_t(rotated & 0x0fu) << base;
                    packed[word] |= uint32_t(rotated & 0x70u) << (base + 8);
                    packed[word] |= uint32_t(rotated & 0x80u) << (base + 4);
                }
            }

            const int     fragment_n = tile_n * (kSm90MixedTileN / kSm90MixedFragmentN) + warpgroup;
            const int     fragment_k = tile_k * (kSm90MixedTileK / kSm90MixedFragmentK) + kb;
            const int64_t dst_idx    = ((int64_t)fragment_k * fragments_n + fragment_n) * 256 + lane * 2;
            dst[dst_idx]             = packed[0];
            dst[dst_idx + 1]         = packed[1];
        }
    }
}

template<int Offset, class D, class S>
__global__ __launch_bounds__(256) void pack_sm90_qparams_kernel(D* dst, const S* src, int output_dim, int total_tiles)
{
    using namespace cute;

    using TiledMma = GmmaMixedPackTraits::TiledMma;
    static_assert(size(TiledMma{}) == Int<256>{});
    static_assert(tile_size<0>(TiledMma{}) == Int<kSm90MixedTileN>{});

    const int tiles_n     = (output_dim + kSm90MixedTileN - 1) / kSm90MixedTileN;
    const int fragments_n = output_dim / kSm90MixedFragmentN;

    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const int group  = tile / tiles_n;
        const int tile_n = tile % tiles_n;

        TiledMma tiled_mma;
        auto     identity = make_identity_tensor(Shape<Int<kSm90MixedTileN>, Int<kSm90MixedFragmentK>>{});
        auto     thr_mma  = tiled_mma.get_thread_slice(threadIdx.x);
        auto     tAcA     = thr_mma.partition_A(identity);
        static_assert(size(tAcA) == Int<8>{});

        const int local_tid  = threadIdx.x % 128;
        const int warpgroup  = threadIdx.x / 128;
        const int fragment_n = tile_n * (kSm90MixedTileN / kSm90MixedFragmentN) + warpgroup;
        if (fragment_n < fragments_n && local_tid % 4 == 0) {
            const int pair = local_tid / 4;
            const int m_lo = get<0>(tAcA(0));
            const int m_hi = get<0>(tAcA(2));

            const S* tile_src = src + (int64_t)group * output_dim + tile_n * kSm90MixedTileN;
            D*       fragment = dst + ((int64_t)group * fragments_n + fragment_n) * kSm90MixedFragmentN;
            static_assert(std::is_same_v<D, S>);
            fragment[2 * pair]     = static_cast<D>(static_cast<int>(tile_src[m_lo]) + Offset);
            fragment[2 * pair + 1] = static_cast<D>(static_cast<int>(tile_src[m_hi]) + Offset);
        }
    }
}

template<class T>
__global__ __launch_bounds__(256) void pack_sm90_u4_qparams_kernel(
    uint8_t* dst, const T* scales, const T* zeros, int output_dim, int total_tiles)
{
    static_assert(std::is_same_v<T, half_t> || std::is_same_v<T, bfloat16_t>);
    using namespace cute;

    using TiledMma = GmmaMixedPackTraits::TiledMma;
    static_assert(size(TiledMma{}) == Int<256>{});
    static_assert(tile_size<0>(TiledMma{}) == Int<kSm90MixedTileN>{});

    const int tiles_n     = (output_dim + kSm90MixedTileN - 1) / kSm90MixedTileN;
    const int fragments_n = output_dim / kSm90MixedFragmentN;

    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const int group  = tile / tiles_n;
        const int tile_n = tile % tiles_n;

        TiledMma tiled_mma;
        auto     identity = make_identity_tensor(Shape<Int<kSm90MixedTileN>, Int<kSm90MixedFragmentK>>{});
        auto     thr_mma  = tiled_mma.get_thread_slice(threadIdx.x);
        auto     tAcA     = thr_mma.partition_A(identity);
        static_assert(size(tAcA) == Int<8>{});

        const int local_tid  = threadIdx.x % 128;
        const int warpgroup  = threadIdx.x / 128;
        const int fragment_n = tile_n * (kSm90MixedTileN / kSm90MixedFragmentN) + warpgroup;
        if (fragment_n < fragments_n && local_tid % 4 == 0) {
            const int pair = local_tid / 4;
            const int m_lo = get<0>(tAcA(0));
            const int m_hi = get<0>(tAcA(2));

            const auto* tile_scales = scales + (int64_t)group * output_dim + tile_n * kSm90MixedTileN;
            const auto* tile_zeros  = zeros ? zeros + (int64_t)group * output_dim + tile_n * kSm90MixedTileN : nullptr;
            auto*       fragment    = dst + ((int64_t)group * fragments_n + fragment_n) * kSm90U4QparamValuesFragment;

            reinterpret_cast<uint32_t*>(fragment)[pair] =
                uint32_t(reinterpret_cast<const uint16_t&>(tile_scales[m_lo]))
                | (uint32_t(reinterpret_cast<const uint16_t&>(tile_scales[m_hi])) << 16);

            uint8_t zero_lo = 0;
            uint8_t zero_hi = 0;
            if (tile_zeros) {
                if constexpr (std::is_same_v<T, half_t>) {
                    zero_lo = static_cast<uint8_t>(__half2int_rz(tile_zeros[m_lo]));
                    zero_hi = static_cast<uint8_t>(__half2int_rz(tile_zeros[m_hi]));
                }
                else {
                    zero_lo = static_cast<uint8_t>(__bfloat162int_rz(tile_zeros[m_lo]));
                    zero_hi = static_cast<uint8_t>(__bfloat162int_rz(tile_zeros[m_hi]));
                }
            }
            const uint32_t zero_pair = zero_lo | (zero_hi << 4);
            const unsigned mask      = __activemask();
            const int      lane_base = (local_tid % 32) & ~15;
            const uint32_t zero_0    = __shfl_sync(mask, zero_pair, lane_base);
            const uint32_t zero_1    = __shfl_sync(mask, zero_pair, lane_base + 4);
            const uint32_t zero_2    = __shfl_sync(mask, zero_pair, lane_base + 8);
            const uint32_t zero_3    = __shfl_sync(mask, zero_pair, lane_base + 12);
            if (local_tid % 16 == 0) {
                // Store the four low zeros followed by the four high zeros so
                // a four-bit lane shift aligns both with the LOP3 nibble mask.
                const uint32_t zero_word = (zero_0 & 0x0fu) | ((zero_1 & 0x0fu) << 4) | ((zero_2 & 0x0fu) << 8)
                                           | ((zero_3 & 0x0fu) << 12) | ((zero_0 & 0xf0u) << 12)
                                           | ((zero_1 & 0xf0u) << 16) | ((zero_2 & 0xf0u) << 20)
                                           | ((zero_3 & 0xf0u) << 24);
                reinterpret_cast<uint32_t*>(fragment + kSm90MixedFragmentN * sizeof(uint16_t))[pair / 4] = zero_word;
            }
        }
    }
}

__global__ void
pack_sm90_fp8_e4m3_scales_kernel(bfloat16_t* dst, const float* src, int group_count, int output_pack_count)
{
    const int idx = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < group_count * output_pack_count) {
        const int        group       = idx / output_pack_count;
        const int        output_pack = idx % output_pack_count;
        const bfloat16_t scale       = __float2bfloat16_rn(src[idx]);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int i = 0; i < Sm90Fp8E4M3Format::kQparamValuesFragment; ++i) {
                const int fragment = output_pack * 2 + half;
                dst[((int64_t)group * output_pack_count * 2 + fragment) * Sm90Fp8E4M3Format::kQparamValuesFragment
                    + i]           = scale;
            }
        }
    }
}

}  // namespace

void PackSm90U4Weight(uint32_t* dst, const uint16_t* src, int output_dim, int input_dim, cudaStream_t stream)
{
    TM_CHECK_NOTNULL(dst);
    TM_CHECK_NOTNULL(src);
    TM_CHECK_GT(output_dim, 0);
    TM_CHECK_GT(input_dim, 0);
    TM_CHECK_EQ(output_dim % kSm90MixedFragmentN, 0);
    TM_CHECK_EQ(input_dim % kSm90MixedTileK, 0);

    const int total_tiles = ((output_dim + kSm90MixedTileN - 1) / kSm90MixedTileN) * (input_dim / kSm90MixedTileK);
    const int grid        = std::min(total_tiles, 65535);
    pack_sm90_u4_weight_kernel<false><<<grid, 256, 0, stream>>>(dst, src, output_dim, input_dim, total_tiles);
    TM_CUDA_CHECK(cudaGetLastError());
}

void PackSm90Fp4PrmtWeight(uint32_t* dst, const uint16_t* src, int output_dim, int input_dim, cudaStream_t stream)
{
    TM_CHECK_NOTNULL(dst);
    TM_CHECK_NOTNULL(src);
    TM_CHECK_GT(output_dim, 0);
    TM_CHECK_GT(input_dim, 0);
    TM_CHECK_EQ(output_dim % kSm90MixedFragmentN, 0);
    TM_CHECK_EQ(input_dim % kSm90MixedTileK, 0);

    const int total_tiles = ((output_dim + kSm90MixedTileN - 1) / kSm90MixedTileN) * (input_dim / kSm90MixedTileK);
    const int grid        = std::min(total_tiles, 65535);
    pack_sm90_u4_weight_kernel<true><<<grid, 256, 0, stream>>>(dst, src, output_dim, input_dim, total_tiles);
    TM_CUDA_CHECK(cudaGetLastError());
}

void PackSm90Fp8E4M3Weight(uint32_t* dst, const uint16_t* src, int output_dim, int input_dim, cudaStream_t stream)
{
    TM_CHECK_NOTNULL(dst);
    TM_CHECK_NOTNULL(src);
    TM_CHECK_GT(output_dim, 0);
    TM_CHECK_GT(input_dim, 0);
    TM_CHECK_EQ(output_dim % kSm90MixedTileN, 0);
    TM_CHECK_EQ(input_dim % kSm90MixedTileK, 0);

    const int total_tiles = (output_dim / kSm90MixedTileN) * (input_dim / kSm90MixedTileK);
    const int grid        = std::min(total_tiles, 65535);
    pack_sm90_fp8_e4m3_weight_kernel<<<grid, 256, 0, stream>>>(dst, src, output_dim, input_dim, total_tiles);
    TM_CUDA_CHECK(cudaGetLastError());
}

template<int Offset = 0, class D, class S>
void PackSm90QParams(D* dst, const S* src, int output_dim, int group_count, cudaStream_t stream)
{
    TM_CHECK_NOTNULL(dst);
    TM_CHECK_NOTNULL(src);
    TM_CHECK_GT(output_dim, 0);
    TM_CHECK_GT(group_count, 0);
    TM_CHECK_EQ(output_dim % kSm90MixedFragmentN, 0);

    const int total_tiles = group_count * ((output_dim + kSm90MixedTileN - 1) / kSm90MixedTileN);
    const int grid        = std::min(total_tiles, 65535);
    pack_sm90_qparams_kernel<Offset><<<grid, 256, 0, stream>>>(dst, src, output_dim, total_tiles);
    TM_CUDA_CHECK(cudaGetLastError());
}

template<class T>
void PackSm90U4QParams(
    uint8_t* dst, const T* scales, const T* zeros, int output_dim, int group_count, cudaStream_t stream)
{
    TM_CHECK_NOTNULL(dst);
    TM_CHECK_NOTNULL(scales);
    TM_CHECK_GT(output_dim, 0);
    TM_CHECK_GT(group_count, 0);
    TM_CHECK_EQ(output_dim % kSm90MixedFragmentN, 0);

    const int total_tiles = group_count * ((output_dim + kSm90MixedTileN - 1) / kSm90MixedTileN);
    const int grid        = std::min(total_tiles, 65535);
    pack_sm90_u4_qparams_kernel<<<grid, 256, 0, stream>>>(dst, scales, zeros, output_dim, total_tiles);
    TM_CUDA_CHECK(cudaGetLastError());
}

template void PackSm90U4QParams(uint8_t*, const half_t*, const half_t*, int, int, cudaStream_t);
template void PackSm90U4QParams(uint8_t*, const bfloat16_t*, const bfloat16_t*, int, int, cudaStream_t);

void PackSm90Fp4QParams(uint8_t* dst, const uint8_t* src, int output_dim, int group_count, cudaStream_t stream)
{
    PackSm90QParams(dst, src, output_dim, group_count, stream);
}

void PackSm90MxFp4QParams(uint8_t* dst, const uint8_t* src, int output_dim, int group_count, cudaStream_t stream)
{
    PackSm90QParams<-127>(dst, src, output_dim, group_count, stream);
}

void PackSm90MxFp4Fp8FoldedQParams(uint8_t*                     dst,
                                   const uint8_t*               src,
                                   int                          output_dim,
                                   int                          group_count,
                                   cudaStream_t                 stream,
                                   Sm90MxFp4Fp8FoldedPackStats* device_stats)
{
    TM_CHECK_NOTNULL(dst);
    TM_CHECK_NOTNULL(src);
    TM_CHECK_GT(output_dim, 0);
    TM_CHECK_GT(group_count, 0);
    TM_CHECK_EQ(output_dim % 64, 0);
    TM_CHECK_EQ(group_count % 4, 0);

    const int total_records = (group_count / 4) * (output_dim / 64);
    const int grid          = std::min(total_records, 65535);
    pack_sm90_mxfp4_fp8_folded_qparams_kernel<<<grid, 128, 0, stream>>>(
        dst, src, output_dim, total_records, device_stats);
    TM_CUDA_CHECK(cudaGetLastError());
}

void PackSm90MxFp4Fp8UnfoldedQParams(
    uint8_t* dst, const uint8_t* src, int output_dim, int group_count, cudaStream_t stream)
{
    TM_CHECK_NOTNULL(dst);
    TM_CHECK_NOTNULL(src);
    TM_CHECK_GT(output_dim, 0);
    TM_CHECK_GT(group_count, 0);
    TM_CHECK_EQ(output_dim % 64, 0);
    TM_CHECK_EQ(group_count % 4, 0);

    const int total_records = (group_count / 4) * (output_dim / 64);
    const int grid          = std::min(total_records, 65535);
    pack_sm90_mxfp4_fp8_unfolded_qparams_kernel<<<grid, 128, 0, stream>>>(dst, src, output_dim, total_records);
    TM_CUDA_CHECK(cudaGetLastError());
}

void PackSm90MxFp4Fp8FoldedWeight(
    uint32_t* dst, const uint16_t* src, int output_dim, int input_dim, cudaStream_t stream)
{
    TM_CHECK_NOTNULL(dst);
    TM_CHECK_NOTNULL(src);
    TM_CHECK_GT(output_dim, 0);
    TM_CHECK_GT(input_dim, 0);
    TM_CHECK_EQ(output_dim % 64, 0);
    TM_CHECK_EQ(input_dim % 128, 0);

    const int total_records = (input_dim / 32) * (output_dim / 64);
    const int grid          = std::min(total_records, 65535);
    pack_sm90_mxfp4_fp8_folded_weight_kernel<<<grid, 128, 0, stream>>>(dst, src, output_dim, input_dim, total_records);
    TM_CUDA_CHECK(cudaGetLastError());
}

void PackSm90MxFp4Fp8UnfoldedWeight(
    uint32_t* dst, const uint16_t* src, int output_dim, int input_dim, cudaStream_t stream)
{
    TM_CHECK_NOTNULL(dst);
    TM_CHECK_NOTNULL(src);
    TM_CHECK_GT(output_dim, 0);
    TM_CHECK_GT(input_dim, 0);
    TM_CHECK_EQ(output_dim % 64, 0);
    TM_CHECK_EQ(input_dim % 128, 0);

    const int total_records = (input_dim / 32) * (output_dim / 64);
    const int grid          = std::min(total_records, 65535);
    pack_sm90_mxfp4_fp8_unfolded_weight_kernel<<<grid, 128, 0, stream>>>(
        dst, src, output_dim, input_dim, total_records);
    TM_CUDA_CHECK(cudaGetLastError());
}

void PackSm90Fp8E4M3Scales(
    bfloat16_t* dst, const float* src, int group_count, int output_pack_count, cudaStream_t stream)
{
    TM_CHECK_NOTNULL(dst);
    TM_CHECK_NOTNULL(src);
    TM_CHECK_GT(group_count, 0);
    TM_CHECK_GT(output_pack_count, 0);

    constexpr int block = 256;
    const int     count = group_count * output_pack_count;
    pack_sm90_fp8_e4m3_scales_kernel<<<(count + block - 1) / block, block, 0, stream>>>(
        dst, src, group_count, output_pack_count);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind::gemm
