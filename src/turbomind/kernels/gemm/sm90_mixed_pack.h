// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cassert>
#include <cstdint>

#include <cuda_runtime.h>
#include <cute/config.hpp>

#include "src/turbomind/kernels/gemm/types.h"

// The native SM90 mixed mainloop requires CUDA 12.3. Keep converter and
// registrar availability on this single compile-time gate.
#ifndef TM_GEMM_HAS_SM90_MIXED
#if defined(__CUDACC_VER_MAJOR__)                                                                                      \
    && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3))
#define TM_GEMM_HAS_SM90_MIXED 1
#else
#define TM_GEMM_HAS_SM90_MIXED 0
#endif
#endif

namespace turbomind::gemm {

inline constexpr Pack kSm90MixedWeightPack = GMMA_64x16_RS | OPERAND_A | 1;
// Version 2 stores each OUT64 qparam fragment in the RS operand-A consumer
// order. FP8's one-scale-per-OUT128 record retains its original layout.
inline constexpr Pack kSm90MixedQParamPack    = GMMA_64x16_RS | OPERAND_U | 2;
// Version 5 stores each MXFP4 scale as the signed unbiased exponent
// (UE8M0 - 127) in one byte, in operand-A consumer order. The mixed mainloop
// injects it into the constant BF16 E2M1 table before the PRMT lookup.
inline constexpr Pack kSm90MxFp4QParamPack    = GMMA_64x16_RS | OPERAND_U | 5;
inline constexpr Pack kSm90MixedFp8QParamPack = GMMA_64x16_RS | OPERAND_U | 1;
// Both FP8 x MXFP4 paths keep a compact E2M1 RS image.  The folded path also
// stores one signed unbiased K128 base exponent and four relative K32 exponent
// shifts; the unfolded path retains the four source UE8M0 exponents.
inline constexpr Pack kSm90MxFp4Fp8FoldedWeightPack = GMMA_64x32_RS | OPERAND_A | 8;
inline constexpr Pack kSm90MxFp4Fp8UnfoldedWeightPack = GMMA_64x32_RS | OPERAND_A | 5;
inline constexpr Pack kSm90MxFp4Fp8FoldedQParamPack = GMMA_64x32_RS | OPERAND_U | 7;
inline constexpr Pack kSm90MxFp4Fp8UnfoldedQParamPack = GMMA_64x32_RS | OPERAND_U | 5;

// Inject an unbiased power-of-two exponent into an existing positive FP32
// scale.  The activation scale is normal and nonzero, so this is an
// exponent-field adjustment rather than a separate conversion and multiply.
__device__ __forceinline__ float inject_unbiased_exponent(float value, int exponent)
{
#if defined(__CUDA_ARCH__)
    const uint32_t delta = static_cast<uint32_t>(exponent) << 23;
    return __uint_as_float(__float_as_uint(value) + delta);
#else
    return value;
#endif
}

// The unfolded format deliberately retains raw UE8M0 bytes.
__device__ __forceinline__ float inject_ue8m0_exponent(float value, uint8_t exponent)
{
    return inject_unbiased_exponent(value, static_cast<int>(exponent) - 127);
}

namespace detail {

__device__ __forceinline__ uint32_t prmt(uint32_t x, uint32_t y, uint32_t selector)
{
#if defined(__CUDA_ARCH__)
    uint32_t out;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(x), "r"(y), "r"(selector));
    return out;
#else
    return 0;
#endif
}

// The unfolded kernel retains its byte-lane pack: one row in each nibble of
// every byte. Keep that conversion local to the unfolded representation; the
// folded hot path below uses the compact PRMT lookup instead.
CUTE_HOST_DEVICE constexpr uint32_t encode_e2m1x4_lanes_unscaled(uint32_t lane)
{
    constexpr uint32_t kLaneBit0 = 0x01010101u;
    const uint32_t sign = (lane & 0x08080808u) << 4;
    const uint32_t e2 = (lane & 0x04040404u) << 4;
    const uint32_t e1 = (lane & 0x02020202u) << 2;
    const uint32_t high = (lane >> 2) & kLaneBit0;
    const uint32_t low_nonzero = ((lane | (lane >> 1)) & kLaneBit0) & ~high;
    const uint32_t mantissa =
        ((lane & kLaneBit0) & ((lane >> 1) | (lane >> 2)) & kLaneBit0) << 2;
    return sign | e2 | e1 | (low_nonzero << 4) | (low_nonzero << 5) | mantissa;
}

__device__ __forceinline__ void unpack_e2m1x8_to_e4m3x4x2(
    uint32_t packed, uint32_t& row_lo, uint32_t& row_hi)
{
    constexpr uint32_t kNibbleLanes = 0x0f0f0f0fu;
    row_lo = encode_e2m1x4_lanes_unscaled(packed & kNibbleLanes);
    row_hi = encode_e2m1x4_lanes_unscaled((packed >> 4) & kNibbleLanes);
}

struct alignas(8) E2m1E4m3ScaleTable {
    uint32_t values0_to_3;
    uint32_t values4_to_7;
};

CUTE_HOST_DEVICE constexpr E2m1E4m3ScaleTable make_e2m1_e4m3_scale_table(
    uint32_t shift)
{
    constexpr uint32_t kValues0To3 = 0x3c383000u;
    constexpr uint32_t kValues4To7 = 0x4c484440u;
    constexpr uint32_t kNonzero0To3 = 0xffffff00u;
    const uint32_t delta = shift * 0x08080808u;
    return {
        kValues0To3 + (delta & kNonzero0To3),
        kValues4To7 + delta,
    };
}

static_assert(sizeof(E2m1E4m3ScaleTable) == 8);

// Two packed E2M1 words are one complete per-thread K32 source fragment.
// Materialize the corresponding four-register E4M3 RS-WGMMA fragment in its
// native contiguous order: {row_lo_k0, row_hi_k0, row_lo_k16, row_hi_k16}.
struct E2m1E4m3ScaleTables {
    uint32_t values0_to_3_lo;
    uint32_t values4_to_7_lo;
    uint32_t values0_to_3_hi;
    uint32_t values4_to_7_hi;
};

// Each packed word carries two rows of four E2M1 values. Magnitudes stay in
// their native nibble slots. Signs are cross-packed: odd nibble sign bits are
// row_lo's ready E4M3 signs, while even nibble sign bits become row_hi's signs
// after one shift. This keeps the source at four bits/value and avoids a sign
// lookup for either output register.
__device__ __forceinline__ void decode_e2m1x8_paired_signs(
    uint32_t packed,
    const E2m1E4m3ScaleTables& tables,
    uint32_t& row_lo,
    uint32_t& row_hi)
{
    constexpr uint32_t kMagnitudeSelectors = 0x00007777u;
    constexpr uint32_t kE4m3Signs = 0x80808080u;

    const uint32_t lo_selector = packed & kMagnitudeSelectors;
    const uint32_t hi_selector = (packed >> 16) & kMagnitudeSelectors;
    const uint32_t lo_value =
        prmt(tables.values0_to_3_lo, tables.values4_to_7_lo, lo_selector);
    const uint32_t hi_value =
        prmt(tables.values0_to_3_hi, tables.values4_to_7_hi, hi_selector);
    row_lo = lo_value | (packed & kE4m3Signs);
    row_hi = hi_value | ((packed << 4) & kE4m3Signs);
}

__device__ __forceinline__ void unpack_e2m1x16_to_e4m3x4x4(
    const uint32_t* packed,
    const E2m1E4m3ScaleTables& tables,
    uint32_t* out)
{
    decode_e2m1x8_paired_signs(packed[0], tables, out[0], out[1]);
    decode_e2m1x8_paired_signs(packed[1], tables, out[2], out[3]);
}

}  // namespace detail

inline constexpr int kSm90MixedTileN     = 128;
inline constexpr int kSm90MixedTileK     = 64;
inline constexpr int kSm90MixedFragmentN = 64;
inline constexpr int kSm90MixedFragmentK = 16;
inline constexpr int kSm90U4WordsPerTile = kSm90MixedTileN * kSm90MixedTileK / 8;
inline constexpr int kSm90U4QparamValuesFragment =
    kSm90MixedFragmentN * sizeof(bfloat16_t) + kSm90MixedFragmentN / 2;

inline constexpr int kSm90Fp8E4M3GroupSize    = 128;
inline constexpr int kSm90Fp8E4M3WordsPerTile = kSm90MixedTileN * kSm90MixedTileK / 4;

template<int GroupSize>
struct Sm90U4Format {
    static_assert(GroupSize == 32 || GroupSize == 128);

    using WeightType = uint4_t;
    using QparamType = uint8_t;
    using QparamSourceType = uint8_t;

    static constexpr Pack kWeightPack           = kSm90MixedWeightPack;
    static constexpr Pack kQparamPack           = kSm90MixedQParamPack;
    static constexpr int  kGroupSize            = GroupSize;
    static constexpr int  kWeightBits           = 4;
    static constexpr int  kScaleGroupN          = 1;
    static constexpr int  kQparamFragmentN      = kSm90MixedFragmentN;
    static constexpr int  kQparamValuesFragment = kSm90U4QparamValuesFragment;
    static constexpr int  kQparamValuesTile     = 2 * kQparamValuesFragment;
    static constexpr int  kFusedSiluBlock       = 64;
    static constexpr bool kHasGlobalScale       = false;
    static constexpr auto kQuantType            = QuantType::kK;
    static constexpr auto kConverterOrder       = kRowMajor;
    static constexpr auto kPublicWeightOrder    = kColMajor;
};

// MXFP4 uses the same register-source A fragment pack. Each qparam byte stores
// the signed unbiased exponent (source UE8M0 - 127) for one K32 group.
struct Sm90MxFp4Format {
    using WeightType = fp4_e2m1_t;
    using QparamType = uint8_t;
    using QparamSourceType = uint8_t;

    static constexpr Pack kWeightPack           = kSm90MixedWeightPack;
    static constexpr Pack kQparamPack           = kSm90MxFp4QParamPack;
    static constexpr int  kGroupSize            = 32;
    static constexpr int  kWeightBits           = 4;
    static constexpr int  kScaleGroupN          = 1;
    static constexpr int  kQparamFragmentN      = kSm90MixedFragmentN;
    static constexpr int  kQparamValuesFragment = kSm90MixedFragmentN;
    static constexpr int  kQparamValuesTile     = kSm90MixedTileN;
    static constexpr int  kFusedSiluBlock       = 64;
    static constexpr bool kHasGlobalScale       = false;
    static constexpr auto kQuantType            = QuantType::kK;
    static constexpr auto kConverterOrder       = kRowMajor;
    static constexpr auto kPublicWeightOrder    = kColMajor;
};

// Folded FP8 K32 RS format.  Persistent weights remain packed E2M1.  Each
// Qparams contain a 256-byte-per-record relative-shift plane followed by a
// separate 16-byte-per-record base plane (one signed unbiased exponent byte
// plus padding).
struct Sm90MxFp4Fp8FoldedFormat {
    using WeightType = fp4_e2m1_t;
    using QparamType = uint8_t;
    using QparamSourceType = uint8_t;

    static constexpr Pack kWeightPack            = kSm90MxFp4Fp8FoldedWeightPack;
    static constexpr Pack kQparamPack            = kSm90MxFp4Fp8FoldedQParamPack;
    static constexpr int  kGroupSize             = 32;
    static constexpr int  kWeightBits            = 4;
    static constexpr int  kScaleGroupN           = 1;
    static constexpr int  kQparamFragmentN       = 64;
    static constexpr int  kQparamValuesFragment  = 272;
    static constexpr int  kQparamValuesTile      = kQparamValuesFragment;
    static constexpr int  kFusedSiluBlock        = 128;
    static constexpr bool kHasGlobalScale        = false;
    static constexpr bool kFolded                 = true;
    static constexpr bool kUnfolded               = false;
    static constexpr auto kQuantType             = QuantType::kK;
    static constexpr auto kConverterOrder        = kRowMajor;
    static constexpr auto kPublicWeightOrder     = kColMajor;
};

// Unfolded FP8 K32 RS format. Each OUT64 x K128 qparam record retains the
// source [group4][row64] UE8M0 bytes without conversion.
struct Sm90MxFp4Fp8UnfoldedFormat {
    using WeightType = fp4_e2m1_t;
    using QparamType = uint8_t;
    using QparamSourceType = uint8_t;

    static constexpr Pack kWeightPack            = kSm90MxFp4Fp8UnfoldedWeightPack;
    static constexpr Pack kQparamPack            = kSm90MxFp4Fp8UnfoldedQParamPack;
    static constexpr int  kGroupSize             = 32;
    static constexpr int  kWeightBits            = 4;
    static constexpr int  kScaleGroupN           = 1;
    static constexpr int  kQparamFragmentN       = 64;
    static constexpr int  kQparamValuesFragment  = 4 * kQparamFragmentN;
    static constexpr int  kQparamValuesTile      = kQparamValuesFragment;
    static constexpr bool kHasGlobalScale        = false;
    static constexpr bool kFolded                 = false;
    static constexpr bool kUnfolded               = true;
    static constexpr auto kQuantType             = QuantType::kK;
    static constexpr auto kConverterOrder        = kRowMajor;
    static constexpr auto kPublicWeightOrder     = kColMajor;
};

struct Sm90MxFp4Fp8FoldedPackStats {
    unsigned long long total_records;
    unsigned long long foldable_records;
};

// NVFP4 keeps E2M1 data but uses one unsigned E4M3 scale per K16 block.
// A separate FP32 tensor-level scale accompanies the E4M3 block-scale matrix.
struct Sm90NvFp4Format {
    using WeightType = fp4_e2m1_t;
    using QparamType = uint8_t;
    using QparamSourceType = uint8_t;

    static constexpr Pack kWeightPack           = kSm90MixedWeightPack;
    static constexpr Pack kQparamPack           = kSm90MixedQParamPack;
    static constexpr int  kGroupSize            = 16;
    static constexpr int  kWeightBits           = 4;
    static constexpr int  kScaleGroupN          = 1;
    static constexpr int  kQparamFragmentN      = kSm90MixedFragmentN;
    static constexpr int  kQparamValuesFragment = kSm90MixedFragmentN;
    static constexpr int  kQparamValuesTile     = kSm90MixedTileN;
    static constexpr int  kFusedSiluBlock       = 64;
    static constexpr bool kHasGlobalScale       = true;
    static constexpr auto kQuantType            = QuantType::kK;
    static constexpr auto kConverterOrder       = kRowMajor;
    static constexpr auto kPublicWeightOrder    = kColMajor;
};

// Each E4M3 K16 lane fragment stores two pair-plane words. The first two
// BF16x2 pairs occupy their final bit positions directly; the last two occupy
// the complementary positions with each byte's nibbles rotated. Each B128
// scale record is one replicated 16-byte tensor-copy record.
struct Sm90Fp8E4M3Format {
    using WeightType = fp8_e4m3_t;
    using QparamType = bfloat16_t;

    static constexpr Pack kWeightPack           = kSm90MixedWeightPack;
    static constexpr Pack kQparamPack           = kSm90MixedFp8QParamPack;
    static constexpr int  kGroupSize            = kSm90Fp8E4M3GroupSize;
    static constexpr int  kWeightBits           = 8;
    static constexpr int  kScaleGroupN          = kSm90MixedTileN;
    static constexpr int  kQparamFragmentN      = kSm90MixedFragmentN;
    static constexpr int  kQparamValuesFragment = 8;
    static constexpr int  kQparamValuesTile     = 2 * kQparamValuesFragment;
    static constexpr int  kFusedSiluBlock       = 64;
    static constexpr bool kHasGlobalScale       = false;
    static constexpr auto kQuantType            = QuantType::kB;
    static constexpr auto kConverterOrder       = kRowMajor;
    static constexpr auto kPublicWeightOrder    = kColMajor;
};

// Transform a dense row-major [N, K] tensor of unpacked UINT4 values (one
// value in the low nibble of every uint16_t) into persistent
// [K/16, N/64, RS fragment] order for the SM90 U4 format.
void PackSm90U4Weight(uint32_t* dst, const uint16_t* src, int output_dim, int input_dim, cudaStream_t stream);

// MXFP4 and NVFP4 share the E2M1 PRMT layout: magnitude selectors are
// contiguous within each 16-bit half and signs occupy their final BF16 lanes.
void PackSm90Fp4PrmtWeight(uint32_t* dst, const uint16_t* src, int output_dim, int input_dim, cudaStream_t stream);

// Transform dense row-major [N, K] E4M3 bytes (held in the low byte of each
// uint16_t source value) into persistent [K/16, N/64, pair-plane RS fragment]
// order.
void PackSm90Fp8E4M3Weight(uint32_t* dst, const uint16_t* src, int output_dim, int input_dim, cudaStream_t stream);

// Transform col-major [N, K/group] BF16 scales and zeros into persistent
// [K/group, N/64, RS qparam fragment] order. Each fragment stores 64 BF16
// scales followed by 64 U4 zero points packed two per byte.
void PackSm90U4QParams(uint8_t*          dst,
                       const bfloat16_t* scales,
                       const bfloat16_t* zeros,
                       int               output_dim,
                       int               group_count,
                       cudaStream_t      stream);

void PackSm90Fp4QParams(uint8_t* dst, const uint8_t* src, int output_dim, int group_count, cudaStream_t stream);

// Convert raw MXFP4 UE8M0 bytes to signed unbiased exponents and reorder them
// into operand-A consumer order. The signed values retain uint8_t storage so
// the persistent representation remains one byte per scale.
void PackSm90MxFp4QParams(uint8_t* dst, const uint8_t* src, int output_dim, int group_count, cudaStream_t stream);

void PackSm90MxFp4Fp8FoldedWeight(
    uint32_t* dst, const uint16_t* src, int output_dim, int input_dim, cudaStream_t stream);

void PackSm90MxFp4Fp8UnfoldedWeight(
    uint32_t* dst, const uint16_t* src, int output_dim, int input_dim, cudaStream_t stream);

void PackSm90MxFp4Fp8FoldedQParams(uint8_t*                     dst,
                                   const uint8_t*               src,
                                   int                          output_dim,
                                   int                          group_count,
                                   cudaStream_t                 stream,
                                   Sm90MxFp4Fp8FoldedPackStats* device_stats = nullptr);

void PackSm90MxFp4Fp8UnfoldedQParams(
    uint8_t* dst, const uint8_t* src, int output_dim, int group_count, cudaStream_t stream);

// Convert compact FP32 B128 scales [K/128, N/128] into one replicated
// 16-byte BF16 record per OUT64 fragment: [K/128, N/64, fragment].
void PackSm90Fp8E4M3Scales(
    bfloat16_t* dst, const float* src, int group_count, int output_pack_count, cudaStream_t stream);

}  // namespace turbomind::gemm
