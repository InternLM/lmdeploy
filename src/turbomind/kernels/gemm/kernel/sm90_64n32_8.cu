
#include <cuda.h>

// We need modifiable TMA, which is added in 12.3
#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 3))

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/gemm_universal_sm90_fp8_wa.h"
#include "src/turbomind/kernels/gemm/gemm_universal_sm90_v3.h"
#include "src/turbomind/kernels/gemm/kernel_impl_sm90.h"
#include "src/turbomind/kernels/gemm/sm90_fp8_wa_traits.h"
#include "src/turbomind/kernels/gemm/sm90_v3_traits.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "src/turbomind/kernels/gemm/registrar.h"

namespace turbomind::gemm {

namespace {

// Registers one SM90 v3 (FP8 act-as-A) GMMA kernel. Grouped-ness follows striding:
// dense (kFlat) is ungrouped, MoE (kIndexed / kBlocked) is grouped.
template<Order raster, Striding striding, class Tile, bool silu = false, int mc_a = 1, int mc_b = 1>
void add_v3(Collector& c)
{
    constexpr bool grouped = striding != Striding::kFlat;
    c.add(std::make_unique<KernelImplSm90<GemmUniversalSm90_v3<raster, mc_a, mc_b, grouped, striding, Tile, silu>>>());
}

// Registers one SM90 FP8 weight-as-A GMMA kernel. TILE_M < 64 dense tiles have no
// (2,1) variant: the U multicast box (kBoxU*4/mcA) is not 128B-aligned.
template<Order raster, Striding striding, class Tile, bool silu = false, int mc_a = 1, int mc_b = 1>
void add_wa(Collector& c)
{
    constexpr bool grouped = striding != Striding::kFlat;
    c.add(
        std::make_unique<KernelImplSm90<GemmUniversalSm90_Fp8Wa<raster, mc_a, mc_b, grouped, striding, Tile, silu>>>());
}

Registrar reg([](Collector& c, int /*arch*/) {
    // Catalog pruned per full-suite scans tmp/sm90_fp8wa_scan1 + tmp/sm90_v3_scan1
    // (2026-07-26, H200, TP/EP 1/2/4/8, swizzle 0-3; both FP8 types exercise the same
    // kernel pool, refs combined). `refs: N` = dispatch records (tuned selections);
    // `// unused` entries had zero refs and are kept visible for re-enabling.
    // --- Dense (v3 act-as-A), row raster ---
    add_v3<kRowMajor, Striding::kFlat, Sm90V3Tile_128x192>(c);               // refs: 2319
    add_v3<kRowMajor, Striding::kFlat, Sm90V3Tile_128x192, false, 2>(c);     // refs: 1020
    add_v3<kRowMajor, Striding::kFlat, Sm90V3Tile_128x192, false, 1, 2>(c);  // refs: 1813
    add_v3<kRowMajor, Striding::kFlat, Sm90V3Tile_128x256, true>(c);         // refs: 4648
    add_v3<kRowMajor, Striding::kFlat, Sm90V3Tile_128x256, true, 2>(c);      // refs: 86
    add_v3<kRowMajor, Striding::kFlat, Sm90V3Tile_128x256, true, 1, 2>(c);   // refs: 1331
    add_v3<kRowMajor, Striding::kFlat, Sm90V3Tile_64x256, true>(c);          // refs: 2133
    add_v3<kRowMajor, Striding::kFlat, Sm90V3Tile_64x256, true, 2>(c);       // refs: 76
    add_v3<kRowMajor, Striding::kFlat, Sm90V3Tile_64x256, true, 1, 2>(c);    // refs: 323

    // --- Dense grouped (kBlocked), col raster ---
    add_v3<kColMajor, Striding::kBlocked, Sm90V3Tile_128x192>(c);               // refs: 1247
    add_v3<kColMajor, Striding::kBlocked, Sm90V3Tile_128x192, false, 2>(c);     // refs: 777
    add_v3<kColMajor, Striding::kBlocked, Sm90V3Tile_128x192, false, 1, 2>(c);  // refs: 99

    // --- Indexed (gate/up), col raster: N192 non-fused; N256 supports both epilogues ---
    add_v3<kColMajor, Striding::kIndexed, Sm90V3Tile_128x192>(c);               // refs: 173
    add_v3<kColMajor, Striding::kIndexed, Sm90V3Tile_128x192, false, 2>(c);     // refs: 99
    add_v3<kColMajor, Striding::kIndexed, Sm90V3Tile_128x192, false, 1, 2>(c);  // refs: 37
    add_v3<kColMajor, Striding::kIndexed, Sm90V3Tile_128x256, true>(c);         // refs: 6903
    add_v3<kColMajor, Striding::kIndexed, Sm90V3Tile_128x256, true, 2>(c);      // refs: 49
    add_v3<kColMajor, Striding::kIndexed, Sm90V3Tile_128x256, true, 1, 2>(c);   // refs: 196
    add_v3<kColMajor, Striding::kIndexed, Sm90V3Tile_64x256, true>(c);          // refs: 3824
    add_v3<kColMajor, Striding::kIndexed, Sm90V3Tile_64x256, true, 2>(c);       // refs: 67
    add_v3<kColMajor, Striding::kIndexed, Sm90V3Tile_64x256, true, 1, 2>(c);    // refs: 16

    // --- Weight-as-A FP8, dense (kFlat), row raster ---
    // OUT=128: plain bf16 epilogue. TILE_M<64: no (2,1) (TMA 128B alignment).
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_8x128>(c);                // refs: 5602
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_8x128, false, 1, 2>(c);   // refs: 266
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_16x128>(c);               // refs: 1745
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_16x128, false, 1, 2>(c);  // refs: 66
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_32x128>(c);               // refs: 1516
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_32x128, false, 1, 2>(c);  // refs: 145
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_64x128>(c);               // refs: 1403
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_64x128, false, 2>(c);     // refs: 46
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_64x128, false, 1, 2>(c);  // refs: 211
    // OUT=256: plain bf16 and fused SiLU->FP8 epilogues.
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_8x256, true>(c);         // refs: 1556
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_8x256, true, 1, 2>(c);   // refs: 111
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_16x256, true>(c);        // refs: 329
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_16x256, true, 1, 2>(c);  // refs: 25
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_32x256, true>(c);        // refs: 221
    // add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_32x256, true, 1, 2>(c);  // unused
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_64x256, true>(c);  // refs: 13
    // add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_64x256, true, 2>(c);  // unused
    // add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_64x256, true, 1, 2>(c);  // unused

    // --- Weight-as-A FP8, blocked, col raster ---
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_8x128>(c);  // refs: 37
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_8x128, false, 2>(c);  // unused
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_8x128, false, 1, 2>(c);  // unused
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_16x128>(c);  // refs: 36
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_16x128, false, 2>(c);  // unused
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_16x128, false, 1, 2>(c);  // unused
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_32x128>(c);  // refs: 63
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_32x128, false, 2>(c);  // unused
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_32x128, false, 1, 2>(c);  // unused
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_64x128>(c);  // refs: 10
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_64x128, false, 2>(c);  // unused
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_64x128, false, 1, 2>(c);  // unused
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_8x256>(c);            // refs: 1377
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_8x256, false, 2>(c);  // refs: 2
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_8x256, false, 1, 2>(c);  // unused
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_16x256>(c);  // refs: 439
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_16x256, false, 2>(c);  // unused
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_16x256, false, 1, 2>(c);  // unused
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_32x256>(c);  // refs: 87
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_32x256, false, 2>(c);  // unused
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_32x256, false, 1, 2>(c);  // unused
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_64x256>(c);  // unused
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_64x256, false, 2>(c);  // unused
    // add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_64x256, false, 1, 2>(c);  // unused

    // --- Weight-as-A FP8, indexed (gate/up), col raster ---
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_8x128>(c);  // refs: 570
    // add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_8x128, false, 2>(c);  // unused
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_8x128, false, 1, 2>(c);  // refs: 20
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_16x128>(c);              // refs: 157
    // add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_16x128, false, 2>(c);  // unused
    // add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_16x128, false, 1, 2>(c);  // unused
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_32x128>(c);  // refs: 96
    // add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_32x128, false, 2>(c);  // unused
    // add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_32x128, false, 1, 2>(c);  // unused
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_64x128>(c);  // refs: 36
    // add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_64x128, false, 2>(c);  // unused
    // add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_64x128, false, 1, 2>(c);  // unused
    // Indexed OUT=256 kernels also select the epilogue at runtime.
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_8x256, true>(c);        // refs: 4553
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_8x256, true, 2>(c);     // refs: 32
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_8x256, true, 1, 2>(c);  // refs: 4
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_16x256, true>(c);       // refs: 1253
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_16x256, true, 2>(c);    // refs: 7
    // add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_16x256, true, 1, 2>(c);  // unused
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_32x256, true>(c);  // refs: 340
    // add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_32x256, true, 2>(c);  // unused
    // add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_32x256, true, 1, 2>(c);  // unused
    // add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_64x256, true>(c);  // unused
    // add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_64x256, true, 2>(c);  // unused
    // add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_64x256, true, 1, 2>(c);  // unused

    // --- 2 math WGs (per-WG BATCH=64): 128x128 plain, 128x256 fused, 64x256_n2 ---
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_64x256_n2, true>(c);
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_64x256_n2, true, 2>(c);
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_64x256_n2, true, 1, 2>(c);
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_128x128>(c);
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_128x128, false, 2>(c);
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_128x128, false, 1, 2>(c);
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_128x256, true>(c);
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_128x256, true, 2>(c);
    add_wa<kRowMajor, Striding::kFlat, Sm90Fp8WaTile_128x256, true, 1, 2>(c);
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_128x128>(c);
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_128x128, false, 2>(c);
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_128x128, false, 1, 2>(c);
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_128x256>(c);
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_128x256, false, 2>(c);
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_128x256, false, 1, 2>(c);
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_64x256_n2>(c);
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_64x256_n2, false, 2>(c);
    add_wa<kColMajor, Striding::kBlocked, Sm90Fp8WaTile_64x256_n2, false, 1, 2>(c);
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_128x128>(c);
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_128x128, false, 2>(c);
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_128x128, false, 1, 2>(c);
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_128x256, true>(c);
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_128x256, true, 2>(c);
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_128x256, true, 1, 2>(c);
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_64x256_n2, true>(c);
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_64x256_n2, true, 2>(c);
    add_wa<kColMajor, Striding::kIndexed, Sm90Fp8WaTile_64x256_n2, true, 1, 2>(c);
});
}  // namespace

}  // namespace turbomind::gemm

#else

// CUDA too old for modifiable TMA: no SM90 GMMA kernels from this TU.

#endif
