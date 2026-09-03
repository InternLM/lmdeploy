// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/cublas.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/gemm_universal_sm90_bf16.h"
#include "src/turbomind/kernels/gemm/kernel/floating_point.h"
#include "src/turbomind/kernels/gemm/kernel_impl_sm90_bf16.h"
#include "src/turbomind/kernels/gemm/sm90_bf16_traits.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

namespace {
void pack(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    TM_CHECK_EQ(linear.weight.dtype(), kBfloat16);
    Tensor trans{{linear.weight.shape(1), linear.weight.shape(0)}, kBfloat16, kDEVICE};
    invokeTransposeAxis01(static_cast<nv_bfloat16*>(trans.raw_data()),
                          static_cast<nv_bfloat16*>(linear.weight.raw_data()),
                          linear.weight.shape(0),
                          linear.weight.shape(1),
                          1,
                          stream);
    linear.weight        = std::move(trans);
    linear.k_desc        = MatrixLayout{kBfloat16,
                                        kColMajor,
                                        linear.input_dim,
                                        linear.output_dim,
                                        (int)linear.weight.stride(0),
                                        0,
                                        0,
                                        nullptr,
                                        nullptr};
    linear.q_desc        = {};
    linear.weight_format = kBfloat16;
}

const Family bf16{
    27, 250, kBfloat16, kBfloat16, 1, 1, 1, 1, true, true, supports_fp<kBfloat16>, pack, 64, kBfloat16};

// Registers one SM90 BF16 GMMA kernel: cluster (1,1), no multicast. Grouped-ness follows
// striding: dense (kFlat) is ungrouped, MoE (kIndexed / kBlocked) is grouped. Tile_ carries
// WGLayout follows tile (M, N): WG_1x2 splits OUT, WG_2x1 splits BATCH.
template<Order raster, Striding striding, class Tile, bool silu = false, int l2_hint_w = 0>
void add_kernel(Collector& c)
{
    constexpr bool grouped = striding != Striding::kFlat;
    c.add<KernelImplSm90Bf16<GemmUniversalSm90_Bf16<raster, 1, 1, grouped, striding, Tile, silu, l2_hint_w>>>();
}

Registrar reg(bf16, [](Collector& c) {
    add_cublas(c, Sm90::is_compatible);

    // Catalog pruned per full-suite scan tmp/sm90_bf16_scan5; refs refreshed per
    // tmp/sm90_bf16_scan8 (2026-07-26, H200, TP/EP 1/2/4/8, swizzle 0-3). `refs: N` =
    // dispatch records (tuned selections) the kernel accumulated across the scan;
    // `// unused` entries had zero refs and are kept visible for re-enabling.
    // Only cluster (1,1) was ever selected; (2,1) / (1,2) variants were dropped entirely.

    // --- Dense (kFlat), row raster ---
    // Legacy N128 tiles never selected. The new 384x128 WG_1x2 kernel is the
    // measured large dense winner; the other validated candidates remain visible.
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_8x128_1x2>(c);   // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_16x128_1x2>(c);  // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_32x128_1x2>(c);  // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_64x128_1x2>(c);  // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_96x128_1x2>(c);  // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_128x128_1x2>(c); // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_192x128_1x2>(c); // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_224x128_1x2>(c); // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_256x128_1x2>(c); // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_320x128_1x2>(c); // validated, slower
    add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_384x128_1x2>(c);
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_320x128_2x1>(c); // validated, slower
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_384x128_2x1>(c); // validated, slower
    add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_8x256_1x2, true>(c);    // refs: 274
    add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_8x256_1x1, true>(c);    // refs: 1267
    add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_16x256_2x1, true>(c);   // refs: 21
    add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_32x256_2x1, true>(c);   // refs: 235
    add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_64x256_2x1, true>(c);   // refs: 268
    add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_96x256_2x1, true>(c);   // refs: 295
    add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_128x256_2x1, true>(c);  // refs: 1160
    add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_160x256_1x2, true>(c);
    add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_192x256_1x2, true>(c);

    // --- Dense (kFlat), col raster: never selected ---
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_8x128_1x2>(c);   // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_16x128_1x2>(c);  // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_32x128_1x2>(c);  // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_64x128_1x2>(c);  // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_96x128_1x2>(c);  // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_128x128_1x2>(c); // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_192x128_1x2>(c); // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_224x128_1x2>(c); // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_256x128_1x2>(c); // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_8x256_1x2, true>(c);   // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_8x256_1x1, true>(c);   // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_16x256_2x1, true>(c);  // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_32x256_2x1, true>(c);  // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_64x256_2x1, true>(c);  // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_96x256_2x1, true>(c);  // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_128x256_2x1, true>(c); // unused

    // --- Dense N128 WG_2x1 (either raster): never selected ---
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_64x128_2x1>(c);  // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_128x128_2x1>(c); // unused

    // --- MoE gate_up (kIndexed), col raster ---
    add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_8x128_1x2>(c);  // refs: 1436
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_16x128_1x2>(c);  // unused
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_32x128_1x2>(c);  // unused
    add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_64x128_1x2>(c);   // refs: 846
    add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_96x128_1x2>(c);   // refs: 183
    add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_128x128_1x2>(c);  // refs: 665
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_192x128_1x2>(c); // unused
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_224x128_1x2>(c); // unused
    add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_256x128_1x2>(c);        // refs: 2207
    add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_8x256_1x2, true>(c);    // refs: 399
    add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_8x256_1x1, true>(c);    // refs: 1362
    add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_16x256_2x1, true>(c);   // refs: 1035
    add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_32x256_2x1, true>(c);   // refs: 495
    add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_64x256_2x1, true>(c);   // refs: 803
    add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_96x256_2x1, true>(c);   // refs: 604
    add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_128x256_2x1, true>(c);  // refs: 2420
    add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_160x256_1x2, true>(c);

    // --- MoE gate_up (kIndexed), row raster ---
    // add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_8x128_1x2>(c);   // unused
    // add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_16x128_1x2>(c);  // unused
    // add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_32x128_1x2>(c);  // unused
    // add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_64x128_1x2>(c);  // unused
    // add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_96x128_1x2>(c);  // unused
    // add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_128x128_1x2>(c); // unused
    // add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_192x128_1x2>(c); // unused
    // add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_224x128_1x2>(c); // unused
    add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_256x128_1x2>(c);  // refs: 1200
    // add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_8x256_1x2, true>(c);   // unused
    // add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_8x256_1x1, true>(c);   // unused
    // add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_16x256_2x1, true>(c);  // unused
    // add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_32x256_2x1, true>(c);  // unused
    // add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_64x256_2x1, true>(c);  // unused
    // add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_96x256_2x1, true>(c);  // unused
    // add_kernel<kRowMajor, Striding::kIndexed, Sm90Bf16Tile_128x256_2x1, true>(c); // unused

    // --- MoE gate_up N128 WG_2x1 / N256 WG_1x2 (either raster): never selected ---
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_64x128_2x1>(c);  // unused
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_128x128_2x1>(c); // unused
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_64x256_1x2>(c);  // unused
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_128x256_1x2>(c); // unused

    // --- MoE down: reuse kIndexed except for 192x256, whose gather path cannot
    // reserve enough math registers without C7512 or spills. ---
    add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_192x256_1x2>(c);
#if 0
    // Dedicated kBlocked kernels.
    add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_8x128_1x2>(c);   // refs: 2637
    add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_16x128_1x2>(c);  // refs: 1419
    add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_32x128_1x2>(c);  // refs: 1375
    // add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_64x128_1x2>(c);  // unused
    add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_96x128_1x2>(c);  // refs: 740
    // add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_128x128_1x2>(c); // unused
    // add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_192x128_1x2>(c); // unused
    // add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_224x128_1x2>(c); // unused
    // add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_256x128_1x2>(c); // unused

    // --- MoE down (kBlocked), row raster: never selected ---
    // add_kernel<kRowMajor, Striding::kBlocked, Sm90Bf16Tile_8x128_1x2>(c);   // unused
    // add_kernel<kRowMajor, Striding::kBlocked, Sm90Bf16Tile_16x128_1x2>(c);  // unused
    // add_kernel<kRowMajor, Striding::kBlocked, Sm90Bf16Tile_32x128_1x2>(c);  // unused
    // add_kernel<kRowMajor, Striding::kBlocked, Sm90Bf16Tile_96x128_1x2>(c);  // unused
    // add_kernel<kRowMajor, Striding::kBlocked, Sm90Bf16Tile_192x128_1x2>(c); // unused
    // add_kernel<kRowMajor, Striding::kBlocked, Sm90Bf16Tile_224x128_1x2>(c); // unused
    // add_kernel<kRowMajor, Striding::kBlocked, Sm90Bf16Tile_256x128_1x2>(c); // unused

    // --- MoE down N128 WG_2x1 (either raster): never selected ---
    // add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_64x128_2x1>(c);  // unused
    // add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_128x128_2x1>(c); // unused
#endif

    // --- Weight L2 evict-first hint (l2_hint_w=1, desc policy_b=1), 1-CTA tiles only ---
    // On the 2-CTA small tiles (8/16/32x128, 16x256) the hint measured 5-9% worse on H200.
    // Only indexed 256x128 was selected; everything else pruned per the scan.
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_64x128_1x2, false, 1>(c);  // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_96x128_1x2, false, 1>(c);  // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_128x128_1x2, false, 1>(c); // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_192x128_1x2, false, 1>(c); // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_224x128_1x2, false, 1>(c); // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_256x128_1x2, false, 1>(c); // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_32x256_2x1, true, 1>(c);   // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_64x256_2x1, true, 1>(c);   // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_96x256_2x1, true, 1>(c);   // unused
    // add_kernel<kRowMajor, Striding::kFlat, Sm90Bf16Tile_128x256_2x1, true, 1>(c);  // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_64x128_1x2, false, 1>(c);  // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_96x128_1x2, false, 1>(c);  // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_128x128_1x2, false, 1>(c); // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_192x128_1x2, false, 1>(c); // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_224x128_1x2, false, 1>(c); // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_256x128_1x2, false, 1>(c); // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_32x256_2x1, true, 1>(c);   // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_64x256_2x1, true, 1>(c);   // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_96x256_2x1, true, 1>(c);   // unused
    // add_kernel<kColMajor, Striding::kFlat, Sm90Bf16Tile_128x256_2x1, true, 1>(c);  // unused
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_64x128_1x2, false, 1>(c);  // unused
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_96x128_1x2, false, 1>(c);  // unused
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_128x128_1x2, false, 1>(c); // unused
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_192x128_1x2, false, 1>(c); // unused
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_224x128_1x2, false, 1>(c); // unused
    add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_256x128_1x2, false, 1>(c);  // refs: 1543
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_32x256_2x1, true, 1>(c);   // unused
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_64x256_2x1, true, 1>(c);   // unused
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_96x256_2x1, true, 1>(c);   // unused
    // add_kernel<kColMajor, Striding::kIndexed, Sm90Bf16Tile_128x256_2x1, true, 1>(c);  // unused
    // add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_96x128_1x2, false, 1>(c);  // unused
    // add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_192x128_1x2, false, 1>(c); // unused
    // add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_224x128_1x2, false, 1>(c); // unused
    // add_kernel<kColMajor, Striding::kBlocked, Sm90Bf16Tile_256x128_1x2, false, 1>(c); // unused
});
}  // namespace

}  // namespace turbomind::gemm
