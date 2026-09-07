// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/kernel/config.h"

namespace turbomind::gemm::config::geometry {

// Shared tile and thread-group geometry for the legacy GEMM catalogs.

// Weight as A: M = output, N = batch.
// Output tile: 64
using _64x8x64_4x1x1 = Config<Shape<64, 8, 64>, Shape<4, 1, 1>>;
using _64x8x128_4x1x1 = Config<Shape<64, 8, 128>, Shape<4, 1, 1>>;
using _64x16x64_4x1x1 = Config<Shape<64, 16, 64>, Shape<4, 1, 1>>;

// Output tile: 128
using _128x8x32_4x1x1 = Config<Shape<128, 8, 32>, Shape<4, 1, 1>>;
using _128x8x64_4x1x1 = Config<Shape<128, 8, 64>, Shape<4, 1, 1>>;
using _128x16x32_4x1x1 = Config<Shape<128, 16, 32>, Shape<4, 1, 1>>;
using _128x16x64_4x1x1 = Config<Shape<128, 16, 64>, Shape<4, 1, 1>>;
using _128x32x32_4x1x1 = Config<Shape<128, 32, 32>, Shape<4, 1, 1>>;
using _128x64x32_4x1x1 = Config<Shape<128, 64, 32>, Shape<4, 1, 1>>;
using _128x96x32_4x1x1 = Config<Shape<128, 96, 32>, Shape<4, 1, 1>>;
using _128x128x32_4x1x1 = Config<Shape<128, 128, 32>, Shape<4, 1, 1>>;

// Output tile: 256
using _256x8x32_4x1x1 = Config<Shape<256, 8, 32>, Shape<4, 1, 1>>;
using _256x8x64_4x1x1 = Config<Shape<256, 8, 64>, Shape<4, 1, 1>>;
using _256x32x32_4x1x1 = Config<Shape<256, 32, 32>, Shape<4, 1, 1>>;
using _256x32x64_4x1x1 = Config<Shape<256, 32, 64>, Shape<4, 1, 1>>;
using _256x64x32_4x1x1 = Config<Shape<256, 64, 32>, Shape<4, 1, 1>>;
using _256x128x32_8x1x1 = Config<Shape<256, 128, 32>, Shape<8, 1, 1>>;

// Weight as B: M = batch, N = output.
// Output tile: 64
using _16x64x64_1x2x2 = Config<Shape<16, 64, 64>, Shape<1, 2, 2>>;
using _16x64x128_1x2x2 = Config<Shape<16, 64, 128>, Shape<1, 2, 2>>;
using _32x64x64_1x2x2 = Config<Shape<32, 64, 64>, Shape<1, 2, 2>>;
using _32x64x128_1x2x2 = Config<Shape<32, 64, 128>, Shape<1, 2, 2>>;
using _48x64x64_1x2x2 = Config<Shape<48, 64, 64>, Shape<1, 2, 2>>;
using _48x64x128_1x2x2 = Config<Shape<48, 64, 128>, Shape<1, 2, 2>>;
using _64x64x64_1x2x2 = Config<Shape<64, 64, 64>, Shape<1, 2, 2>>;
using _64x64x64_2x2x1 = Config<Shape<64, 64, 64>, Shape<2, 2, 1>>;
using _64x64x128_1x2x2 = Config<Shape<64, 64, 128>, Shape<1, 2, 2>>;
using _96x64x32_2x2x1 = Config<Shape<96, 64, 32>, Shape<2, 2, 1>>;
using _96x64x64_1x2x2 = Config<Shape<96, 64, 64>, Shape<1, 2, 2>>;
using _96x64x64_2x2x1 = Config<Shape<96, 64, 64>, Shape<2, 2, 1>>;

// Output tile: 128
using _8x128x32_1x4x1 = Config<Shape<8, 128, 32>, Shape<1, 4, 1>>;
using _8x128x64_1x4x1 = Config<Shape<8, 128, 64>, Shape<1, 4, 1>>;
using _8x128x128_1x4x1 = Config<Shape<8, 128, 128>, Shape<1, 4, 1>>;
using _16x128x32_1x4x1 = Config<Shape<16, 128, 32>, Shape<1, 4, 1>>;
using _16x128x64_1x4x1 = Config<Shape<16, 128, 64>, Shape<1, 4, 1>>;
using _16x128x128_1x4x2 = Config<Shape<16, 128, 128>, Shape<1, 4, 2>>;
using _32x128x32_1x4x1 = Config<Shape<32, 128, 32>, Shape<1, 4, 1>>;
using _32x128x64_1x4x1 = Config<Shape<32, 128, 64>, Shape<1, 4, 1>>;
using _32x128x128_1x4x2 = Config<Shape<32, 128, 128>, Shape<1, 4, 2>>;
using _48x128x32_1x4x1 = Config<Shape<48, 128, 32>, Shape<1, 4, 1>>;
using _48x128x64_1x4x1 = Config<Shape<48, 128, 64>, Shape<1, 4, 1>>;
using _48x128x128_1x4x2 = Config<Shape<48, 128, 128>, Shape<1, 4, 2>>;
using _64x128x16_1x4x1 = Config<Shape<64, 128, 16>, Shape<1, 4, 1>>;
using _64x128x32_1x4x1 = Config<Shape<64, 128, 32>, Shape<1, 4, 1>>;
using _64x128x32_2x2x1 = Config<Shape<64, 128, 32>, Shape<2, 2, 1>>;
using _64x128x64_1x4x1 = Config<Shape<64, 128, 64>, Shape<1, 4, 1>>;
using _64x128x128_1x4x2 = Config<Shape<64, 128, 128>, Shape<1, 4, 2>>;
using _96x128x32_1x4x1 = Config<Shape<96, 128, 32>, Shape<1, 4, 1>>;
using _96x128x32_2x2x1 = Config<Shape<96, 128, 32>, Shape<2, 2, 1>>;
using _96x128x128_1x4x2 = Config<Shape<96, 128, 128>, Shape<1, 4, 2>>;
using _128x128x16_2x2x1 = Config<Shape<128, 128, 16>, Shape<2, 2, 1>>;
using _128x128x32_1x4x1 = Config<Shape<128, 128, 32>, Shape<1, 4, 1>>;
using _128x128x32_2x2x1 = Config<Shape<128, 128, 32>, Shape<2, 2, 1>>;
using _128x128x64_1x4x2 = Config<Shape<128, 128, 64>, Shape<1, 4, 2>>;
using _128x128x64_2x2x1 = Config<Shape<128, 128, 64>, Shape<2, 2, 1>>;
using _256x128x16_4x2x1 = Config<Shape<256, 128, 16>, Shape<4, 2, 1>>;
using _256x128x64_4x2x1 = Config<Shape<256, 128, 64>, Shape<4, 2, 1>>;

// Output tile: 256
using _8x256x32_1x4x1 = Config<Shape<8, 256, 32>, Shape<1, 4, 1>>;
using _8x256x64_1x4x1 = Config<Shape<8, 256, 64>, Shape<1, 4, 1>>;
using _16x256x32_1x4x1 = Config<Shape<16, 256, 32>, Shape<1, 4, 1>>;
using _16x256x64_1x4x1 = Config<Shape<16, 256, 64>, Shape<1, 4, 1>>;
using _32x256x32_1x4x1 = Config<Shape<32, 256, 32>, Shape<1, 4, 1>>;
using _32x256x64_1x4x1 = Config<Shape<32, 256, 64>, Shape<1, 4, 1>>;
using _48x256x64_1x4x1 = Config<Shape<48, 256, 64>, Shape<1, 4, 1>>;
using _64x256x16_1x4x1 = Config<Shape<64, 256, 16>, Shape<1, 4, 1>>;
using _64x256x32_1x4x1 = Config<Shape<64, 256, 32>, Shape<1, 4, 1>>;
using _96x256x32_1x8x1 = Config<Shape<96, 256, 32>, Shape<1, 8, 1>>;
using _128x256x16_2x4x1 = Config<Shape<128, 256, 16>, Shape<2, 4, 1>>;
using _128x256x32_1x8x1 = Config<Shape<128, 256, 32>, Shape<1, 8, 1>>;
using _128x256x32_2x4x1 = Config<Shape<128, 256, 32>, Shape<2, 4, 1>>;
using _128x256x64_1x8x1 = Config<Shape<128, 256, 64>, Shape<1, 8, 1>>;
using _128x256x64_2x4x1 = Config<Shape<128, 256, 64>, Shape<2, 4, 1>>;

}  // namespace turbomind::gemm::config::geometry
