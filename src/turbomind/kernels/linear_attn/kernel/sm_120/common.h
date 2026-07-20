#pragma once

#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/linear_attn/kernel/tma_desc.h"
#include "src/turbomind/kernels/linear_attn/registry.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/copy_traits_sm90.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/tensor.hpp>
#include <cute/underscore.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

namespace turbomind::linear_attn::delta_rule {
namespace {

constexpr int kChunk32Size = 32;
constexpr int kHeadDim     = 128;

constexpr int kFusedGdrBlockDv           = 64;
constexpr int kContextParallelGdrBlockDv = 32;
constexpr int kFusedGdrHBlockDv          = kFusedGdrBlockDv;

static_assert(kHeadDim % kFusedGdrBlockDv == 0);
static_assert(kHeadDim % kContextParallelGdrBlockDv == 0);

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
