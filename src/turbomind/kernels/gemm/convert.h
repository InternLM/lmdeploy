// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <utility>
#include <vector>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind {
class LinearWeight;
}

namespace turbomind::gemm {

struct WeightBridge;

struct LayoutConverter {

    Order order;
    Pack  pack;

    virtual int Convert(const void*         S,  //
                        const MatrixLayout& Sdesc,
                        void*               D,
                        MatrixLayout&       Ddesc,
                        cudaStream_t        stream) const = 0;
};

template<class Arch, Order order, uint32_t pack, class Stype, class Dtype>
const LayoutConverter& GetImpl();

// TM_GEMM_WEIGHT_PACK: unset/-1 = auto, 0 = force plain, 1 = force pack
int WeightPackEnv();

void ApplyWeightBridge(LinearWeight&, const WeightBridge&, cudaStream_t);
void PackWeight(LinearWeight&, const LayoutConverter&, cudaStream_t);
void PackWeight(LinearWeight&, Pack, void (*)(uint32_t*, const uint16_t*, int, int, cudaStream_t), cudaStream_t);
void PackQParams(LinearWeight&, const LayoutConverter&, QuantDesc, cudaStream_t);
void PackQParams(
    LinearWeight&, QuantDesc, Pack, void (*)(uint8_t*, const uint8_t*, int, int, cudaStream_t), cudaStream_t);

// Free with `cudaFree`
void* MakeStridedPtrs(const std::vector<std::pair<void*, int>>& ptrs, cudaStream_t stream);

}  // namespace turbomind::gemm
