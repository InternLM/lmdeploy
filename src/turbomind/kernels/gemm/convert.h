// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <utility>
#include <vector>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

struct LayoutConverter {

    Order order;
    Pack  pack;
    // Optional physical size of a persistent packed element.  Zero means
    // that the packed representation has the public dtype's bit width.
    int   storage_bits{};

    virtual int Convert(const void*         S,  //
                        const MatrixLayout& Sdesc,
                        void*               D,
                        MatrixLayout&       Ddesc,
                        cudaStream_t        stream) const = 0;
};

enum class QParamEncoding
{
    kDefault,
    kBf16ScaleEffZero,
    kBf16BlockScale,
    kMxFp4UnbiasedExponent,
    kMxFp4Fp8Folded,
    kMxFp4Fp8Unfolded,
    kNvFp4Scale,
};

struct ConverterRequest {
    DataType data_type{};
    DataType weight_type{};
    DataType input_type{};
    bool     grouped{};
    int      sm{};
    int      input_dim{};
    int      output_dim{};
    int      group_size{};
    Epilogue epilogue{};
};

struct ConverterSet {
    const LayoutConverter* weight{};
    const LayoutConverter* qparams{};
    QParamEncoding         qparam_encoding{QParamEncoding::kDefault};
};

// Pointers to singletons
ConverterSet GetConverters(const ConverterRequest& request);

// TM_GEMM_WEIGHT_PACK: unset/-1 = auto, 0 = force plain, 1 = force pack
int WeightPackEnv();

// Whether this extension was compiled with the native SM90 prepacked mixed
// precision converters and kernel catalog.
bool HasSm90MixedKernel();

// Free with `cudaFree`
void* MakeStridedPtrs(const std::vector<std::pair<void*, int>>& ptrs, cudaStream_t stream);

}  // namespace turbomind::gemm
