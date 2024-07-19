// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/utils/cublasMMWrapper.h"

namespace turbomind {

template<typename T>
class LlamaLinear {
public:
    enum Type {
        kGemm,
        kFusedSiluFfn,
        kFusedAdd
    };

    struct Pitched {
        const T* ptr;
        int      pitch;
        Pitched(const T* ptr, int pitch = 0): ptr{ptr}, pitch{pitch} {}
    };

    LlamaLinear(cublasMMWrapper* cublas_wrapper, cudaStream_t stream);

    void forward(T*                         output_data,
                 Pitched                    input_data,
                 int                        batch_size,
                 const LlamaDenseWeight<T>& weight,
                 Type                       type      = kGemm,
                 int*                       lora_mask = nullptr);

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace turbomind
