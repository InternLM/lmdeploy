// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include <istream>
#include <ostream>

namespace turbomind {

template<typename T>
class LlamaLinear {
public:
    enum Type
    {
        kGemm,
        kFusedSiluFfn,
        kFusedAdd
    };

    struct Pitched {
        const T* ptr;
        int      pitch;
        Pitched(const T* ptr, int pitch = 0): ptr{ptr}, pitch{pitch} {}
    };

    LlamaLinear(cublasMMWrapper* cublas_wrapper, cudaStream_t stream, IAllocator* allocator);

    void forward(T*                         output_data,
                 Pitched                    input_data,
                 int8_t*                    quant_input_data,
                 float*                     quant_scale,
                 int                        batch_size,
                 const LlamaDenseWeight<T>& weight,
                 Type                       type      = kGemm,
                 int*                       lora_mask = nullptr);

    void set_measure(bool measure);

    [[maybe_unused]] int Export(std::ostream& os);

    [[maybe_unused]] int Import(std::istream& is);

    std::pair<int*, int*> getQQQBuffer();

    void setQQQBuffer(int* reduce_buf, int* workspace_buf);

    std::vector<int> GetTuningSeq() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace turbomind
