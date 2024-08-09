// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>

namespace turbomind::gemm {

class CacheFlushing {
public:
    static void flush(cudaStream_t stream = {});

private:
    CacheFlushing();
    void operator()(cudaStream_t stream) const;

    uint32_t* buffer_;
    size_t    size_;
};

}  // namespace turbomind::gemm
