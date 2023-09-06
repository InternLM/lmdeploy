// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "metric.h"
#include "src/turbomind/macro.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace turbomind {

extern bool g_dump_kernel_info_once;

class GemmS4F16 {
public:
    GemmS4F16();

    ~GemmS4F16();

    enum Type
    {
        kGemm,
        kFusedSiluFfn
    };

    void Measure(half*                C,
                 const uint*          A,
                 const half*          B,
                 const half2*         Q,
                 int                  m,
                 int                  n,
                 int                  k,
                 int                  group_size,
                 Type                 type,
                 std::vector<Metric>& metrics,
                 cudaStream_t         st);

    void Run(half*        C,
             const uint*  A,
             const half*  B,
             const half2* Q,
             int          m,
             int          n,
             int          k,
             int          group_size,
             Type         type,
             int          algo_id,
             cudaStream_t st);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
