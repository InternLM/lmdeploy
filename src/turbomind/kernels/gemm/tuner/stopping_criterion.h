// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/tuner/stats.h"
#include <memory>

namespace turbomind::gemm {

class StoppingCriterion {
public:
    virtual ~StoppingCriterion()                 = default;
    virtual bool should_stop(const Stats& stats) = 0;
};

std::unique_ptr<StoppingCriterion> CreateStoppingCriterion(int min_iter, int max_iter, float max_ms);

}  // namespace turbomind::gemm
