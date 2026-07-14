// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/tuner/stopping_criterion.h"
#include <climits>
#include <functional>
#include <memory>
#include <vector>

namespace turbomind::gemm {

struct Measurement {
    cudaError_t status;
    int         sample_count;
    float       mean;
    float       variance;
};

using Launcher = std::function<int(LaunchSpec, cudaStream_t)>;

class Measurer {
public:
    Measurer(std::unique_ptr<StoppingCriterion> stop_criterion);

    ~Measurer();

    std::vector<Measurement>
    Measure(const std::vector<LaunchSpec>& specs, const Launcher& launcher, cudaStream_t stream);

private:
    Measurement MeasureOne(LaunchSpec spec, const Launcher& launcher, cudaStream_t stream);

    std::pair<float, cudaError_t> ColdRun(LaunchSpec spec, const Launcher& launcher, cudaStream_t stream);

private:
    cudaEvent_t                        ev_beg_;
    cudaEvent_t                        ev_end_;
    std::unique_ptr<StoppingCriterion> stop_criterion_;
};

}  // namespace turbomind::gemm
