// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/tuner/measurer.h"

#include <vector>

namespace turbomind::gemm {

class Sampler {
public:
    explicit Sampler(Measurer& measurer, int k_clusters): measurer_{measurer}, k_clusters_{k_clusters} {}

    std::vector<LaunchSpec> Run(std::vector<LaunchSpec> specs, const Launcher& launcher, cudaStream_t stream);

private:
    Measurer& measurer_;
    int       k_clusters_;
};

}  // namespace turbomind::gemm
