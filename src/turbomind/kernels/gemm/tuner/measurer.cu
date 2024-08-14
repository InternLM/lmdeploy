// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/tuner/cache_utils.h"
#include "src/turbomind/kernels/gemm/tuner/measurer.h"
#include <iostream>

namespace turbomind::gemm {

Measurer::Measurer(std::unique_ptr<StoppingCriterion> stop_criterion): stop_criterion_{std::move(stop_criterion)}
{
    cudaEventCreate(&ev_beg_);
    cudaEventCreate(&ev_end_);
}

Measurer::~Measurer()
{
    cudaEventDestroy(ev_beg_);
    cudaEventDestroy(ev_end_);
    ev_beg_ = ev_end_ = {};
}

std::vector<Measurement>
Measurer::Measure(const std::vector<LaunchSpec>& specs, const Launcher& launcher, cudaStream_t stream)
{
    std::vector<Measurement> m;
    m.reserve(specs.size());
    for (const auto& spec : specs) {
        auto measure = MeasureOne(spec, launcher, stream);
        if (measure.sample_count) {
            m.push_back(measure);
        }
        /// TODO: report error
    }
    return m;
}

Measurement Measurer::MeasureOne(LaunchSpec spec, const Launcher& launcher, cudaStream_t stream)
{
    Stats       stats{};
    cudaError_t status = cudaSuccess;
    while (true) {
        float ms{};
        std::tie(ms, status) = ColdRun(spec, launcher, stream);
        if (status != cudaSuccess) {
            break;
        }
        stats.add_sample(ms);
        // std::cout << spec.kernel->name() << " " << spec.swizzle << " " << stats.count() << " " << stats.mean() << " "
        //           << stats.get_variance() << "\n";
        if (stop_criterion_->should_stop(stats)) {
            break;
        }
    }
    return Measurement{
        status,
        stats.count(),
        stats.mean(),
        stats.get_variance(),
    };
}

std::pair<float, cudaError_t> Measurer::ColdRun(LaunchSpec spec, const Launcher& launcher, cudaStream_t stream)
{
    CacheFlushing::flush(stream);

    cudaEventRecord(ev_beg_, stream);

    launcher(spec, stream);

    cudaEventRecord(ev_end_, stream);
    cudaEventSynchronize(ev_end_);

    const auto status = cudaGetLastError();
    float      ms{};

    if (status == cudaSuccess) {
        cudaEventElapsedTime(&ms, ev_beg_, ev_end_);
    }

    return {ms, status};
}

}  // namespace turbomind::gemm
