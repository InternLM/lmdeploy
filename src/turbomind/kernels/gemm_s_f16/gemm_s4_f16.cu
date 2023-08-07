// Copyright (c) OpenMMLab. All rights reserved.

#include "gemm_s4_f16.h"
#include "gemm_s4_f16_kernel.h"
#include "metric.h"
#include <algorithm>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace turbomind {

bool g_dump_kernel_info_once = false;

struct GemmS4F16::Impl {

    template<int GS>
    void Generate()
    {
        std::vector<std::unique_ptr<IGemmKernel>> k;

        // k.emplace_back(new GemmKernel<Shape<256, 128, 32>, Shape<32, 128, 32>, 5, GS>{});
        // k.emplace_back(new GemmKernel<Shape<256, 128, 32>, Shape<32, 128, 32>, 4, GS>{});
        // k.emplace_back(new GemmKernel<Shape<256, 128, 32>, Shape<32, 128, 32>, 3, GS>{});
        // k.emplace_back(new GemmKernel<Shape<256, 128, 32>, Shape<32, 128, 32>, 2, GS>{});

        k.emplace_back(new GemmKernel<Shape<128, 128, 64>, Shape<32, 128, 32>, 4, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 128, 64>, Shape<32, 128, 32>, 3, GS>{});
        // k.emplace_back(new GemmKernel<Shape<128, 128, 64>, Shape<32, 128, 32>, 2, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 96, 32>, Shape<32, 96, 32>, 5, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 96, 32>, Shape<32, 96, 32>, 4, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 96, 32>, Shape<32, 96, 32>, 3, GS>{});
        // k.emplace_back(new GemmKernel<Shape<128, 96, 32>, Shape<32, 96, 32>, 2, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 64, 32>, Shape<32, 64, 32>, 5, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 64, 32>, Shape<32, 64, 32>, 4, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 64, 32>, Shape<32, 64, 32>, 3, GS>{});
        // k.emplace_back(new GemmKernel<Shape<128, 64, 32>, Shape<32, 64, 32>, 2, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 64, 64>, Shape<32, 64, 32>, 5, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 64, 64>, Shape<32, 64, 32>, 4, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 64, 64>, Shape<32, 64, 32>, 3, GS>{});
        // k.emplace_back(new GemmKernel<Shape<128, 64, 64>, Shape<32, 64, 32>, 2, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 32, 128>, Shape<32, 32, 64>, 4, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 32, 128>, Shape<32, 32, 64>, 3, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 32, 128>, Shape<32, 32, 64>, 2, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 16, 256>, Shape<32, 16, 64>, 3, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 16, 256>, Shape<32, 16, 64>, 2, GS>{});
        k.emplace_back(new GemmKernel<Shape<128, 8, 512>, Shape<32, 8, 128>, 2, GS>{});

        k.emplace_back(new GemmKernel<Shape<64, 128, 128>, Shape<32, 128, 32>, 3, GS>{});
        k.emplace_back(new GemmKernel<Shape<64, 128, 128>, Shape<32, 128, 32>, 2, GS>{});
        k.emplace_back(new GemmKernel<Shape<64, 96, 128>, Shape<32, 96, 32>, 4, GS>{});
        k.emplace_back(new GemmKernel<Shape<64, 96, 128>, Shape<32, 96, 32>, 3, GS>{});
        // k.emplace_back(new GemmKernel<Shape<64, 96, 128>, Shape<32, 96, 32>, 2, GS>{});
        k.emplace_back(new GemmKernel<Shape<64, 64, 128>, Shape<32, 64, 32>, 4, GS>{});
        k.emplace_back(new GemmKernel<Shape<64, 64, 128>, Shape<32, 64, 32>, 3, GS>{});
        // k.emplace_back(new GemmKernel<Shape<64, 64, 128>, Shape<32, 64, 32>, 2, GS>{});
        k.emplace_back(new GemmKernel<Shape<64, 32, 128>, Shape<32, 32, 32>, 4, GS>{});
        k.emplace_back(new GemmKernel<Shape<64, 32, 128>, Shape<32, 32, 32>, 3, GS>{});
        // k.emplace_back(new GemmKernel<Shape<64, 32, 128>, Shape<32, 32, 32>, 2, GS>{});
        k.emplace_back(new GemmKernel<Shape<64, 16, 256>, Shape<32, 16, 32>, 3, GS>{});
        // k.emplace_back(new GemmKernel<Shape<64, 16, 256>, Shape<32, 16, 32>, 2, GS>{});
        k.emplace_back(new GemmKernel<Shape<64, 8, 512>, Shape<32, 8, 128>, 2, GS>{});
        // k.emplace_back(new GemmKernel<Shape<64, 8, 256>, Shape<32, 8, 32>, 3, GS>{});
        k.emplace_back(new GemmKernel<Shape<64, 8, 256>, Shape<32, 8, 32>, 2, GS>{});

        kernels_.push_back(std::move(k));
        group_sizes_.push_back(GS);
    }

    void Measure(half*                C,
                 const uint*          A,
                 const half*          B,
                 const half2*         Q,
                 int                  m,
                 int                  n,
                 int                  k,
                 int                  group_size,
                 std::vector<Metric>& metrics,
                 cudaStream_t         st)
    {
        int gid = -1;
        for (size_t i = 0; i < group_sizes_.size(); ++i) {
            if (group_sizes_[i] == group_size) {
                gid = i;
                break;
            }
        }
        if (gid < 0) {
            throw std::runtime_error("unsupported group size");
        }
        const auto& kernels = kernels_[gid];
        metrics             = std::vector<Metric>(kernels.size());

        int best = 0;

        for (size_t i = 0; i < kernels.size(); ++i) {
            metrics[i].id = i;
            kernels[i]->GetMetric(&metrics[i], m, n, k);
            if (!metrics[i].feasible) {
                metrics[i].time  = std::numeric_limits<float>::infinity();
                metrics[i].count = 1;
                continue;
            }
            if (Compare(metrics[i], metrics[best])) {
                best = i;
            }
            for (size_t j = 0; j < kWarmup + kMeasure; ++j) {
                if (j == kWarmup) {
                    cudaEventRecord(ev_start_, st);
                }
                kernels[i]->Launch(C, A, B, Q, m, n, k, st);
            }
            cudaEventRecord(ev_end_, st);
            cudaEventSynchronize(ev_end_);
            float ms{};
            cudaEventElapsedTime(&ms, ev_start_, ev_end_);
            metrics[i].time  = ms;
            metrics[i].count = kMeasure;
        }

        metrics[best].best = 1;

        // sort metrics
        std::vector<int> indices(kernels.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::stable_sort(
            indices.begin(), indices.end(), [&](int i, int j) { return metrics[i].time < metrics[j].time; });

        if (g_dump_kernel_info_once) {
            DumpMetrics(std::cerr, metrics, indices);
            g_dump_kernel_info_once = 0;
        }

        std::vector<Metric> tmp;
        for (size_t i = 0; i < indices.size(); ++i) {
            tmp.push_back(metrics[indices[i]]);
        }
        metrics.swap(tmp);
    }

    static bool Compare(const Metric& a, const Metric& b)
    {
        if (a.feasible != b.feasible) {
            return a.feasible > b.feasible;
        }
        if (a.normalized != b.normalized) {
            return a.normalized < b.normalized;
        }
        if (a.cost != b.cost) {
            return a.cost < b.cost;
        }
        if (a.nice != b.nice) {
            return a.nice > b.nice;
        }
        return a.occupancy > b.occupancy;
    }

    int Estimate(int gid, int m, int n, int k)
    {
        const auto& kernels = kernels_.at(gid);

        int                 best = 0;
        std::vector<Metric> metrics(kernels.size());
        for (size_t i = 0; i < kernels.size(); ++i) {
            metrics[i].id = i;
            kernels[i]->GetMetric(&metrics[i], m, n, k);
            if (Compare(metrics[i], metrics[best])) {
                best = i;
            }
        }

        if (g_dump_kernel_info_once) {
            std::vector<int> indices(kernels.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::stable_sort(
                indices.begin(), indices.end(), [&](int i, int j) { return Compare(metrics[i], metrics[j]); });
            DumpMetrics(std::cerr, metrics, indices);
            g_dump_kernel_info_once = 0;
        }

        return best;
    }

    void Run(half*        C,
             const uint*  A,
             const half*  B,
             const half2* Q,
             int          m,
             int          n,
             int          k,
             int          group_size,
             int          algo_id,
             cudaStream_t st)
    {
        for (size_t i = 0; i < group_sizes_.size(); ++i) {
            if (group_sizes_[i] == group_size) {
                if (algo_id < 0) {
                    algo_id = Estimate(i, m, n, k);
                }
                if (algo_id < 0) {
                    throw std::runtime_error("no feasible kernel found");
                }
                kernels_[i].at(algo_id)->Launch(C, A, B, Q, m, n, k, st);
                return;
            }
        }
        throw std::runtime_error("unsupported group size");
    }

    Impl()
    {
        cudaEventCreate(&ev_start_);
        cudaEventCreate(&ev_end_);

        /// TODO: add more group sizes
        Generate<128>();
    }

    ~Impl()
    {
        cudaEventDestroy(ev_end_);
        cudaEventDestroy(ev_start_);
    }

    std::vector<std::vector<std::unique_ptr<IGemmKernel>>> kernels_;

    std::vector<int> group_sizes_;

    static constexpr int kWarmup  = 10;
    static constexpr int kMeasure = 100;

    cudaEvent_t ev_start_{};
    cudaEvent_t ev_end_{};
};

GemmS4F16::GemmS4F16(): impl_(std::make_unique<Impl>()) {}

GemmS4F16::~GemmS4F16() = default;

void GemmS4F16::Measure(half*                C,
                        const uint*          A,
                        const half*          B,
                        const half2*         Q,
                        int                  m,
                        int                  n,
                        int                  k,
                        int                  group_size,
                        std::vector<Metric>& metrics,
                        cudaStream_t         st)
{
    impl_->Measure(C, A, B, Q, m, n, k, group_size, metrics, st);
}

void GemmS4F16::Run(half*        C,
                    const uint*  A,
                    const half*  B,
                    const half2* Q,
                    int          m,
                    int          n,
                    int          k,
                    int          group_size,
                    int          algo_id,
                    cudaStream_t st)
{
    impl_->Run(C, A, B, Q, m, n, k, group_size, algo_id, st);
}

}  // namespace turbomind