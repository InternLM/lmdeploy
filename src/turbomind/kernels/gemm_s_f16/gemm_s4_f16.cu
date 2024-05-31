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
#include <tuple>
#include <vector>

namespace turbomind {

bool g_dump_kernel_info_once = false;

namespace ops {

struct Identity {
    static __inline__ __device__ void apply(uint data, int m, int n, half* C, int M, int N)
    {
        if (n < N) {
            (uint&)C[n * M + m] = (uint&)data;
        }
    }
};

struct SiluActivation {
    static __inline__ __device__ void apply(uint data, int m, int n, half* C, int M, int N)
    {
        auto  u    = __half22float2((half2&)data);
        float silu = u.x / (1.f + __expf(-u.x));
        half  val  = __float2half_rn(silu * u.y);

        if (n < N) {
            C[n * (M / 2) + m / 2] = val;
        }
    }
};

struct Add {
    static __inline__ __device__ void apply(uint data, int m, int n, half* C, int M, int N)
    {
        if (n < N) {
            C[n * M + m] += ((half2&)data).x;
            C[n * M + m + 1] += ((half2&)data).y;
        }
    }
};

}  // namespace ops

template<typename... Ts>
struct OutputOps {

    template<int index>
    static __inline__ __device__ void apply(uint data, int m, int n, half* C, int M, int N)
    {
        std::tuple_element_t<index, std::tuple<Ts...>>::apply(data, m, n, C, M, N);
    }
};

struct GemmS4F16::Impl {

    using Kernels = std::vector<std::unique_ptr<IGemmKernel>>;

    template<int GS, typename Op>
    void Generate(std::vector<Kernels>& kernels)
    {
        // smem size (KB):
        // sm75: 64
        // sm80: 163
        // sm86: 99
        // sm89: 99
        // sm90: 227

        Kernels k;

        // 256
        k.emplace_back(new GemmKernel<Shape<256, 128, 32>, Shape<32, 128, 32>, 3, GS, Op>{});
        k.emplace_back(new GemmKernel<Shape<256, 64, 64>, Shape<64, 64, 32>, 3, GS, Op>{});
        k.emplace_back(new GemmKernel<Shape<256, 64, 32>, Shape<64, 64, 32>, 3, GS, Op>{});
        k.emplace_back(new GemmKernel<Shape<256, 32, 64>, Shape<64, 32, 32>, 3, GS, Op>{});
        k.emplace_back(new GemmKernel<Shape<256, 16, 256>, Shape<32, 16, 128>, 3, GS, Op>{});
        k.emplace_back(new GemmKernel<Shape<256, 8, 256>, Shape<32, 8, 128>, 3, GS, Op>{});

        // 128
        k.emplace_back(new GemmKernel<Shape<128, 128, 64>, Shape<32, 128, 32>, 3, GS, Op>{});
        k.emplace_back(new GemmKernel<Shape<128, 128, 32>, Shape<32, 128, 32>, 3, GS, Op>{});
        k.emplace_back(new GemmKernel<Shape<128, 96, 64>, Shape<32, 96, 32>, 3, GS, Op>{});
        k.emplace_back(new GemmKernel<Shape<128, 64, 64>, Shape<32, 64, 32>, 3, GS, Op>{});
        k.emplace_back(new GemmKernel<Shape<128, 64, 32>, Shape<32, 64, 32>, 3, GS, Op>{});
        k.emplace_back(new GemmKernel<Shape<128, 32, 128>, Shape<32, 32, 64>, 3, GS, Op>{});
        k.emplace_back(new GemmKernel<Shape<128, 16, 256>, Shape<32, 16, 64>, 3, GS, Op>{});
        k.emplace_back(new GemmKernel<Shape<128, 8, 512>, Shape<32, 8, 128>, 3, GS, Op>{});
        k.emplace_back(new GemmKernel<Shape<128, 8, 512>, Shape<32, 8, 128>, 2, GS, Op>{});  // for 86/89

        // 64
        k.emplace_back(new GemmKernel<Shape<64, 16, 256>, Shape<32, 16, 32>, 3, GS, Op>{});
        k.emplace_back(new GemmKernel<Shape<64, 8, 256>, Shape<32, 8, 32>, 3, GS, Op>{});

        kernels.push_back(std::move(k));
    }

    void Measure(half*                 C,
                 const uint*           A,
                 const half*           B,
                 const half2*          Q,
                 int                   m,
                 int                   n,
                 int                   k,
                 int                   group_size,
                 Type                  type,
                 std::vector<Metric>&  metrics,
                 cudaStream_t          st,
                 std::vector<Kernels>& _kernels)
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
        const auto& kernels = _kernels[gid];
        metrics             = std::vector<Metric>(kernels.size());

        int best = 0;

        for (size_t i = 0; i < kernels.size(); ++i) {
            metrics[i].id = i;
            kernels[i]->GetMetric(metrics[i], m, n, k);
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
                kernels[i]->Launch(C, A, B, Q, m, n, k, type, st);
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

        if (a.prefer != b.prefer) {
            return a.prefer > b.prefer;
        }

        return a.grid_norm < b.grid_norm;
    }

    int Estimate(int m, int n, int k, Kernels& kernels)
    {
        int                 best = 0;
        std::vector<Metric> metrics(kernels.size());
        for (size_t i = 0; i < kernels.size(); ++i) {
            metrics[i].id = i;
            kernels[i]->GetMetric(metrics[i], m, n, k);
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

    void Run(half*                 C,
             const uint*           A,
             const half*           B,
             const half2*          Q,
             int                   m,
             int                   n,
             int                   k,
             int                   group_size,
             Type                  type,
             int                   algo_id,
             cudaStream_t          st,
             std::vector<Kernels>& kernels)
    {
        for (size_t i = 0; i < group_sizes_.size(); ++i) {
            if (group_sizes_[i] == group_size) {
                if (algo_id < 0) {
                    algo_id = Estimate(m, n, k, kernels[i]);
                }
                if (algo_id < 0) {
                    throw std::runtime_error("no feasible kernel found");
                }
                kernels[i].at(algo_id)->Launch(C, A, B, Q, m, n, k, type, st);
                return;
            }
        }
        throw std::runtime_error("unsupported group size");
    }

    Impl()
    {
        cudaEventCreate(&ev_start_);
        cudaEventCreate(&ev_end_);

        using Ops = OutputOps<ops::Identity, ops::SiluActivation, ops::Add>;

        /// TODO: add more group sizes
        Generate<128, Ops>(kernels_);
        group_sizes_.push_back(128);
    }

    ~Impl()
    {
        cudaEventDestroy(ev_end_);
        cudaEventDestroy(ev_start_);
    }

    std::vector<Kernels> kernels_;

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
                        Type                 type,
                        std::vector<Metric>& metrics,
                        cudaStream_t         st)
{
    impl_->Measure(C, A, B, Q, m, n, k, group_size, type, metrics, st, impl_->kernels_);
}

void GemmS4F16::Run(half*        C,
                    const uint*  A,
                    const half*  B,
                    const half2* Q,
                    int          m,
                    int          n,
                    int          k,
                    int          group_size,
                    Type         type,
                    int          algo_id,
                    cudaStream_t st)
{
    impl_->Run(C, A, B, Q, m, n, k, group_size, type, algo_id, st, impl_->kernels_);
}

}  // namespace turbomind
