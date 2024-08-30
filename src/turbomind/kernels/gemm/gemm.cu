// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/dispatch_cache.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/tuner/params.h"
#include "src/turbomind/kernels/gemm/tuner/sampler.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

namespace turbomind::gemm {

void ExportDispatchCache(std::ostream& os, const std::vector<std::pair<GemmDesc, LaunchSpec>>& entries);

void ImportDispatchCache(std::istream&                                 is,
                         std::vector<std::pair<GemmDesc, LaunchSpec>>& entries,
                         const std::vector<std::unique_ptr<Kernel>>&   kernels);

namespace {

template<class Cmp>
std::vector<int> ArgSort(size_t size, const Cmp& cmp)
{
    std::vector<int> idxs(size);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::stable_sort(idxs.begin(), idxs.end(), cmp);
    return idxs;
}

inline int get_batch_dim(const GemmDesc& desc)
{
    return desc.batch_dim == 0 ? desc.m : desc.n;
}

inline int get_batch_dim(const KernelDesc& desc, int batch_dim)
{
    return batch_dim == 0 ? desc.cta_tile.x : desc.cta_tile.y;
}

}  // namespace

struct Gemm::Impl {

    Impl():
        props_{GetCudaDeviceProps()},
        arch_{props_->major * 100 + props_->minor * 10},
        registry_{props_},
        cache_{registry_.kernels()}
    {
        if (auto str = std::getenv("TM_GEMM_TUNE")) {
            try {
                ParseTuningParams(tuning_, str);
            }
            catch (...) {
                std::cerr << "[Gemm2] Failed to parse `TM_GEMM_TUNE`, default value will be used.\n";
                tuning_ = {};
            }
        }
        measurer_.emplace(CreateStoppingCriterion(tuning_.min_iter, tuning_.max_iter, tuning_.max_time));
    }

    // find launch spec in dispatch cache, dispatch by heuristic on cache miss
    LaunchSpec Dispatch(DispatchPolicy policy, GemmDesc desc, size_t barriers_size, size_t partials_size)
    {
        if (policy & DispatchPolicy::kReuse) {
            if (auto spec = cache_.LowerBound(desc)) {
                return *spec;
            }
            std::cerr << "Failed to find a feasible kernel in the cache, will dispatch by heuristic.\n";
        }

        if (auto spec = cache_.Find(desc)) {
            return *spec;
        }

        auto specs = Find(desc, barriers_size, partials_size, 1);
        if (!specs.empty()) {
            cache_.Insert(desc, specs.front());
            return specs.front();
        }
        return {};
    }

    std::vector<LaunchSpec> Find(const GemmDesc& desc, size_t barrier_size, size_t partials_size, int top_k)
    {
        std::vector<Kernel*> feasible;
        std::copy_if(registry_.kernels().begin(), registry_.kernels().end(), std::back_inserter(feasible), [&](auto p) {
            return p->is_feasible(desc);
        });
        if (feasible.empty()) {
            return {};
        }

        if (1) {
            int max_batch_size = 0;
            for (const auto& k : feasible) {
                max_batch_size = std::max(get_batch_dim(k->desc(), desc.batch_dim), max_batch_size);
            }
            const int batch_size = get_batch_dim(desc);
            for (const auto& k : feasible) {
                const auto x = get_batch_dim(k->desc(), desc.batch_dim);
                if (x >= batch_size) {
                    max_batch_size = std::min(max_batch_size, x);
                }
            }
            auto pred = [&](auto k) { return get_batch_dim(k->desc(), desc.batch_dim) > max_batch_size; };
            feasible.erase(std::remove_if(feasible.begin(), feasible.end(), pred), feasible.end());
        }

        std::vector<std::vector<LaunchSpec>> clusters;
        {
            std::vector<LaunchSpec> tmp;
            tmp.reserve(feasible.size());
            for (const auto& k : feasible) {
                LaunchSpec spec{k};
                tmp.push_back(spec);
            }
            clusters = Cluster(tmp, ClusteringParam{false, true});
        }
        std::vector<Kernel*> proxies;
        proxies.reserve(clusters.size());

        for (const auto& c : clusters) {
            proxies.push_back(c.front().kernel);
        }

        //             cluster_id, splits, metrics
        std::vector<std::tuple<int, int, KernelMetric>> metrics;

        for (int cluster_id = 0; cluster_id < (int)proxies.size(); ++cluster_id) {
            auto&     kernel     = *proxies[cluster_id];
            const int max_splits = kernel.GetMaxSplits(desc.m, desc.n, desc.k, barrier_size, partials_size);

            auto ms = kernel.Estimate_v2({desc.m, desc.n, desc.k},  //
                                         std::min(max_splits, tuning_.max_splits),
                                         tuning_.max_waves,
                                         props_->multiProcessorCount);

            for (const auto& [splits, metric] : ms) {
                metrics.emplace_back(cluster_id, splits, metric);
            }
        }

        // std::cerr << "#kernel: " << kernels.size() << ", #cluster: " << clusters.size()
        //           << ", #metric: " << metrics.size() << "\n";

        std::vector<int64_t> mio_cost;
        std::vector<int64_t> mma_cost;
        for (const auto& [_, s, m] : metrics) {
            mio_cost.push_back(m.mio_cost);
            mma_cost.push_back(m.mma_cost);
        }

        const auto mio_max = *std::max_element(mio_cost.begin(), mio_cost.end());
        const auto mma_max = *std::max_element(mma_cost.begin(), mma_cost.end());

        std::vector<float> mio_ratio;
        std::vector<float> mma_ratio;
        std::vector<float> avg_ratio;
        for (size_t i = 0; i < metrics.size(); ++i) {
            mio_ratio.push_back(static_cast<float>(mio_cost[i]) / mio_max);
            mma_ratio.push_back(static_cast<float>(mma_cost[i]) / mma_max);
            avg_ratio.push_back(.5f * (mio_ratio.back() + mma_ratio.back()));
        }

        auto idxs = ArgSort(metrics.size(), [&](int i, int j) {  //
            return avg_ratio[i] < avg_ratio[j];
        });

        // for (const auto& i : idxs) {
        //     auto [cid, s, m] = metrics[i];
        //     std::cout << clusters[cid].front().kernel->name() << " s" << s << " " << avg_ratio[i] << " " <<
        //     mio_ratio[i]
        //               << " " << mma_ratio[i] << " " << m.mio_cost << " " << m.mma_cost << "\n";
        // }

        top_k = top_k > 0 ? std::min<int>(idxs.size(), top_k) : (int)idxs.size();
        std::vector<LaunchSpec> ret;
        ret.reserve(top_k);
        for (int i = 0; i < top_k; ++i) {
            const auto& [cluster_id, splits, cost] = metrics[idxs[i]];
            // Apply `splits` to all kernels in the cluster
            for (const auto& s : clusters[cluster_id]) {
                ret.push_back(LaunchSpec{s.kernel, tuning_.swizzle.at(0), splits});
            }
        }

        return ret;
    }

    template<class LaunchFunc>
    int Measure(const GemmDesc& desc,
                size_t          barriers_size,
                size_t          partials_size,
                int             top_k,
                LaunchFunc      launch_func,
                cudaStream_t    st)
    {
        // Early exit on exact match
        if (cache_.Find(desc)) {
            return 0;
        }
        // std::cerr << "GEMM: " << desc.m << "x" << desc.n << "x" << desc.k << "\n";

        const auto tmp = Find(desc, barriers_size, partials_size, tuning_.top_k);

        std::vector<LaunchSpec> specs;
        for (const auto& spec : tmp) {
            // populate swizzle parameters
            const auto swis = FilterSwizzleParam(*spec.kernel, desc.m, desc.n, desc.k, tuning_.swizzle);
            for (const auto& swi : swis) {
                specs.push_back(spec);
                specs.back().swizzle = swi;
            }
        }

        specs = Sampler{*measurer_, tuning_.clusters}.Run(specs, launch_func, st);

        // for (const auto& s : specs) {
        //     std::cout << s.kernel->name()          //
        //               << " swizzle=" << s.swizzle  //
        //               << ", splits=" << s.splits   //
        //               << ", measured=" << s.measured << "ms\n";
        // }

        if (!specs.empty()) {
            cache_.Insert(desc, specs.front());
        }
        else {
            std::cerr << "No valid kernel found for the problem\n";
            return -1;
        }

        return 0;
    }

    std::vector<int> FilterSwizzleParam(Kernel& kernel, int m, int n, int k, const std::vector<int>& swis)
    {
        std::vector<int> swizzles;
        for (const auto& swi : swis) {
            // To use splits=1 here, swizzling must not depends on split count
            swizzles.push_back(kernel.GetSwizzle(m, n, k, 1, swi));
        }
        if (swizzles.size() == 1) {
            return swizzles;
        }

        // De-duplicate possible swizzles while keep the order
        std::sort(swizzles.begin(), swizzles.end());
        swizzles.erase(std::unique(swizzles.begin(), swizzles.end()), swizzles.end());

        std::vector<int> tmp;
        std::copy_if(swis.begin(), swis.end(), std::back_inserter(tmp), [&](int swi) {
            return std::find(swizzles.begin(), swizzles.end(), swi) != swizzles.end();
        });
        tmp.swap(swizzles);

        return swizzles;
    }

    /// TODO: move to cuda utils
    static std::unique_ptr<cudaDeviceProp> GetCudaDeviceProps()
    {
        auto props     = std::make_unique<cudaDeviceProp>();
        int  device_id = -1;
        cudaGetDevice(&device_id);
        cudaGetDeviceProperties(props.get(), device_id);
        return props;
    }

    std::shared_ptr<cudaDeviceProp> props_;

    int arch_;

    Registry registry_;

    TuningParams tuning_;

    std::optional<Measurer> measurer_;

    DispatchCache cache_;
};

// implementation of GEMM interfaces

Gemm::Gemm(): impl_{new Impl{}} {}

Gemm::~Gemm() = default;

int Gemm::Run(const Operation&    operation,
              float               alpha,
              const void*         A,
              const MatrixLayout& Adesc,
              const void*         U,
              const MatrixLayout& Udesc,
              const void*         B,
              const MatrixLayout& Bdesc,
              const void*         V,
              const MatrixLayout& Vdesc,
              float               beta,
              const void*         C,
              const MatrixLayout& Cdesc,
              void*               D,
              const MatrixLayout& Ddesc,
              const Workspace&    workspace,
              cudaStream_t        stream)
{

    if (Adesc.rows != Ddesc.rows || Bdesc.cols != Ddesc.cols || Adesc.cols != Bdesc.rows) {
        return -1;
    }

    const int m = Ddesc.rows;
    const int n = Ddesc.cols;
    const int k = Adesc.cols;

    const GemmDesc desc{
        impl_->arch_,
        Adesc.type,
        Bdesc.type,
        Ddesc.type,
        Adesc.order,
        Bdesc.order,
        Ddesc.order,
        Adesc.pack,
        Bdesc.pack,
        Udesc.pack,
        Vdesc.pack,
        operation.quant_a,
        operation.quant_b,
        operation.epilogue,
        m,
        n,
        k,
    };

    const auto launch = [=](LaunchSpec spec, cudaStream_t st) {
        auto _workspace = workspace;
        return spec.kernel->Launch(operation,
                                   alpha,
                                   A,
                                   Adesc,
                                   U,
                                   Udesc,
                                   B,
                                   Bdesc,
                                   V,
                                   Vdesc,
                                   beta,
                                   C,
                                   Cdesc,
                                   D,
                                   Ddesc,
                                   spec.swizzle,
                                   spec.splits,
                                   _workspace,
                                   st);
    };

    if (operation.reserved) {
        auto specs = impl_->Find(desc, workspace.barriers_size, workspace.partials_size, 0);
        auto cases = (std::vector<std::function<LaunchSpec()>>*)operation.reserved;
        for (const auto& spec : specs) {
            cases->push_back([=] {
                launch(spec, stream);
                return spec;
            });
        }
        return -1;
    }

    LaunchSpec spec{};

    if (operation.dispatch & DispatchPolicy::kMeasure) {
        impl_->Measure(desc, workspace.barriers_size, workspace.partials_size, 1, launch, stream);
    }

    spec = impl_->Dispatch(operation.dispatch, desc, workspace.barriers_size, workspace.partials_size);

    if (spec.kernel) {
        // std::cout << "[Gemm] dispatch: " << spec.kernel->name()  //
        //           << " split_k=" << spec.splits                  //
        //           << " swizzle=" << spec.swizzle << std::endl;
        return launch(spec, stream);
    }

    printf("No feasible kernel found for the problem.\n");

    return -1;
}

int Gemm::Export(std::ostream& os)
{
    return impl_->cache_.Export(os);
}

int Gemm::Import(std::istream& is)
{
    return impl_->cache_.Import(is);
}

std::vector<int> Gemm::GetTuningSeq() const
{
    return impl_->tuning_.seq;
}

}  // namespace turbomind::gemm
