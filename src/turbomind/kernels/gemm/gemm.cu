// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/tune/args.h"
#include "src/turbomind/kernels/gemm/tune/sampler.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <algorithm>
#include <map>
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

inline decltype(auto) as_tuple(const GemmDesc& d)
{
    return std::tie(d.arch,
                    d.type_a,
                    d.type_b,
                    d.type_c,
                    d.order_a,
                    d.order_b,
                    d.order_c,
                    d.pack_a,
                    d.pack_b,
                    d.pack_u,
                    d.pack_v,
                    d.quant_a.type,
                    d.quant_a.group_size,
                    d.quant_b.type,
                    d.quant_b.group_size,
                    // d.epilogue,
                    d.n,
                    d.k,
                    d.m);
}

inline bool is_compatible(GemmDesc a, GemmDesc b)
{
    // skip batch dim & epilogue flags
    a.m = b.m  = 0;
    a.epilogue = b.epilogue = Epilogue::kNone;
    return as_tuple(a) == as_tuple(b);
}

template<class Cmp>
std::vector<int> ArgSort(size_t size, const Cmp& cmp)
{
    std::vector<int> idxs(size);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::stable_sort(idxs.begin(), idxs.end(), cmp);
    return idxs;
}

}  // namespace

inline bool operator<(const GemmDesc& a, const GemmDesc& b)
{
    return as_tuple(a) < as_tuple(b);
}

struct Gemm::Impl {

    Impl(): props_{GetCudaDeviceProps()}, arch_{props_->major * 100 + props_->minor * 10}, registry_{props_}
    {
        if (auto str = std::getenv("TM_GEMM_TUNE_ARGS")) {
            try {
                ParseTuningArgs(tuning_, str);
            }
            catch (...) {
                std::cerr << "[Gemm2] Failed to parse `TM_GEMM_TUNE_ARGS`, default value will be used.\n";
                tuning_ = {};
            }
        }
        measurer_.emplace(CreateStoppingCriterion(tuning_.max_iter, tuning_.max_time));
    }

    // find launch spec in dispatch cache, dispatch by heuristic on cache miss
    LaunchSpec Dispatch(DispatchPolicy policy, GemmDesc desc, size_t barriers_size, size_t partials_size)
    {
        if (policy & DispatchPolicy::kReuse) {
            auto it = dispatch_cache_.lower_bound(desc);
            if (it != dispatch_cache_.end() && is_compatible(it->first, desc) && it->second.kernel->is_feasible(desc)) {
                return it->second;
            }
            // if (it != dispatch_cache_.end()) {
            //     std::cout << is_compatible(it->first, desc) << " " << it->second.kernel->is_feasible(desc) << "\n";
            // }
            std::cerr << "Failed to find a feasible kernel in the cache, will dispatch by heuristic.\n";
        }

        if (auto it = dispatch_cache_.find(desc); it != dispatch_cache_.end()) {
            return it->second;
        }

        auto specs = Find(desc, barriers_size, partials_size, 1);
        if (!specs.empty()) {
            dispatch_cache_.emplace(desc, specs.front());
            return specs.front();
        }
        return {};
    }

    std::vector<LaunchSpec> Find(const GemmDesc& desc, size_t barrier_size, size_t partials_size, int top_k)
    {
        std::vector<Kernel*> kernels;
        for (const auto& k : registry_.kernels()) {
            if (k->is_feasible(desc)) {
                kernels.push_back(k.get());
            }
        }
        if (kernels.empty()) {
            return {};
        }

        std::vector<std::tuple<Kernel*, int, KernelMetric>> metrics;

        for (const auto& k : kernels) {
            const int max_splits = k->GetMaxSplits(desc.m, desc.n, barrier_size, partials_size);

            auto ms = k->Estimate_v2({desc.m, desc.n, desc.k},  //
                                     std::min(max_splits, tuning_.max_splits),
                                     tuning_.max_waves,
                                     props_->multiProcessorCount);

            for (const auto& [splits, metric] : ms) {
                metrics.emplace_back(k, splits, metric);
            }
        }

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
        //     auto [k, s, m] = metrics[i];
        //     std::cout << k->name() << " s" << s << " " << avg_ratio[i] << " " << mio_ratio[i] << " " << mma_ratio[i]
        //               << " " << m.mio_cost << " " << m.mma_cost << "\n";
        // }

        top_k = top_k > 0 ? std::min<int>(idxs.size(), top_k) : (int)idxs.size();
        std::vector<LaunchSpec> ret;
        ret.reserve(top_k);
        for (int i = 0; i < top_k; ++i) {
            const auto& [kernel, splits, cost] = metrics[idxs[i]];
            ret.push_back(LaunchSpec{kernel, tuning_.swizzle.at(0), splits});
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
        if (dispatch_cache_.find(desc) != dispatch_cache_.end()) {
            return 0;
        }
        std::cerr << "GEMM: " << desc.m << "x" << desc.n << "x" << desc.k << "\n";

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

        for (const auto& s : specs) {
            std::cout << s.kernel->name()          //
                      << " swizzle=" << s.swizzle  //
                      << ", splits=" << s.splits   //
                      << ", measured=" << s.measured << "ms\n";
        }

        if (!specs.empty()) {
            dispatch_cache_[desc] = specs.front();
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

    int Export(std::ostream& os)
    {
        std::vector<std::pair<GemmDesc, LaunchSpec>> entries;
        for (const auto& entry : dispatch_cache_) {
            entries.push_back(entry);
        }
        ExportDispatchCache(os, entries);
        Summary(entries);
        return dispatch_cache_.size();
    }

    int Import(std::istream& is)
    {
        std::vector<std::pair<GemmDesc, LaunchSpec>> entries;
        ImportDispatchCache(is, entries, registry_.kernels());
        for (const auto& entry : entries) {
            dispatch_cache_.insert(entry);
        }
        return dispatch_cache_.size();
    }

    // Print a summary of how many cases a kernel is used
    void Summary(const std::vector<std::pair<GemmDesc, LaunchSpec>>& entries)
    {
        std::vector<Kernel*> uses{nullptr};
        for (const auto& k : registry_.kernels()) {
            uses.push_back(k.get());
        }
        for (const auto& [_, s] : entries) {
            uses.push_back(s.kernel);
        }
        std::sort(uses.begin(), uses.end());
        std::vector<std::pair<int, Kernel*>> count;
        for (size_t i = 1; i < uses.size(); ++i) {
            if (uses[i] != uses[i - 1]) {
                count.emplace_back(-1, uses[i]);
            }
            ++count.back().first;
        }
        std::sort(count.begin(), count.end(), std::greater<>{});
        for (const auto& [n, k] : count) {
            std::cout << k->name() << ": " << n << "\n";
        }
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

    TuningArgs tuning_;

    std::optional<Measurer> measurer_;

    std::map<GemmDesc, LaunchSpec> dispatch_cache_;
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
        Cdesc.type,
        Adesc.order,
        Bdesc.order,
        Cdesc.order,
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

    const auto launch = [&](LaunchSpec spec, cudaStream_t st) {
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
    return impl_->Export(os);
}

int Gemm::Import(std::istream& is)
{
    return impl_->Import(is);
}

}  // namespace turbomind::gemm
