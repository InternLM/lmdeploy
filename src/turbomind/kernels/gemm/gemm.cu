// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/dispatch_cache.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/tuner/params.h"
#include "src/turbomind/kernels/gemm/tuner/sampler.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/linear_weight.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
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
        if (std::getenv("TM_GEMM_WARN_CACHE_MISS")) {
            warn_cache_miss_ = true;
        }
        measurer_.emplace(CreateStoppingCriterion(tuning_.min_iter, tuning_.max_iter, tuning_.max_time));
    }

    // find launch spec in dispatch cache, dispatch by heuristic on cache miss
    LaunchSpec Dispatch(Context& ctx, DispatchPolicy policy, size_t barriers_size, size_t partials_size)
    {
        const auto& desc = ctx.desc();
        if (policy & DispatchPolicy::kReuse) {
            if (auto spec = cache_.LowerBound(desc)) {
                return *spec;
            }
            if (warn_cache_miss_) {
                std::cerr << "Failed to find a feasible kernel in the cache, will dispatch by heuristic: "
                          << to_string(ctx.desc()) << std::endl;
            }
        }

        if (auto spec = cache_.Find(desc)) {
            return *spec;
        }

        auto specs = Find(ctx, barriers_size, partials_size, 1);
        if (!specs.empty()) {
            cache_.Insert(desc, specs.front());
            return specs.front();
        }
        return {};
    }

    int Launch(const LaunchSpec& spec, const Arguments& args, cudaStream_t stream)
    {
        auto workspace = args.workspace;
        return spec.kernel->Launch(args.operation,
                                   args.alpha,
                                   args.A,
                                   args.Adesc,
                                   args.U,
                                   args.Udesc,
                                   args.B,
                                   args.Bdesc,
                                   args.V,
                                   args.Vdesc,
                                   args.global_scale,
                                   args.global_scale_desc,
                                   args.beta,
                                   args.C,
                                   args.Cdesc,
                                   args.D,
                                   args.Ddesc,
                                   args.W,
                                   args.Wdesc,
                                   spec.swizzle,
                                   spec.splits,
                                   workspace,
                                   stream);
    }

    void PrintSelection(const char* kind, const GemmDesc& desc, const LaunchSpec& spec) const
    {
        if (verbose_) {
            std::cout << "[Gemm] " << kind << " " << to_string(desc) << " " << spec.kernel->name()
                      << " family=" << spec.kernel->desc().family << " backend=" << spec.kernel->desc().backend
                      << " splits=" << spec.splits << " swizzle=" << spec.swizzle << "\n";
        }
    }

    std::vector<LaunchSpec> Find(Context& ctx, size_t barrier_size, size_t partials_size, int top_k)
    {
        std::vector<Kernel*> feasible = ctx.Filter(registry_.kernels());

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

        std::vector<std::pair<int, LaunchSpec>> specs;

        PopulateParam param{};
        param.max_splits    = tuning_.max_splits;
        param.max_waves     = tuning_.max_waves;
        param.swizzle       = tuning_.swizzle.at(0);
        param.barriers_size = barrier_size;
        param.partials_size = partials_size;

        for (int cluster_id = 0; cluster_id < (int)proxies.size(); ++cluster_id) {
            auto& kernel = *proxies[cluster_id];

            auto tmp = ctx.Populate(kernel, param);
            for (const auto& s : tmp) {
                specs.emplace_back(cluster_id, s);
            }
        }

        // std::cerr << "#kernel: " << kernels.size() << ", #cluster: " << clusters.size()
        //           << ", #metric: " << metrics.size() << "\n";

        int64_t mio_max = 0;
        int64_t mma_max = 0;
        for (const auto& [_, s] : specs) {
            auto& [mio, mma] = s.estimated;
            mio_max          = std::max(mio_max, mio);
            mma_max          = std::max(mma_max, mma);
        }
        std::vector<float> mio_ratio;
        std::vector<float> mma_ratio;
        std::vector<float> avg_ratio;
        for (const auto& [_, s] : specs) {
            auto& [mio, mma] = s.estimated;
            mio_ratio.push_back((float)mio / mio_max);
            mma_ratio.push_back((float)mma / mma_max);
            avg_ratio.push_back(.5 * (mio_ratio.back() + mma_ratio.back()));
        }
        auto idxs = ArgSort(specs.size(), [&](int i, int j) {  //
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
            const auto& [cluster_id, spec] = specs[idxs[i]];
            // Apply `splits` to all kernels in the cluster
            for (const auto& s : clusters[cluster_id]) {
                auto tmp   = spec;
                tmp.kernel = s.kernel;
                ret.push_back(tmp);
            }
        }

        return ret;
    }

    template<class LaunchFunc>
    std::optional<LaunchSpec>
    Measure(Context& ctx, size_t barriers_size, size_t partials_size, LaunchFunc launch_func, cudaStream_t stream)
    {
        const auto candidates = Find(ctx, barriers_size, partials_size, tuning_.top_k);

        std::vector<LaunchSpec> specs;
        for (const auto& candidate : candidates) {
            auto swizzled = ctx.Swizzle(candidate, tuning_.swizzle);
            specs.insert(specs.end(), swizzled.begin(), swizzled.end());
        }

        specs = Sampler{*measurer_, tuning_.clusters}.Run(std::move(specs), launch_func, stream);

        if (verbose_) {
            for (const auto& spec : specs) {
                std::cout << "[tune] " << to_string(ctx.desc()) << " " << spec.kernel->name()
                          << " swizzle=" << spec.swizzle << " splits=" << spec.splits << " measured=" << spec.measured
                          << "\n";
            }
        }

        if (specs.empty()) {
            std::cerr << "No valid kernel found for the problem\n";
            return std::nullopt;
        }

        cache_.Insert(ctx.desc(), specs.front());
        return specs.front();
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

    bool warn_cache_miss_{};

    const bool verbose_{std::getenv("TM_GEMM_VERBOSE") != nullptr};

    std::optional<Measurer> measurer_;

    DispatchCache cache_;
};

// implementation of GEMM interfaces

Gemm::Gemm(): impl_{new Impl{}} {}

Gemm::~Gemm() = default;

std::optional<WeightPlan> Gemm::GetWeightPlan(const WeightQuery& query) const
{
    const Family* selected{};
    WeightBridge  selected_bridge{};
    bool          selected_preferred{};
    bool          ambiguous{};

    for (const Family* family : impl_->registry_.families()) {
        auto bridge = family->supports(query.weight_format, query.data_type, query.output_dtype, query.grouped);
        if (!bridge) {
            continue;
        }
        const bool preferred = query.input_dtype != kNull && family->input_format().dtype == query.input_dtype;
        if (!selected || preferred > selected_preferred
            || (preferred == selected_preferred && family->priority > selected->priority)) {
            selected           = family;
            selected_bridge    = *bridge;
            selected_preferred = preferred;
            ambiguous          = false;
        }
        else if (preferred == selected_preferred && family->priority == selected->priority) {
            ambiguous = true;
        }
    }

    if (!selected || ambiguous) {
        return std::nullopt;
    }
    WeightPlan plan;
    plan.family_        = selected;
    plan.bridge_        = selected_bridge;
    plan.output_format_ = selected->output_format(Epilogue::kNone);
    return plan;
}

std::vector<DataType> Gemm::DataTypes(const DataFormat& weight_format) const
{
    std::vector<DataType> dtypes;
    for (const Family* family : impl_->registry_.families()) {
        const DataType dtype = family->data_type();
        if (family->supports(weight_format, dtype, kNull, false)
            && std::find(dtypes.begin(), dtypes.end(), dtype) == dtypes.end()) {
            dtypes.push_back(dtype);
        }
    }
    return dtypes;
}

std::optional<ExecPlan> Gemm::GetExecPlan(const Arguments& args)
{
    Context context{*impl_->props_};
    if (!context.Init(args.operation, args.Adesc, args.Udesc, args.Bdesc, args.Vdesc, args.Cdesc, args.Ddesc)) {
        return std::nullopt;
    }

    LaunchSpec launch =
        impl_->Dispatch(context, args.operation.dispatch, args.workspace.barriers_size, args.workspace.partials_size);
    if (!launch.kernel) {
        return std::nullopt;
    }

    impl_->PrintSelection("plan", context.desc(), launch);
    return ExecPlan{context.desc(), launch};
}

std::optional<ExecPlan> Gemm::Tune(const Arguments& args)
{
    Context context{*impl_->props_};
    if (!context.Init(args.operation, args.Adesc, args.Udesc, args.Bdesc, args.Vdesc, args.Cdesc, args.Ddesc)) {
        return std::nullopt;
    }

    if (args.operation.dispatch & DispatchPolicy::kReuse) {
        if (auto selected = impl_->cache_.Find(context.desc())) {
            impl_->PrintSelection("plan", context.desc(), *selected);
            return ExecPlan{context.desc(), *selected};
        }
    }

    const auto launch = [&](LaunchSpec spec, cudaStream_t stream) { return impl_->Launch(spec, args, stream); };

    std::optional<LaunchSpec> selected =
        impl_->Measure(context, args.workspace.barriers_size, args.workspace.partials_size, launch, args.stream);
    if (!selected) {
        return std::nullopt;
    }
    impl_->PrintSelection("tune", context.desc(), *selected);
    return ExecPlan{context.desc(), *selected};
}

int Gemm::Run(const ExecPlan& plan, const Arguments& args)
{
    if (!plan.launch_.kernel) {
        TM_LOG_FATAL("No feasible kernel found for the problem: {}", to_string(plan.desc_));
        return -1;
    }
    return impl_->Launch(plan.launch_, args, args.stream);
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
