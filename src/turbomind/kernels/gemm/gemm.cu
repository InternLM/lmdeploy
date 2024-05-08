// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/cache_utils.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

namespace turbomind::gemm {

void ExportDispatchCache(std::ostream& os, const std::vector<std::pair<GemmDesc, LaunchSpec>>& entries);

void ImportDispatchCache(std::istream&                                 is,
                         std::vector<std::pair<GemmDesc, LaunchSpec>>& entries,
                         const std::vector<std::unique_ptr<Kernel>>&   kernels);

namespace {

inline decltype(auto) as_tuple(const GemmDesc& d)
{
    return std::tie(d.type_a,
                    d.type_b,
                    d.type_c,
                    d.order_a,
                    d.order_b,
                    d.order_c,
                    d.quant_b.type,
                    d.quant_b.group_size,
                    d.epilogue,
                    d.n,
                    d.k,
                    d.m);
}

inline bool is_compatible(GemmDesc a, GemmDesc b)
{
    a.m = b.m = 0;
    return as_tuple(a) == as_tuple(b);
}

}  // namespace

inline bool operator<(const GemmDesc& a, const GemmDesc& b)
{
    return as_tuple(a) < as_tuple(b);
}

struct Gemm::Impl {

    Impl(): props_{GetCudaDeviceProps()}, registry_{props_} {}

    // find launch spec in dispatch cache, dispatch by heuristic on cache miss
    LaunchSpec Dispatch(DispatchPolicy policy, GemmDesc desc, size_t barriers_size, size_t workspace_size)
    {
        if (policy == DispatchPolicy::kUseCached) {
            auto it = dispatch_cache_.lower_bound(desc);
            if (is_compatible(it->first, desc) && it->second.kernel->is_feasible(desc)) {
                return it->second;
            }
            std::cout << "Failed to find a feasible kernel in the cache, will dispatch by heuristic.\n";
        }

        if (auto it = dispatch_cache_.find(desc); it != dispatch_cache_.end()) {
            return it->second;
        }

        auto specs = Find(desc, barriers_size, workspace_size, 1);

        if (specs.empty()) {
            return {};
        }

        const auto& [spec, _] = specs.front();

        dispatch_cache_.emplace(desc, spec);

        return spec;
    }

    std::vector<std::pair<LaunchSpec, float>>
    Find(const GemmDesc& desc, size_t barrier_size, size_t workspace_size, int top_k)
    {
        std::vector<Kernel*> kernels;

        for (const auto& k : registry_.kernels()) {
            if (k->is_feasible(desc)) {
                kernels.push_back(k.get());
            }
        }

        // is a better than b
        auto compare = [&](const Kernel* a, const Kernel* b) {
            const int m_a = a->cta_tile_size().x;
            const int m_b = b->cta_tile_size().x;
            if (std::max(m_a, m_b) <= desc.m) {  // m_0 < m_1 <= M
                return m_a > m_b;
            }
            if (desc.m <= std::min(m_a, m_b)) {  // M <= m_0 < m_1
                return m_a < m_b;
            }
            // m_0 <= M <= m_1
            return m_a > m_b;
        };

        auto best_cta_m = (*std::min_element(kernels.begin(), kernels.end(), compare))->cta_tile_size().x;
        kernels.erase(
            std::remove_if(kernels.begin(), kernels.end(), [&](auto k) { return k->cta_tile_size().x != best_cta_m; }),
            kernels.end());

        //                    cost     splits
        std::vector<std::pair<int64_t, int>> costs;

        for (const auto& k : kernels) {
            std::cout << "\n" << k->name() << "\n";
            int max_split_k = k->GetMaxSplits(desc.m, desc.n, barrier_size, workspace_size);
            std::cout << "max_split_k: " << max_split_k << "\n";
            auto [splits, cost] = k->EstimateSplits(desc.m,  //
                                                    desc.n,
                                                    desc.k,
                                                    max_split_k,
                                                    props_->multiProcessorCount,
                                                    1,
                                                    1)
                                      .front();
            costs.emplace_back(cost, splits);
        }

        std::vector<int> idxs(kernels.size());
        std::iota(idxs.begin(), idxs.end(), 0);

        top_k = std::min<int>(idxs.size(), top_k);

        std::partial_sort(idxs.begin(), idxs.begin() + top_k, idxs.end(), [&](int i, int j) {
            return costs[i] < costs[j];  //
        });

        std::vector<std::pair<LaunchSpec, float>> ret;
        ret.reserve(top_k);

        for (int i = 0; i < top_k; ++i) {
            const auto& [cost, splits] = costs[idxs[i]];
            ret.emplace_back(LaunchSpec{kernels[idxs[i]], 3, splits}, static_cast<float>(cost));
        }

        return ret;
    }

    template<class LaunchFunc>
    int Measure(const GemmDesc& desc,
                size_t          barriers_size,
                size_t          workspace_size,
                int             top_k,
                LaunchFunc      launch_func,
                cudaStream_t    st)
    {
        std::vector<Kernel*> kernels;
        for (const auto& k : registry_.kernels()) {
            if (k->is_feasible(desc)) {
                kernels.push_back(k.get());
            }
        }

        std::vector<LaunchSpec> specs;
        for (const auto& k : kernels) {
            int max_splits = k->GetMaxSplits(desc.m, desc.n, barriers_size, workspace_size);
            max_splits     = std::min(max_splits, 8);
            auto splits    = k->EstimateSplits(desc.m,  //
                                            desc.n,
                                            desc.k,
                                            max_splits,
                                            props_->multiProcessorCount,
                                            32,
                                            10);
            for (const auto& [split_k, _] : splits) {
                for (const auto& swizzle : {3}) {
                    specs.push_back(LaunchSpec{k, swizzle, split_k});
                }
            }
        }

        /// TODO: filter kernels by heuristic
        constexpr int   kMinIteration = 10;
        constexpr int   kMaxIteration = 100;
        constexpr float kMaxDuration  = 25;  // std::milli

        constexpr float kFloatInf = std::numeric_limits<float>::infinity();

        cudaEvent_t ev_beg;
        cudaEvent_t ev_end;

        cudaEventCreate(&ev_beg);
        cudaEventCreate(&ev_end);

        std::vector<float> measurements;

        for (const auto& spec : specs) {
            int                iter = 0;
            float              accum{};
            std::vector<float> duration;
            std::cout << "measuring " << spec.kernel->name() << " with swizzle=" << spec.swizzle
                      << ", splits=" << spec.splits << "\n";
            while (true) {
                CacheFlushing::flush(st);
                // cudaStreamSynchronize(st);

                cudaEventRecord(ev_beg, st);

                launch_func(spec, st);

                cudaEventRecord(ev_end, st);
                cudaEventSynchronize(ev_end);

                auto err = cudaGetLastError();

                float delta{};
                cudaEventElapsedTime(&delta, ev_beg, ev_end);
                duration.push_back(delta);

                ++iter;
                accum += delta;

                if (err != cudaSuccess) {
                    std::cout << cudaGetErrorString(err) << "\n";
                    // std::abort();
                    duration.back() = kFloatInf;
                    break;
                }

                if (iter >= kMaxIteration) {
                    break;
                }
                if (iter >= kMinIteration && accum >= kMaxDuration) {
                    break;
                }
            }

            measurements.push_back(accum / static_cast<float>(iter));
        }

        cudaEventDestroy(ev_beg);
        cudaEventDestroy(ev_end);

        std::vector<int> idxs(measurements.size());
        std::iota(idxs.begin(), idxs.end(), 0);
        std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {  //
            return measurements[i] < measurements[j];
        });

        for (const auto& i : idxs) {
            std::cout << specs[i].kernel->name() << " swizzle=" << specs[i].swizzle << ", splits=" << specs[i].splits
                      << ", time=" << measurements[i] << "ms\n";
        }

        LaunchSpec spec{};
        if (!idxs.empty() && measurements[idxs[0]] != kFloatInf) {
            spec                  = specs[idxs[0]];
            dispatch_cache_[desc] = spec;
        }
        else {
            std::cout << "No valid kernel found for problem.\n";
            return -1;
        }

        // When an aligned kernel is selected, find the fastest unaligned kernel for `m - 1`
        if (desc.m > 1 && spec.kernel->align_m()) {
            auto unaligned_desc = desc;
            --unaligned_desc.m;
            LaunchSpec unaligned_spec{};
            for (const auto& i : idxs) {
                if (measurements[i] == kFloatInf) {
                    break;
                }
                if (!specs[i].kernel->align_m()) {
                    unaligned_spec = specs[i];
                    break;
                }
            }
            if (unaligned_spec.kernel) {
                dispatch_cache_[unaligned_desc] = unaligned_spec;
            }
            else {
                std::cout << "No non-aligned kernel found for problem.\n";
                return -1;
            }
        }

        return 0;
    }

    int Export(std::ostream& os)
    {
        std::vector<std::pair<GemmDesc, LaunchSpec>> entries;
        for (const auto& entry : dispatch_cache_) {
            entries.push_back(entry);
        }
        ExportDispatchCache(os, entries);
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

private:
    /// TODO: move to cuda utils
    static std::unique_ptr<cudaDeviceProp> GetCudaDeviceProps()
    {
        auto props     = std::make_unique<cudaDeviceProp>();
        int  device_id = -1;
        cudaGetDevice(&device_id);
        cudaGetDeviceProperties(props.get(), device_id);
        return props;
    }

private:
    std::shared_ptr<cudaDeviceProp> props_;

    std::map<GemmDesc, LaunchSpec> dispatch_cache_;
    Registry                       registry_;
};

// implmenetation of GEMM interfaces

Gemm::Gemm(): impl_{new Impl{}} {}

Gemm::~Gemm() = default;

int Gemm::Run(const Operation&    operation,
              const void*         alpha,
              const void*         A,
              const MatrixLayout& Adesc,
              const void*         B,
              const MatrixLayout& Bdesc,
              const void*         Q,
              const MatrixLayout& Qdesc,
              const void*         beta,
              const void*         C,
              const MatrixLayout& Cdesc,
              void*               D,
              const MatrixLayout& Ddesc,
              void*               barriers,
              size_t              barriers_size,
              void*               workspace,
              size_t              workspace_size,
              cudaStream_t        stream)
{

    if (Adesc.rows != Ddesc.rows || Bdesc.cols != Ddesc.cols || Adesc.cols != Bdesc.rows) {
        return -1;
    }

    const int m = Ddesc.rows;
    const int n = Ddesc.cols;
    const int k = Adesc.cols;

    const GemmDesc desc{
        Adesc.type,
        Bdesc.type,
        Cdesc.type,
        Adesc.order,
        Bdesc.order,
        Cdesc.order,
        operation.quant_desc,
        operation.epilogue,
        m,
        n,
        k,
    };

    const auto launch = [&](LaunchSpec spec, cudaStream_t st) {
        size_t tmp_barriers_size  = barriers_size;
        size_t tmp_workspace_size = workspace_size;
        return spec.kernel->Launch(operation,
                                   alpha,
                                   A,
                                   Adesc,
                                   B,
                                   Bdesc,
                                   Q,
                                   Qdesc,
                                   beta,
                                   C,
                                   Cdesc,
                                   D,
                                   Ddesc,
                                   spec.swizzle,
                                   spec.splits,
                                   barriers,
                                   tmp_barriers_size,
                                   workspace,
                                   tmp_workspace_size,
                                   st);
    };

    LaunchSpec spec{};

    if (operation.dispatch == DispatchPolicy::kMeasure) {
        impl_->Measure(desc, barriers_size, workspace_size, 1, launch, stream);
    }

    spec = impl_->Dispatch(operation.dispatch, desc, barriers_size, workspace_size);

    if (spec.kernel) {
        std::cout << "[Gemm] dispatch: " << spec.kernel->name()  //
                  << " split_k=" << spec.splits                  //
                  << " swizzle=" << spec.swizzle << std::endl;
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