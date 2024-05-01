#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/gemm_template.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

namespace turbomind::gemm {

template void invoke(half*        C,
                     const half*  A,
                     const half*  B,
                     const half*  Q,
                     int          m,
                     int          n,
                     int          k,
                     int          splits,
                     void*        workspace,
                     cudaStream_t st);
template void invoke(half*          C,
                     const half*    A,
                     const uint4_t* B,
                     const half*    Q,
                     int            m,
                     int            n,
                     int            k,
                     int            splits,
                     void*          workspace,
                     cudaStream_t   st);
template void invoke(half*          C,
                     const half*    A,
                     const uint8_t* B,
                     const half*    Q,
                     int            m,
                     int            n,
                     int            k,
                     int            splits,
                     void*          workspace,
                     cudaStream_t   st);

namespace {

inline decltype(auto) as_tuple(const GemmDesc& d)
{
    return std::tie(
        d.layout_A, d.layout_B, d.layout_C, d.type_A, d.type_B, d.type_C, d.quant_type, d.epilogue, d.n, d.k, d.m);
}

}  // namespace

inline bool operator<(const GemmDesc& a, const GemmDesc& b)
{
    return as_tuple(a) < as_tuple(b);
}

struct Gemm::Impl {

    Impl(): props_{GetCudaDeviceProps()}, registry_{props_} {}

    struct LaunchSpec {
        Kernel* kernel;
        int     splits;
    };

    // find launch spec in dispatch cache, dispatch by heuristic on cache miss
    LaunchSpec Dispatch(GemmDesc desc, size_t barriers_size, size_t workspace_size)
    {
        if (auto it = dispatch_cache_.find(desc); it != dispatch_cache_.end()) {
            return it->second;
        }

        auto kernels = Find(desc, barriers_size, workspace_size, 1);
        if (kernels.empty()) {
            return {};
        }

        const auto& [kernel, splits, cost] = kernels.front();

        LaunchSpec spec{kernel, splits};

        dispatch_cache_.emplace(desc, spec);

        return spec;
    }

    std::vector<std::tuple<Kernel*, int, int64_t>>
    Find(const GemmDesc& desc, size_t barrier_size, size_t workspace_size, int top_k)
    {
        std::vector<Kernel*> kernels;
        //                    cost     splits
        std::vector<std::pair<int64_t, int>> costs;

        for (const auto& k : registry_.kernels()) {
            std::cout << k->name() << "\n";
            if (k->is_feasible(desc) != true) {
                continue;
            }

            const int max_split_k = k->GetMaxSplits(desc.m, desc.n, barrier_size, workspace_size);
            printf("max_split_k = %d\n", max_split_k);

            auto [splits, cost] =
                k->FindSplitCount(desc.m, desc.n, desc.k, max_split_k, props_->multiProcessorCount, 1);

            costs.emplace_back(cost, splits);
            kernels.push_back(k.get());
        }

        // is a better than b
        auto compare = [&](const Kernel* a, const Kernel* b) {
            const int m_a = a->cta_tile_size().x;
            const int m_b = b->cta_tile_size().x;
            if (std::max(m_a, m_b) <= desc.m) {  // m_0 < m_1 <= M
                return m_b < m_a;
            }
            if (desc.m <= std::min(m_a, m_b)) {  // M <= m_0 < m_1
                return m_a < m_b;
            }
            // m_0 <= M <= m_1
            return round_up(desc.m, m_a) < round_up(desc.m, m_b);
        };

        std::vector<int> idxs(kernels.size());
        std::iota(idxs.begin(), idxs.end(), 0);

        top_k = std::min<int>(idxs.size(), top_k);

        std::partial_sort(idxs.begin(), idxs.begin() + top_k, idxs.end(), [&](int i, int j) {
            if (compare(kernels[i], kernels[j])) {
                return true;
            }
            else if (compare(kernels[j], kernels[i])) {
                return false;
            }
            else {
                return costs[i] < costs[j];  //
            }
        });

        std::vector<std::tuple<Kernel*, int, int64_t>> ret;
        ret.reserve(top_k);

        for (int i = 0; i < top_k; ++i) {
            const auto& [cost, splits] = costs[idxs[i]];
            ret.emplace_back(kernels[idxs[i]], splits, cost);
        }

        return ret;
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

int Gemm::Run(LayoutType   layout_A,
              LayoutType   layout_B,
              LayoutType   layout_C,
              EpilogueType epilogue,
              int          m,
              int          n,
              int          k,
              const void*  A,
              DataType     type_A,
              int          lda,
              const void*  B,
              DataType     type_B,
              int          ldb,
              const void*  Q,
              QuantType    quant_type,
              int          ldq,
              const float* beta,
              void*        C,
              DataType     type_C,
              int          ldc,
              int*         barriers,
              size_t       barriers_size,
              void*        workspace,
              size_t       workspace_size,
              cudaStream_t stream)
{
    const GemmDesc desc{layout_A,  //
                        layout_B,
                        layout_C,
                        type_A,
                        type_B,
                        type_C,
                        quant_type,
                        epilogue,
                        m,
                        n,
                        k};

    auto spec = impl_->Dispatch(desc, barriers_size, workspace_size);

    if (spec.kernel) {
        return spec.kernel->Launch(m,
                                   n,
                                   k,
                                   A,
                                   lda,
                                   B,
                                   ldb,
                                   Q,
                                   ldq,
                                   *beta,
                                   C,
                                   ldc,
                                   spec.splits,
                                   epilogue,
                                   barriers,
                                   barriers_size,
                                   workspace,
                                   workspace_size,
                                   stream);
    }

    printf("No feasible kernel found for the problem.\n");

    return -1;
}

}  // namespace turbomind::gemm