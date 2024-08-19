// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>

namespace turbomind::gemm {

bool Kernel::is_feasible(const GemmDesc& desc) const noexcept
{
    constexpr bool debug = 0;

    if constexpr (debug)
        printf("S\n");

    if (!is_arch_compatible(desc_.arch, desc.arch)) {
        return false;
    }

    if constexpr (debug)
        printf("S0\n");

    if (std::tie(desc.order_a, desc.order_b, desc.order_c) != std::tie(desc_.order_a, desc_.order_b, desc_.order_c)) {
        return false;
    }

    if constexpr (debug)
        printf("A\n");

    if (std::tie(desc.type_a, desc.type_b, desc.type_c) != std::tie(desc_.type_a, desc_.type_b, desc_.type_c)) {
        return false;
    }

    if constexpr (debug) {
        printf("B\n");
        printf("%X %X %X %X\n", desc.pack_a, desc_.pack_a, desc.pack_u, desc_.pack_u);
    }

    if (std::tie(desc.pack_a, desc.pack_u) != std::tie(desc_.pack_a, desc_.pack_u)) {
        return false;
    }

    if constexpr (debug) {
        printf("C\n");
        printf("%X %X %X %X\n", desc.pack_b, desc_.pack_b, desc.pack_v, desc_.pack_v);
    }

    if (std::tie(desc.pack_b, desc.pack_v) != std::tie(desc_.pack_b, desc_.pack_v)) {
        return false;
    }

    if constexpr (debug)
        printf("D\n");

    if (desc.quant_a.type != desc_.quant_a.type || desc.quant_a.group_size != desc_.quant_a.group_size) {
        return false;
    }

    if constexpr (debug)
        printf("E\n");

    if (desc.quant_b.type != desc_.quant_b.type || desc.quant_b.group_size != desc_.quant_b.group_size) {
        return false;
    }

    if constexpr (debug)
        printf("F\n");

    if (desc.m % desc_.align.x || desc.n % desc_.align.y || desc.k % desc_.align.z) {
        return false;
    }

    if constexpr (debug)
        printf("success\n");

    return true;
}

std::vector<std::pair<int, KernelMetric>>
Kernel::Estimate_v2(std::array<int, 3> size, int max_splits, int max_waves, int sm_count) const
{
    const auto [m, n, k]        = size;
    const int64_t tiled_shape_m = ceil_div(m, desc_.cta_tile.x);
    const int64_t tiled_shape_n = ceil_div(n, desc_.cta_tile.y);
    const int     chunk_cnt_k   = ceil_div(k, chunk_size_k_);

    // Despite we only have sm_count * constant tensor cores, this is the granularity for scheduling
    const int   concurrency     = sm_count * desc_.max_active_ctas;
    const float waves_per_split = float(tiled_shape_m * tiled_shape_n) / concurrency;
    const float splits_per_wave = 1.f / waves_per_split;

    // Tile quantization
    const int64_t ceil_m = tiled_shape_m * desc_.cta_tile.x;
    const int64_t ceil_n = tiled_shape_n * desc_.cta_tile.y;

    std::vector<std::pair<int, KernelMetric>> metrics;

    for (int splits = 1; splits <= max_splits; ++splits) {
        // Split quantization, penalize uneven splits
        const int64_t split_ceil_k = ceil_div(chunk_cnt_k, splits) * chunk_size_k_;
        // Footprint for single split
        const int64_t split_mma_cost = ceil_m * ceil_n * split_ceil_k;
        // Footprint for single wave
        const int64_t wave_mma_cost = split_mma_cost * splits_per_wave;

        // Wave quantization
        // const int waves = (int)std::ceil(wave_per_split * splits);

        // Bold simulation of thread block scheduling
        const int   grid_size    = tiled_shape_m * tiled_shape_n * splits;
        const int   full_waves   = grid_size / concurrency;
        const int   residue      = grid_size % concurrency;
        const float partial_wave = (float)ceil_div(residue, sm_count) / desc_.max_active_ctas;
        const float waves        = full_waves + partial_wave;

        if (splits > 1 && waves > max_waves) {
            break;
        }
        // ceil(tiled_mn / C * splits) * C / tiled_mn * ceil_m * ceil_n * split_ceil_k
        const int64_t mma_cost = wave_mma_cost * waves;

        // IO has less severe quantization effect
        const int64_t mio_cost_a = get_size(desc_.type_a, tiled_shape_n * m * split_ceil_k) * splits;
        const int64_t mio_cost_b = get_size(desc_.type_b, tiled_shape_m * n * split_ceil_k) * splits;
        /// TODO: read type from `desc_.accum` when added
        const int64_t mio_cost_c = get_size(DataType::F32, (int64_t)m * n) * (splits - 1) * 2;
        const int64_t mio_cost   = mio_cost_a + mio_cost_b + mio_cost_c;

        // std::cout << name() << " " << splits << " " << waves << " " << (float)mio_cost << " " << (float)mma_cost
        //           << "\n";

        metrics.emplace_back(splits, KernelMetric{mio_cost, mma_cost});
    }

    return metrics;
}

std::string Kernel::GetName() const
{
    std::stringstream ss;

    ss << "sm" << desc_.arch / 10;
    ss << "_" << to_string(desc_.type_a);  //
    if ((int)desc_.quant_a.type) {
        ss << "g" << desc_.quant_a.group_size;
    }
    ss << "_" << to_string(desc_.type_b);  //
    if ((int)desc_.quant_b.type) {
        ss << "g" << desc_.quant_b.group_size;
    }
    ss << "_" << to_string(desc_.type_c);
    ss << "_"                                                                            //
       << (desc_.order_a == kColMajor ? 'n' : 't')                                       //
       << (desc_.order_b == kColMajor ? 'n' : 't')                                       //
       << (desc_.order_c == kColMajor ? 'n' : 't');                                      //
    ss << "_" << desc_.cta_tile.x << "x" << desc_.cta_tile.y << "x" << desc_.cta_tile.z  //
       << "_" << desc_.stages                                                            //
       << "_" << to_string(desc_.op_class)                                               //
       << "_" << desc_.mma_tile.x << "x" << desc_.mma_tile.y << "x" << desc_.mma_tile.z  //
       << "_c" << desc_.c_tile.x << "x" << desc_.c_tile.y                                //
       << "_a" << desc_.align.x << "x" << desc_.align.y << "x" << desc_.align.z          //
       << "_" << desc_.policy_a << desc_.policy_b;

    return ss.str();
}

template<class Op>
inline static bool cmp(const int3& a, const int3& b, Op op)
{
    return op(std::tie(a.x, a.y, a.z), std::tie(b.x, b.y, b.z));
}

std::vector<std::vector<LaunchSpec>> Cluster(const std::vector<LaunchSpec>& specs, const ClusteringParam& param)
{
    std::vector<const LaunchSpec*> ptrs;  // pointer into `specs`
    for (auto& s : specs) {
        ptrs.push_back(&s);
    }

    auto less = [&](const LaunchSpec* u, const LaunchSpec* v) {
        const auto& a = u->kernel->desc();
        const auto& b = v->kernel->desc();
        if (!cmp(a.cta_tile, b.cta_tile, std::equal_to<>{})) {
            return cmp(a.cta_tile, b.cta_tile, std::less<>{});
        }
        if (!cmp(a.mma_tile, b.mma_tile, std::equal_to<>{})) {
            return cmp(a.mma_tile, b.mma_tile, std::less<>{});
        }
        if (param.cache_policy) {
            const auto pa = std::tie(a.policy_a, a.policy_b);
            const auto pb = std::tie(b.policy_a, b.policy_b);
            if (pa != pb) {
                return pa < pb;
            }
        }
        if (param.max_active_ctas) {
            if (a.max_active_ctas != b.max_active_ctas) {
                return a.max_active_ctas < b.max_active_ctas;
            }
        }
        return u->splits < v->splits;
    };

    std::stable_sort(ptrs.begin(), ptrs.end(), less);

    if (ptrs.empty()) {
        return {};
    }
    std::vector<std::vector<LaunchSpec>> clusters{{*ptrs[0]}};

    auto equal = [&](const LaunchSpec* u, const LaunchSpec* v) {  //
        return !less(u, v) && !less(v, u);
    };
    int p = 0;
    for (size_t i = 1; i < ptrs.size(); ++i) {
        if (equal(ptrs[p], ptrs[i])) {
            clusters.back().push_back(*ptrs[i]);
        }
        else {
            clusters.push_back({*ptrs[i]});
            p = i;
        }
    }

    return clusters;
}

}  // namespace turbomind::gemm
