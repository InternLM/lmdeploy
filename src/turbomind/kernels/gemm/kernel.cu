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

std::vector<std::pair<int, float>> Kernel::Estimate(int   m,
                                                    int   n,
                                                    int   k,
                                                    int   max_split_k,
                                                    int   sm_count,
                                                    int   max_wave_count,
                                                    int   top_k,
                                                    float bytes_per_second,
                                                    float fma_per_second)
{
    const int tiled_shape_m = ceil_div(m, desc_.cta_tile.x);
    const int tiled_shape_n = ceil_div(n, desc_.cta_tile.y);
    const int chunk_cnt_k   = ceil_div(k, chunk_size_k_);

    // std::cout << tiled_shape_m << " " << tiled_shape_n << " " << chunk_cnt_k << std::endl;

    const int     cta_m  = desc_.cta_tile.x;
    const int     cta_n  = desc_.cta_tile.y;
    const int64_t cta_mn = cta_m * cta_n;

    const int tiled_shape_mn = tiled_shape_m * tiled_shape_n;
    // const int concurrency    = sm_count * std::min(2, max_active_ctas_);
    // const int   concurrency    = sm_count;
    const int   concurrency    = sm_count * max_active_ctas_;
    const float wave_per_split = float(tiled_shape_mn) / float(concurrency);

    //                     cost    volume   waves  split-k
    std::vector<std::tuple<float, int64_t, float, int>> estimations;

    for (int splits = 1; splits <= max_split_k; ++splits) {
        const float waves      = wave_per_split * splits;
        const int   wave_count = (int)std::ceil(waves);

        if (splits > 1 && wave_count > max_wave_count) {
            break;
        }

        const int64_t gemm_size_k = ceil_div(chunk_cnt_k, splits) * chunk_size_k_;

        const int64_t fma_volume   = cta_mn * gemm_size_k;  // cta volume
        const int64_t ldg_volume   = (cta_m * 2 + cta_n / 2) * gemm_size_k;
        const int64_t split_volume = 2 * cta_mn * 4;  // 2 for W/R, 4 for float

        // (CTA_M * CTA_N) * ceil(K / S) * ceil(S * TILED_M * TILED_N / C) * C
        const float fma_cost = fma_volume * wave_count * concurrency / fma_per_second;

        // (CTA_M + CTA_N) * ceil(K / S) * ceil(S * TILED_M * TILED_N / C) * C
        const float ldg_cost = ldg_volume * wave_count * concurrency / bytes_per_second;

        // (CTA_M * CTA_N) * ceil(S * TILED_M * TILED_N / C) * C
        // const float split_cost = splits > 1 ? split_volume * wave_count * concurrency / bytes_per_second : 0;

        // (CTA_M * CTA_N) * (S - 1) * ceil(TILED_M * TILED_N / C) * C
        const float split_cost =
            split_volume * (splits - 1) * std::ceil(wave_per_split) * concurrency / bytes_per_second;

        // (CTA_M * CTA_N) * ceil(TILED_M * TILED_N / C) * C
        const float output_volume = cta_mn * 2;
        const float output_cost   = output_volume * std::ceil(wave_per_split) * concurrency / bytes_per_second;

        const float epi_cost = split_cost + output_cost;

        // Non-perfect latency hiding
        float cost = std::pow(fma_cost + ldg_cost, 0.2) * std::pow(std::max(fma_cost, ldg_cost), 0.8) + epi_cost;

        // std::cout << splits << " waves=" << waves << " fma=" << fma_cost * 1e3 << " ldg=" << ldg_cost * 1e3
        //           << " spk=" << split_cost * 1e3 << " cost=" << cost * 1e3 << std::endl;

        estimations.emplace_back(cost, 0, waves, splits);

        float w = (float)tiled_shape_mn / concurrency;
        if (splits == 1 && std::ceil(w) - w < 0.05) {
            break;
        }
    }

    for (auto i = (int)estimations.size() - 1; i >= 1; --i) {
        if (std::get<0>(estimations[i]) == std::get<0>(estimations[i - 1])) {
            std::get<0>(estimations[i]) = -1;
        }
    }
    estimations.erase(
        std::remove_if(estimations.begin(), estimations.end(), [](auto e) { return std::get<0>(e) == -1; }),
        estimations.end());

    top_k = std::min<int>(top_k, estimations.size());

    std::vector<int> idxs(estimations.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    std::partial_sort(idxs.begin(), idxs.begin() + top_k, idxs.end(), [&](int i, int j) {  //
        return estimations[i] < estimations[j];
    });

    std::vector<std::pair<int, float>> ret;
    ret.reserve(top_k);

    for (int i = 0; i < top_k; ++i) {
        auto& [cost, volume, waves, splits] = estimations[idxs[i]];
        if (i == 0) {
            // std::cout << "* " << cost * 1e3 << " " << volume << " " << waves << " " << splits << std::endl;
        }
        ret.emplace_back(splits, cost);
    }

    return ret;
}

bool Kernel::is_feasible(const GemmDesc& desc) const noexcept
{
    constexpr bool debug = false;

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

    return true;
}

static int64_t get_size(DataType type, int64_t size)
{
    switch (type) {
        case DataType::U64:
            return size * 8;
        case DataType::F32:
        case DataType::U32:
            return size * 4;
        case DataType::BF16:
        case DataType::F16:
        case DataType::U16:
            return size * 2;
        case DataType::U8:
        case DataType::F8_E4M3:
        case DataType::F8_E5M2:
            return size;
        case DataType::U4:
            return size / 2;
        default:
            std::cerr << to_string(type) << "\n";
            return -1;
    }
}

int64_t Kernel::GetTilingCost(const std::array<int, 3>& size) const
{
    const auto [m, n, k] = size;

    const int64_t tiled_shape_m = ceil_div(m, desc_.cta_tile.x);
    const int64_t tiled_shape_n = ceil_div(n, desc_.cta_tile.y);

    const int ceil_m = m;  // round_up(m, desc_.cta_tile.x);
    const int ceil_n = n;  // round_up(n, desc_.cta_tile.y);

    const int64_t cost_n = get_size(desc_.type_a, tiled_shape_m * ceil_n * k);
    const int64_t cost_m = get_size(desc_.type_b, tiled_shape_n * ceil_m * k);

    return cost_n + cost_m;
}

std::vector<std::pair<int, int64_t>>
Kernel::GetSplitingCost(const std::array<int, 3>& size, int max_splits, int max_waves, int sm_count) const
{
    const auto [m, n, k] = size;

    const int tiled_shape_m = ceil_div(m, desc_.cta_tile.x);
    const int tiled_shape_n = ceil_div(n, desc_.cta_tile.y);
    const int chunk_cnt_k   = ceil_div(k, chunk_size_k_);

    std::vector<std::pair<int, int64_t>> ret;

    // Dispite we only have sm_count * constant tensor cores, this is the granularity for scheduling
    const float concurrency    = sm_count * max_active_ctas_;
    const float wave_per_split = float(tiled_shape_m * tiled_shape_n) / concurrency;
    const float split_per_wave = 1.f / wave_per_split;

    // Tile quantization
    const int64_t ceil_m = tiled_shape_m * desc_.cta_tile.x;
    const int64_t ceil_n = tiled_shape_n * desc_.cta_tile.y;

    for (int splits = 1; splits <= max_splits; ++splits) {
        // Split quantization
        const int64_t split_ceil_k = ceil_div(chunk_cnt_k, splits) * chunk_size_k_;
        // Footprint for single split
        const float split_cost = ceil_m * ceil_n * split_ceil_k;
        // Footprint for single wave
        const float wave_cost = split_cost * split_per_wave;
        // Wave quantization
        const int waves = (int)std::ceil(wave_per_split * splits);
        if (splits > 1 && waves > max_waves) {
            break;
        }
        // ceil(tiled_mn / C * splits) * C / tiled_mn * ceil_m * ceil_n * split_ceil_k
        const float cost = wave_cost * waves;

        std::cout << name() << " " << splits << " " << cost << "\n";

        ret.emplace_back(splits, cost);
    }

    std::stable_sort(ret.begin(), ret.end(), [](auto a, auto b) {  //
        return a.second < b.second;
    });

    return ret;
}

std::vector<std::pair<int, KernelMetric>>
Kernel::Estimate_v2(std::array<int, 3> size, int max_splits, int max_waves, int sm_count) const
{
    const auto [m, n, k]        = size;
    const int64_t tiled_shape_m = ceil_div(m, desc_.cta_tile.x);
    const int64_t tiled_shape_n = ceil_div(n, desc_.cta_tile.y);
    const int     chunk_cnt_k   = ceil_div(k, chunk_size_k_);

    // Dispite we only have sm_count * constant tensor cores, this is the granularity for scheduling
    const int   concurrency     = sm_count * max_active_ctas_;
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
        const float partial_wave = (float)ceil_div(residue, sm_count) / max_active_ctas_;
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

        // std::cout << name() << " " << splits << " " << (float)mio_cost << " " << (float)mma_cost << "\n";

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
        if (param.stages && a.stages != b.stages) {
            return a.stages < b.stages;
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
