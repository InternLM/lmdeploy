
#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include "src/turbomind/utils/monotonic.h"
#include <algorithm>
#include <cub/block/block_reduce.cuh>
#include <iostream>
#include <tuple>

namespace turbomind::gemm {

static std::optional<GemmDesc> get_gemm_desc(const Operation&    operation,
                                             const MatrixLayout& Adesc,
                                             const MatrixLayout& Udesc,
                                             const MatrixLayout& Bdesc,
                                             const MatrixLayout& Vdesc,
                                             const MatrixLayout& Cdesc,
                                             const MatrixLayout& Ddesc,
                                             int                 arch)
{

    // Constant dimensions are set to the exact value
    // Variable dimensions are set to sum of the values

    const int m0 = Adesc.rows, k0 = Adesc.cols;
    const int k1 = Bdesc.rows, n0 = Bdesc.cols;
    const int m1 = Ddesc.rows, n1 = Ddesc.cols;

    const int l0 = Adesc.num, l1 = Bdesc.num, l2 = Ddesc.num;

    if (m0 != m1 || n0 != n1 || k0 != k1 || l0 != l1 || l0 != l2) {
        fprintf(stderr, "%d %d %d %d %d %d %d %d %d\n", m0, m1, n0, n1, k0, k1, l0, l1, l2);
        return {};
    }

    GemmDesc desc{arch,
                  Adesc.type,
                  Bdesc.type,
                  Ddesc.type,
                  Adesc.order,
                  Bdesc.order,
                  Ddesc.order,
                  get_mode(Adesc),
                  get_mode(Bdesc),
                  get_mode(Ddesc),
                  Adesc.pack,
                  Bdesc.pack,
                  Udesc.pack,
                  Vdesc.pack,
                  operation.quant_a,
                  operation.quant_b,
                  operation.epilogue,
                  operation.batch_dim,
                  -1};

    desc.m   = m0;
    desc.n   = n0;
    desc.k   = k0;
    desc.num = std::max(l0, 1);

    if (desc.num > 1) {
        desc.group_axis = operation.batch_dim;
    }

    return desc;
}

std::vector<LaunchSpec> get_swizzle(const int4& shape, const LaunchSpec& spec, const std::vector<int>& swizzle)
{
    std::vector<int> vec;
    const int        max_swizzle = spec.kernel->GetMaxSwizzle(shape);
    for (const auto& s : swizzle) {
        if (s <= max_swizzle && std::find(vec.begin(), vec.end(), s) == vec.end()) {
            vec.push_back(s);
        }
    }
    std::vector<LaunchSpec> ret;
    for (const auto& s : vec) {
        auto tmp    = spec;
        tmp.swizzle = s;
        ret.push_back(tmp);
    }
    return ret;
}

Context::Context(const cudaDeviceProp& prop)
{
    arch_     = prop.major * 100 + prop.minor * 10;
    sm_count_ = prop.multiProcessorCount;
}

bool Context::Init(const Operation&    operation,
                   const MatrixLayout& Adesc,
                   const MatrixLayout& Udesc,
                   const MatrixLayout& Bdesc,
                   const MatrixLayout& Vdesc,
                   const MatrixLayout& Cdesc,
                   const MatrixLayout& Ddesc)
{
    auto desc = get_gemm_desc(operation, Adesc, Udesc, Bdesc, Vdesc, Cdesc, Ddesc, arch_);
    if (!desc) {
        return false;
    }

    desc_       = *desc;
    desc_trans_ = transpose(desc_);

    return true;
}

std::vector<Kernel*> Context::Filter(const std::vector<Kernel*>& kernels) const
{
    std::vector<std::pair<Kernel*, int>> feasible;
    auto get_batch_dim  = [](auto k, auto& g) { return g.batch_dim ? k->desc().cta_tile.y : k->desc().cta_tile.x; };
    int  max_batch_size = 0;  // max batch size of single CTA tile

    for (auto& k : kernels) {
        auto& g = get_desc(*k);
        if (k->is_feasible(g)) {
            auto bsz = get_batch_dim(k, g);
            feasible.emplace_back(k, bsz);
            max_batch_size = std::max(bsz, max_batch_size);
        }
    }

    // Batch size of the GEMM problem
    const int batch_size = desc_.batch_dim ? desc_.n : desc_.m;
    // std::cout << "BATCH SIZE: " << batch_size << "\n";

    // Find smallest kernel the problem can fit into (may not exist)
    for (const auto& [k, bsz] : feasible) {
        if (bsz >= batch_size) {
            max_batch_size = std::min(max_batch_size, bsz);
        }
    }

    const auto pred = [&](auto k) {  //
        return k.second > max_batch_size;
    };
    feasible.erase(std::remove_if(feasible.begin(), feasible.end(), pred), feasible.end());

    std::vector<Kernel*> ret;
    for (auto& [k, bsz] : feasible) {
        // std::cout << "KERNEL: " << k->name() << ", BSZ: " << bsz << std::endl;
        ret.push_back(k);
    }

    return ret;
}

std::vector<LaunchSpec> Context::Populate(const Kernel& kernel, const PopulateParam& param) const
{
    // early exit for cuBLAS backend
    if (kernel.desc().backend) {
        return {LaunchSpec{const_cast<Kernel*>(&kernel), 0, 1}};
    }

    const auto& gemm = get_desc(kernel);

    const int m = gemm.m, n = gemm.n, k = gemm.k, num = std::max(1, gemm.num);

    const auto& desc = kernel.desc();
    const auto& info = kernel.info();

    const int64_t tiled_shape_m = cdiv(m, desc.cta_tile.x * (desc.group_axis == 0 ? num : 1));
    const int64_t tiled_shape_n = cdiv(n, desc.cta_tile.y * (desc.group_axis == 1 ? num : 1));
    const int     chunk_cnt_k   = cdiv(k, kernel.chunk_size_k());

    // Despite we only have sm_count * constant tensor cores, this is the granularity for scheduling
    const int   concurrency     = sm_count_ * kernel.info().max_active_ctas;
    const float waves_per_split = float(tiled_shape_m * tiled_shape_n) / concurrency;
    const float splits_per_wave = 1.f / waves_per_split;

    // Tile quantization
    const int64_t ceil_m = tiled_shape_m * desc.cta_tile.x;
    const int64_t ceil_n = tiled_shape_n * desc.cta_tile.y;

    // int max_splits = kernel.GetMaxSplits(m, n, k, param.barriers_size, param.partials_size);
    int max_splits = kernel.GetMaxSplits({m, n, k, num}, 0, param.barriers_size, param.partials_size);

    // std::cout << "max_splits: " << max_splits << std::endl;

    max_splits = std::min(param.max_splits, max_splits);

    std::vector<LaunchSpec> specs;

    /// TODO: revise this according to the lastest scheduler
    for (int splits = 1; splits <= max_splits; ++splits) {
        // Split quantization, penalize uneven splits
        const int64_t split_ceil_k = cdiv(chunk_cnt_k, splits) * kernel.chunk_size_k();
        // Footprint for single split
        const int64_t split_mma_cost = ceil_m * ceil_n * split_ceil_k;
        // Footprint for single wave
        const int64_t wave_mma_cost = split_mma_cost * splits_per_wave;

        // Wave quantization
        // const int waves = (int)std::ceil(wave_per_split * splits);

        // Bold simulation of thread block scheduling
        const int   grid_size    = tiled_shape_m * tiled_shape_n * splits * num;
        const int   full_waves   = grid_size / concurrency;
        const int   residue      = grid_size % concurrency;
        const float partial_wave = (float)cdiv(residue, sm_count_) / info.max_active_ctas;
        const float waves        = full_waves + partial_wave;

        if (splits > 1 && waves > param.max_waves) {
            break;
        }
        // ceil(tiled_mn / C * splits) * C / tiled_mn * ceil_m * ceil_n * split_ceil_k
        const int64_t mma_cost = wave_mma_cost * waves;

        // IO has less severe quantization effect
        const int64_t mio_cost_a = byte_size(desc.type_a, tiled_shape_n * m * split_ceil_k) * splits * num;
        const int64_t mio_cost_b = byte_size(desc.type_b, tiled_shape_m * n * split_ceil_k) * splits * num;
        /// TODO: read type from `desc_.accum` when added
        const int64_t mio_cost_c = byte_size(desc.type_c, (int64_t)m * n) * (splits - 1) * 2 * num;
        const int64_t mio_cost   = mio_cost_a + mio_cost_b + mio_cost_c;

        // std::cout << kernel.name() << " " << splits << " " << waves << " " << (float)mio_cost << " " <<
        // (float)mma_cost
        //           << "\n";

        // metrics.emplace_back(splits, KernelMetric{mio_cost, mma_cost});

        LaunchSpec spec{};
        spec.kernel    = const_cast<Kernel*>(&kernel);
        spec.splits    = splits;
        spec.swizzle   = param.swizzle;
        spec.estimated = {mio_cost, mma_cost};
        specs.push_back(spec);
    }

    return specs;
}

std::vector<LaunchSpec> Context::Swizzle(const LaunchSpec& spec, const std::vector<int>& swizzle) const
{
    auto& desc = get_desc(*spec.kernel);
    return get_swizzle({desc.m, desc.n, desc.k, desc.num}, spec, swizzle);
}

}  // namespace turbomind::gemm
