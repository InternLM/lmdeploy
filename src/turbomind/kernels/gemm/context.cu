
#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/desc.h"
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
                  operation.context && operation.context->is_dynamic_sched()};

    desc.m   = m0;
    desc.n   = n0;
    desc.k   = k0;
    desc.num = l0;

    return desc;
}

std::vector<LaunchSpec> get_swizzle(const int4& shape, const LaunchSpec& spec, const std::vector<int>& swizzle)
{
    const auto [m, n, k, _] = shape;
    std::vector<int> vec;
    for (const auto& s : swizzle) {
        auto x = spec.kernel->GetSwizzle(m, n, k, spec.splits, s);
        if (std::find(vec.begin(), vec.end(), x) == vec.end()) {
            vec.push_back(x);
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

std::vector<Kernel*> filter_by_batch_size(std::vector<Kernel*> kernels, const GemmDesc& desc, int batch_size)
{
    auto get_batch_dim = [idx = desc.batch_dim](const Kernel* k) {
        return idx == 0 ? k->desc().cta_tile.x : k->desc().cta_tile.y;
    };

    int max_batch_size = 0;
    for (const auto& k : kernels) {
        max_batch_size = std::max(get_batch_dim(k), max_batch_size);
    }
    for (const auto& k : kernels) {
        const auto x = get_batch_dim(k);
        if (x >= batch_size) {
            max_batch_size = std::min(max_batch_size, x);
        }
    }
    const auto pred = [&](auto k) { return get_batch_dim(k) > max_batch_size; };
    kernels.erase(std::remove_if(kernels.begin(), kernels.end(), pred), kernels.end());

    return kernels;
}

Context::Context(const cudaDeviceProp& prop)
{
    arch_     = prop.major * 100 + prop.minor * 10;
    sm_count_ = prop.multiProcessorCount;
}

StaticGemmContext::StaticGemmContext(const cudaDeviceProp& prop): Context{prop} {}

std::optional<GemmDesc> StaticGemmContext::Init(const Operation&    operation,
                                                const MatrixLayout& Adesc,
                                                const MatrixLayout& Udesc,
                                                const MatrixLayout& Bdesc,
                                                const MatrixLayout& Vdesc,
                                                const MatrixLayout& Cdesc,
                                                const MatrixLayout& Ddesc)
{

    desc_ = get_gemm_desc(operation, Adesc, Udesc, Bdesc, Vdesc, Cdesc, Ddesc, arch_);
    return desc_;
}

std::vector<Kernel*> StaticGemmContext::Filter(const std::vector<Kernel*>& kernels) const
{
    return filter_by_batch_size(kernels, *desc_, desc_->batch_dim == 0 ? desc_->m : desc_->n);
}

std::vector<LaunchSpec> StaticGemmContext::Populate(const Kernel& kernel, const PopulateParam& param) const
{
    if (kernel.desc().backend) {
        return {LaunchSpec{const_cast<Kernel*>(&kernel), 0, 1}};
    }

    const int m = desc_->m, n = desc_->n, k = desc_->k;

    const auto& desc = kernel.desc();

    const int64_t tiled_shape_m = cdiv(m, desc.cta_tile.x);
    const int64_t tiled_shape_n = cdiv(n, desc.cta_tile.y);
    const int     chunk_cnt_k   = cdiv(k, kernel.chunk_size_k());

    // Despite we only have sm_count * constant tensor cores, this is the granularity for scheduling
    const int   concurrency     = sm_count_ * kernel.desc().max_active_ctas;
    const float waves_per_split = float(tiled_shape_m * tiled_shape_n) / concurrency;
    const float splits_per_wave = 1.f / waves_per_split;

    // Tile quantization
    const int64_t ceil_m = tiled_shape_m * desc.cta_tile.x;
    const int64_t ceil_n = tiled_shape_n * desc.cta_tile.y;

    // int max_splits = kernel.GetMaxSplits(m, n, k, param.barriers_size, param.partials_size);
    int max_splits =
        kernel.GetMaxSplits({m, n, k, 1}, tiled_shape_m * tiled_shape_n, param.barriers_size, param.partials_size);

    // std::cout << "max_splits: " << max_splits << std::endl;

    max_splits = std::min(param.max_splits, max_splits);

    std::vector<LaunchSpec> specs;

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
        const int   grid_size    = tiled_shape_m * tiled_shape_n * splits;
        const int   full_waves   = grid_size / concurrency;
        const int   residue      = grid_size % concurrency;
        const float partial_wave = (float)cdiv(residue, sm_count_) / desc.max_active_ctas;
        const float waves        = full_waves + partial_wave;

        if (splits > 1 && waves > param.max_waves) {
            break;
        }
        // ceil(tiled_mn / C * splits) * C / tiled_mn * ceil_m * ceil_n * split_ceil_k
        const int64_t mma_cost = wave_mma_cost * waves;

        // IO has less severe quantization effect
        const int64_t mio_cost_a = byte_size(desc.type_a, tiled_shape_n * m * split_ceil_k) * splits;
        const int64_t mio_cost_b = byte_size(desc.type_b, tiled_shape_m * n * split_ceil_k) * splits;
        /// TODO: read type from `desc_.accum` when added
        const int64_t mio_cost_c = byte_size(desc.type_c, (int64_t)m * n) * (splits - 1) * 2;
        const int64_t mio_cost   = mio_cost_a + mio_cost_b + mio_cost_c;

        // std::cout << name() << " " << splits << " " << waves << " " << (float)mio_cost << " " << (float)mma_cost
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

std::vector<LaunchSpec> StaticGemmContext::Swizzle(const LaunchSpec& spec, const std::vector<int>& swizzle) const
{
    return get_swizzle({desc_->m, desc_->n, desc_->k, desc_->num}, spec, swizzle);
}

static void release(Tape& tape, cudaStream_t st)
{
    if (tape.buffer) {
        cudaFreeAsync(tape.buffer, st);
    }
    tape = {};
}

static void resize(Tape& tape, int ctas, int num, cudaStream_t st)
{
    auto allocate = [&](void* base) {
        Monotonic alloc{base};
        alloc(&tape.tile_offsets, ctas);
        alloc(&tape.iter_k_ranges, ctas);
        alloc(&tape.tile_ids, ctas);
        alloc(&tape.gemm_shapes, num);
        alloc(&tape.tiled_shapes, num);
        return (char*)alloc.ptr() - (char*)base;
    };
    if (tape.max_ctas < ctas || tape.max_num < num) {
        release(tape, st);
        const auto size = allocate(0);
        cudaMallocAsync(&tape.buffer, size, st);
        allocate(tape.buffer);
        tape.max_ctas = ctas;
        tape.max_num  = num;
    }
    tape.ctas = ctas;
}

template<Order order>
__global__ void schedule_gemm_split_k(Tape tape, GemmScheduler<order> sched, dim3 grid)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= tape.max_ctas) {
        return;
    }

    int       idx         = tid;
    const int block_idx_x = idx % grid.x;
    idx /= grid.x;
    const int block_idx_y = idx % grid.y;
    const int block_idx_z = idx / grid.y;

    sched.init(block_idx_x, block_idx_y, block_idx_z);

    if (tid == 0) {
        tape.gemm_shapes[tid]  = sched.gemm_shape();
        tape.tiled_shapes[tid] = sched.tiled_shape();
    }

    tape.tile_offsets[tid]  = sched.tile_offset();
    tape.iter_k_ranges[tid] = sched.iter_k_range();
    tape.tile_ids[tid]      = sched.tile_id();
}

DynamicGemmContext::DynamicGemmContext(const cudaDeviceProp& prop, cudaStream_t stream):
    StaticGemmContext{prop}, stream_{stream}, tape_{}
{
}

DynamicGemmContext::~DynamicGemmContext()
{
    release(tape_, stream_);
}

static inline bool operator==(int4 a, int4 b)
{
    return std::tie(a.x, a.y, a.z, a.w) == std::tie(b.x, b.y, b.z, b.w);
}

static inline bool operator==(LaunchSpec a, LaunchSpec b)
{
    return std::tie(a.kernel, a.splits, a.swizzle) == std::tie(b.kernel, b.splits, b.swizzle);
}

Tape DynamicGemmContext::Schedule(const LaunchSpec& spec)
{
    const int4 shape{desc_->m, desc_->n, desc_->k, desc_->num};
    if (shape == last_shape_ && spec == last_spec_) {
        return tape_;
    }

    const auto cta_tile     = spec.kernel->cta_tile_size();
    const auto chunk_k_size = spec.kernel->chunk_size_k();

    const int2 tiled_mn = get_tiled_shape(shape.x, shape.y, cta_tile.x, cta_tile.y);

    GemmScheduler<kColMajor> sched{shape, tiled_mn, spec.splits, spec.swizzle, cta_tile.z, chunk_k_size};

    const dim3 grid = sched.get_grid_shape();
    const int  ctas = grid.z * grid.y * grid.x;

    // std::cout << grid.x << " " << grid.y << " " << grid.x << "\n";

    resize(tape_, ctas, 1, stream_);

    constexpr int threads = 256;
    const int     blocks  = cdiv(ctas, threads);

    schedule_gemm_split_k<<<blocks, threads, 0, stream_>>>(tape_, sched, grid);

    last_shape_ = shape;
    last_spec_  = spec;

    return tape_;
}

MoeGemmContext::MoeGemmContext(int expert_num, int experts_per_token, const cudaDeviceProp& prop, cudaStream_t stream):
    Context{prop},
    expert_num_{expert_num},
    experts_per_token_{experts_per_token},
    stream_{stream},
    tokens_{},
    offsets_{},
    tape_{}
{
    resize(tape_, 256 << 10, expert_num_, stream_);
}

MoeGemmContext::~MoeGemmContext()
{
    release(tape_, stream_);
}

std::optional<GemmDesc> MoeGemmContext::Init(const Operation&    operation,
                                             const MatrixLayout& Adesc,
                                             const MatrixLayout& Udesc,
                                             const MatrixLayout& Bdesc,
                                             const MatrixLayout& Vdesc,
                                             const MatrixLayout& Cdesc,
                                             const MatrixLayout& Ddesc)
{

    desc_ = get_gemm_desc(operation, Adesc, Udesc, Bdesc, Vdesc, Cdesc, Ddesc, arch_);

    if (!desc_) {
        return {};
    }

    // fprintf(
    //     stderr, "%d %d %d vs %d %d %d\n", desc_->n, desc_->k, desc_->m, output_dim_, input_dim_, experts_per_token_);

    if (desc_->m % experts_per_token_ != 0 || desc_->num != expert_num_) {
        fprintf(stderr, "Context shape mismatch\n");
        return {};
    }

    output_dim_ = desc_->n;
    input_dim_  = desc_->k;

    // desc_->align_m = 1;  // gcd([m])
    // desc_->num     = expert_num_;

    tokens_ = desc_->m / experts_per_token_;

    // printf("tokens = %d, num = %d\n", tokens_, desc_->num);

    return desc_;
}

std::vector<LaunchSpec> MoeGemmContext::Populate(const Kernel& kernel, const PopulateParam& param) const
{
    const int n = output_dim_, k = input_dim_;

    const KernelDesc& desc = kernel.desc();

    // Note: cdiv(t * e, E) * E >= t * e
    const int batch_size = ceil_div(tokens_ * experts_per_token_, expert_num_);
    const int num        = std::min(tokens_ * experts_per_token_, expert_num_);

    const int64_t tiled_shape_m  = cdiv(batch_size, desc.cta_tile.x);
    const int64_t tiled_shape_n  = cdiv(n, desc.cta_tile.y);
    const int64_t tiled_shape_mn = tiled_shape_m * tiled_shape_n;
    const int     chunk_cnt_k    = cdiv(k, kernel.chunk_size_k());

    // Despite we only have sm_count * constant tensor cores, this is the granularity for scheduling
    const int   concurrency     = sm_count_ * kernel.desc().max_active_ctas;
    const float waves_per_split = float(tiled_shape_m * tiled_shape_n) / concurrency;
    const float splits_per_wave = 1.f / waves_per_split;

    // Tile quantization
    const int64_t ceil_m = tiled_shape_m * desc.cta_tile.x;
    const int64_t ceil_n = tiled_shape_n * desc.cta_tile.y;

    int max_splits =
        kernel.GetMaxSplits({batch_size, n, k, num}, tiled_shape_mn, param.barriers_size, param.partials_size);

    max_splits = std::min(param.max_splits, max_splits);
    // std::cout << "max_splits: " << max_splits << "\n";

    std::vector<LaunchSpec> specs;

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
        const float partial_wave = (float)cdiv(residue, sm_count_) / desc.max_active_ctas;
        const float waves        = full_waves + partial_wave;

        // std::cout << splits << " " << grid_size << " " << concurrency << " " << waves << std::endl;

        if (splits > 1 && waves > param.max_waves) {
            break;
        }
        // ceil(tiled_mn / C * splits) * C / tiled_mn * ceil_m * ceil_n * split_ceil_k
        const int64_t mma_cost = wave_mma_cost * waves;

        // IO has less severe quantization effect
        const int64_t mio_cost_a = byte_size(desc.type_a, tiled_shape_n * batch_size * split_ceil_k) * num * splits;
        const int64_t mio_cost_b = byte_size(desc.type_b, tiled_shape_m * n * split_ceil_k) * num * splits;
        /// TODO: read type from `desc_.accum` when added
        const int64_t mio_cost_c = byte_size(desc.type_c, (int64_t)batch_size * n) * num * (splits - 1) * 2;
        const int64_t mio_cost   = mio_cost_a + mio_cost_b + mio_cost_c;

        LaunchSpec spec{};
        spec.kernel    = const_cast<Kernel*>(&kernel);
        spec.splits    = splits;
        spec.swizzle   = param.swizzle;
        spec.estimated = {mio_cost, mma_cost};
        specs.push_back(spec);
    }

    return specs;
}

template<int block_dim, class Sched>
__global__ void schedule_gemm_moe(Tape       tape,
                                  const int* offsets,
                                  Sched      sched,
                                  int3       cta_tile,
                                  int        log_tile,
                                  int        expert_num,
                                  int        output_dims,
                                  int        input_dims,
                                  int        max_ctas)
{
    const int e = blockIdx.x;

    __shared__ int  shared_sum_tiles;
    __shared__ int  shared_tiles;
    __shared__ int4 shared_grid;

    {
        const int tokens = threadIdx.x <= e ? offsets[threadIdx.x + 1] - offsets[threadIdx.x] : 0;

        // Update tiled shape according to actual batch size
        auto tiled_shape = sched.tiled_shape();
        tiled_shape.x    = cdiv(tokens, cta_tile.x);

        const dim3 grid = sched.get_grid_shape(tiled_shape, log_tile);

        // Num of tiles for the expert
        const int tiles = grid.x * grid.y;

        using BlockReduce = cub::BlockReduce<int2, block_dim>;

        __shared__ typename BlockReduce::TempStorage temp_storage;

        auto plus = [](int2 a, int2 b) -> int2 { return {a.x + b.x, a.y + b.y}; };

        // Sum tokens/cta_m in [0, e) range (exclusive)
        // Not sure if `num_valid = 0` works (as no init value is supplied). Conditioning is used instead.
        const int2 sum_tokens_tiles =
            BlockReduce{temp_storage}.Reduce(threadIdx.x < e ? int2{tokens, tiles} : int2{}, plus);

        // Only thread-0 has the reduced value
        if (threadIdx.x == 0) {
            shared_sum_tiles = sum_tokens_tiles.y;
        }

        // Shared properties of current expert
        if (threadIdx.x == e) {
            shared_grid = {(int)grid.x, (int)grid.y, (int)grid.z, 1};

            shared_tiles = tiles;

            tape.gemm_shapes[e]  = {tokens, output_dims, input_dims, 1};
            tape.tiled_shapes[e] = tiled_shape;
        }
    }

    __syncthreads();

    const int4 grid   = shared_grid;
    const int  ctas   = shared_tiles * grid.z;
    const int  tiles  = shared_sum_tiles;
    const int  offset = shared_sum_tiles * grid.z;

    for (int i = threadIdx.x; i < ctas; i += block_dim) {
        int idx = i;

        // We need fast div-mod
        const int block_idx_x = idx % grid.x;

        idx = idx / grid.x;

        const int block_idx_y = idx % grid.y;
        const int block_idx_z = idx / grid.y;

        sched.init(block_idx_x, block_idx_y, block_idx_z);

        auto tile_offset = sched.tile_offset();
        tile_offset.w    = e;

        tape.tile_offsets[offset + i]  = tile_offset;
        tape.iter_k_ranges[offset + i] = sched.iter_k_range();
        tape.tile_ids[offset + i]      = tiles + sched.tile_id();
    }

    if (e == expert_num - 1) {
        for (int i = threadIdx.x + offset + ctas; i < max_ctas; i += block_dim) {
            tape.tile_offsets[i]  = int4{-1, -1, -1, -1};
            tape.iter_k_ranges[i] = int2{-1, -1};
            tape.tile_ids[i]      = -1;
        }
    }
}

Tape MoeGemmContext::Schedule(const LaunchSpec& spec)
{
    const int3 cta_tile = spec.kernel->cta_tile_size();

    const int sum_m = tokens_ * experts_per_token_;

    const int max_m_tiles = sum_m / cta_tile.x + std::min(sum_m, expert_num_);

    const int proxy_m = max_m_tiles * cta_tile.x;

    using Sched = GemmScheduler<kColMajor>;

    const int4 gemm_shape{proxy_m, output_dim_, input_dim_, 1};
    const int2 tiled_mn = get_tiled_shape(proxy_m, output_dim_, cta_tile.x, cta_tile.y);

    const int4 tiled_shape{tiled_mn.x, tiled_mn.y, spec.splits, 1};
    const dim3 grid = Sched::get_grid_shape(tiled_shape, spec.swizzle);

    // std::cout << "splits: " << spec.splits << std::endl;
    // std::cout << tiled_mn.x << " " << tiled_mn.y << " " << grid.x << " " << grid.y << " " << grid.z << std::endl;

    const int ctas = grid.x * grid.y * grid.z;

    resize(tape_, ctas, expert_num_, stream_);

    Sched sched{gemm_shape, tiled_mn, spec.splits, spec.swizzle, cta_tile.z, spec.kernel->chunk_size_k()};

    constexpr int threads = 256;
    const int     blocks  = expert_num_;
    schedule_gemm_moe<threads><<<blocks, threads, 0, stream_>>>(tape_,  //
                                                                offsets_,
                                                                sched,
                                                                cta_tile,
                                                                spec.swizzle,
                                                                expert_num_,
                                                                output_dim_,
                                                                input_dim_,
                                                                ctas);

    return tape_;
}

std::vector<Kernel*> MoeGemmContext::Filter(const std::vector<Kernel*>& kernels) const
{
    // const int avg_m = cdiv(tokens_ * experts_per_token_, expert_num_);
    const int max_m = cdiv(tokens_ * experts_per_token_, 1);
    return filter_by_batch_size(kernels, *desc_, desc_->batch_dim == 0 ? max_m : desc_->n);
}

std::vector<LaunchSpec> MoeGemmContext::Swizzle(const LaunchSpec& spec, const std::vector<int>& swizzle) const
{
    const int avg_m = cdiv(tokens_ * experts_per_token_, expert_num_);
    return get_swizzle({avg_m, desc_->n, desc_->k, desc_->num}, spec, swizzle);
}

}  // namespace turbomind::gemm
