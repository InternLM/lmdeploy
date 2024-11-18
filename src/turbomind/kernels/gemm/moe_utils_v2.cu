// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_scan.cuh>

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"

namespace turbomind {

template<int top_k, int block_dim>
__global__ void MoeGateKernel_V2(float*       scales,  // [e,n]
                                 int8_t*      masks,   // [E,n], padded
                                 int*         accum,   // [E,tiles]
                                 const float* logits,  // [E,n]
                                 int          log_tile,
                                 int          tiles,
                                 int          tokens,
                                 int          tokens_padded,
                                 int          experts)
{
    constexpr int max_tiles = kMoeGateMaxTiles;

    // Brute-force per thread top-k using a flat thread mapping
    const int ti = threadIdx.x + blockIdx.x * blockDim.x;

    // Clear masks
    for (int e = 0; e < experts; ++e) {
        if (ti < tokens_padded) {
            masks[e * tokens_padded + ti] = -1;
        }
    }

    __shared__ int shared_accum[32][max_tiles];

    for (int i = threadIdx.x; i < experts * max_tiles; i += block_dim) {
        int e = i / max_tiles;
        int t = i % max_tiles;
        if (e < experts && t < tiles) {
            shared_accum[e][t] = 0;
        }
    }

    __syncthreads();

    if (ti < tokens) {

        static_assert(top_k <= 32);
        int mask = -1;

        float max_logit = 0.f;

        // Find top-k
        PRAGMA_UNROLL
        for (int k = 0; k < top_k; ++k) {
            int   max_bit = 0;
            float max_val = -std::numeric_limits<float>::infinity();
            int   bit     = 1;
            for (int e = 0; e < experts; ++e) {
                const auto val = logits[ti * experts + e];
                // const auto val = logits[e * tokens + ti];
                if ((mask & bit) && val > max_val) {
                    max_bit = bit;
                    max_val = val;
                }
                bit *= 2;
            }
            mask -= max_bit;
            if (k == 0) {
                max_logit = max_val;
            }
        }

        mask = ~mask;

        Array<float, top_k> top_val;
        PRAGMA_UNROLL
        for (int i = 0; i < top_k; ++i) {
            const int lowbit = (mask & -mask);
            const int e      = 31 - __clz(lowbit);

            // printf("e = %d, ti = %d, idx = %d\n", e, ti, i);

            masks[e * tokens_padded + ti] = i;
            atomicAdd(&shared_accum[e][ti >> log_tile], 1);
            top_val[i] = logits[ti * experts + e];
            // top_val[i] = logits[e * tokens + ti];

            mask -= lowbit;
        }

        float prob_sum = 0.f;
        PRAGMA_UNROLL
        for (int i = 0; i < top_k; ++i) {
            top_val[i] = expf(top_val[i] - max_logit);
            prob_sum += top_val[i];
        }

        PRAGMA_UNROLL
        for (int i = 0; i < top_k; ++i) {
            scales[i * tokens + ti] = fdividef(top_val[i], prob_sum);
        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < experts * max_tiles; i += block_dim) {
        int e = i / max_tiles;
        int t = i % max_tiles;
        if (e < experts && t < tiles) {
            atomicAdd(accum + e * tiles + t, shared_accum[e][t]);
        }
    }
}

template<int block_dim, class Mask>
__global__ void MoeScanKernel_v2(int*       f2n,      // [e*n]
                                 int*       en2f,     // [e,n]
                                 int*       offsets,  // [E+1]
                                 Mask*      masks,    // [E,n], padded
                                 const int* accum,    // [E,tiles]
                                 int        log_tile,
                                 int        tiles,
                                 int        tokens,
                                 int        tokens_padded,
                                 int        experts)
{
    using BlockReduce = cub::BlockReduce<int, block_dim>;
    using BlockScan   = cub::BlockScan<int, block_dim>;

    __shared__ union TempStorage {
        typename BlockReduce::TempStorage reduce;
        typename BlockScan::TempStorage   scan;
    } temp_storage;

    constexpr int vec_size = kMoeGateVecSize;

    using Vec = Array<Mask, vec_size>;

    const int tile_id = blockIdx.x;
    const int ei      = blockIdx.y;

    const int  global_tile_id = ei * tiles + tile_id;
    const bool is_valid       = global_tile_id <= experts * tiles;

#if 0
    int vacc[4]{};
    {
        int idx = threadIdx.x;
        PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
            if (idx < global_tile_id) {
                vacc[i] = accum[idx];
            }
            idx += block_dim;
        }
    }

    int offset = BlockReduce{temp_storage.reduce}.Sum(vacc);
#else

    int vacc = 0;
    for (int i = threadIdx.x; i < global_tile_id; i += block_dim) {
        if (is_valid && i < global_tile_id) {
            vacc += accum[i];
        }
    }

    int offset = BlockReduce{temp_storage.reduce}.Sum(vacc);

#endif

    __shared__ int shared_offset;

    if (threadIdx.x == 0) {
        shared_offset = offset;
        if (tile_id == 0) {
            offsets[ei] = offset;
        }
    }

    if (ei == experts) {
        return;
    }

    __syncthreads();

    offset = shared_offset;

    const int token_vecs = tokens_padded / vec_size;

    const int tile_size     = 1 << log_tile;
    const int tile_vec_size = tile_size / vec_size;

    const int tile_vec_beg    = tile_id * tile_vec_size;
    const int tile_vec_end    = std::min(tile_vec_beg + tile_vec_size, token_vecs);
    const int tile_vec_padded = tile_vec_beg + round_up(tile_vec_size, block_dim);

    // if (threadIdx.x == 0) {
    //     printf("%d %d %d\n", tile_vec_beg, tile_vec_end, tile_vec_padded);
    // }

    auto mask_ptr = (Vec*)masks + ei * token_vecs;

    for (int vi = tile_vec_beg + threadIdx.x; vi < tile_vec_padded; vi += block_dim) {

        const bool pred = vi < tile_vec_end;

        Vec data;
        fill(data, Mask{-1});
        if (pred) {
            Ldg(data, mask_ptr[vi].data());
        }

        int prefix[vec_size];
        PRAGMA_UNROLL
        for (int i = 0; i < vec_size; ++i) {
            prefix[i] = int(data[i] >= 0);
        }

        int block_sum = 0;

        BlockScan{temp_storage.scan}.ExclusiveSum(prefix, prefix, block_sum);
        __syncthreads();

        PRAGMA_UNROLL
        for (int i = 0; i < vec_size; ++i) {
            if (pred && data[i] >= 0) {
                const int flat_id = prefix[i] + offset;
                const int ti      = vi * vec_size + i;
                f2n[flat_id]      = ti;
                // No ti is generated for padded tokens so we are safe
                en2f[data[i] * tokens + ti] = flat_id;
            }
        }

        offset += block_sum;
    }
}

template<int max_expert_num,
         int max_top_k,
         //  bool norm_top_k,
         int items_per_thread,
         int block_dim,
         int access_size,
         class Mask>
__global__ void MoeGateKernel_v8(float*       scales,  // [e,n]
                                 Mask*        masks,   // [E,n], padded
                                 int*         accum,   // [E,tiles]
                                 const float* logits,  // [n,E]
                                 int          log_tile,
                                 int          tiles,
                                 int          token_num,
                                 int          token_num_padded,
                                 int          expert_num,
                                 int          top_k,
                                 bool         norm_topk)
{
    constexpr int max_tiles         = kMoeGateMaxTiles;
    constexpr int threads_per_token = max_expert_num / items_per_thread;  // 8
    constexpr int tokens_per_cta    = block_dim / threads_per_token;

    // We use bits in a uint32_t to represent selected experts
    static_assert(items_per_thread <= 32);
    // We use warp-level primitives for reduction
    static_assert(threads_per_token <= 32);

    static_assert((threads_per_token & (threads_per_token - 1)) == 0);

    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int ti = thread_idx / threads_per_token;
    const int ei = thread_idx % threads_per_token;

    const int bti = threadIdx.x / threads_per_token;

    const int warp_ti = threadIdx.x % WARP_SIZE / threads_per_token;

    const int warp_offset  = thread_idx / WARP_SIZE * WARP_SIZE / threads_per_token;
    const int block_offset = thread_idx / block_dim * block_dim / threads_per_token;

    float data[items_per_thread];
    int   idxs[items_per_thread];

#if 0
    PRAGMA_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        data[i] = -std::numeric_limits<float>::infinity();
        idxs[i] = threads_per_token * (i / access_size * access_size) + i % access_size + ei * access_size;
    }
    if (ti < token_num) {
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; i += access_size) {
            const int e = threads_per_token * i + ei * access_size;
            if (e < expert_num) {
                Ldg((Array<float, access_size>&)data[i], &logits[ti * expert_num + e]);
            }
        }
    }

    __shared__ union {
        struct {
            // +1 padding greatly reduced (-80%) bank conflicts
            int   shared_accum[max_tiles][max_expert_num + 1];
            float shared_scales[max_top_k][tokens_per_cta];
            int   shared_exp_id[max_top_k][tokens_per_cta];
        };
    } smem;
#elif 1
    PRAGMA_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        data[i] = -std::numeric_limits<float>::infinity();
        // idxs[i] = threads_per_token * (i / access_size * access_size) + i % access_size + ei * access_size;
        idxs[i] = ei * items_per_thread + i;
    }
    if (ti < token_num) {
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; i += access_size) {
            // const int e = threads_per_token * i + ei * access_size;
            const int e = ei * items_per_thread + i;
            if (e < expert_num) {
                Ldg((Array<float, access_size>&)data[i], &logits[ti * expert_num + e]);
            }
        }
    }

    __shared__ union {
        struct {
            // +1 padding greatly reduced (-80%) bank conflicts
            int   shared_accum[max_tiles][max_expert_num + 1];
            float shared_scales[max_top_k][tokens_per_cta];
            int   shared_exp_id[max_top_k][tokens_per_cta];
        };
    } smem;
#else

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    constexpr int vecs_per_thread = items_per_thread / access_size;

    using Vec            = Array<float, access_size>;
    constexpr int banks  = 128 / sizeof(Vec);
    constexpr int chunks = 4;  // block_dim / WARP_SIZE;

    __shared__ union {
        Vec shared_data[chunks][vecs_per_thread * WARP_SIZE / banks][banks + 1];
        struct {
            // +1 padding greatly reduced (-80%) bank conflicts
            int   shared_accum[max_tiles][max_expert_num + 1];
            float shared_scales[max_top_k][tokens_per_cta];
            int   shared_exp_id[max_top_k][tokens_per_cta];
        };
    } smem;

    __align__(16) Vec vecs[vecs_per_thread];

    {
        const int warp_end = min(warp_offset + WARP_SIZE / threads_per_token, token_num) * expert_num;
        int       p        = warp_offset * expert_num + access_size * lane_id;
        PRAGMA_UNROLL
        for (int i = 0; i < vecs_per_thread; ++i) {
            fill(vecs[i], -std::numeric_limits<float>::infinity());
            // const int p = warp_offset * expert_num + access_size * (lane_id + i * WARP_SIZE);
            if (p < warp_end) {
                Ldg(vecs[i], &logits[p]);
            }
            p += access_size * WARP_SIZE;
        }
    }

    PRAGMA_UNROLL
    for (int c = 0; c < block_dim / WARP_SIZE; c += chunks) {
        PRAGMA_UNROLL
        for (int i = 0; i < vecs_per_thread; ++i) {
            int p = i * WARP_SIZE + lane_id;
            if (c <= warp_id && warp_id < c + chunks) {
                Store(smem.shared_data[warp_id - c][p / banks][p % banks].data(), vecs[i]);
            }
        }

        __syncwarp();

        PRAGMA_UNROLL
        for (int i = 0; i < vecs_per_thread; ++i) {
            int p = lane_id * vecs_per_thread + i;
            if (c <= warp_id && warp_id < c + chunks) {
                Load(vecs[i], smem.shared_data[warp_id - c][p / banks][p % banks].data());
            }
        }

        __syncthreads();
    }

    PRAGMA_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        idxs[i] = ei * items_per_thread + i;
    }
    PRAGMA_UNROLL
    for (int i = 0; i < vecs_per_thread; ++i) {
        (Array<float, access_size>&)data[i * access_size] = vecs[i];
    }

#endif

    constexpr float kLog2e = 1.4426950408889634074;

    unsigned mask = (unsigned)-1;
    float    max_logit;

    int   count{};
    float sum_prob{};

    const int warp_ti_offset = warp_ti * threads_per_token;

    auto run = [&](int k) {
        unsigned bit     = 1;
        unsigned max_bit = 0;
        float    max_val = -std::numeric_limits<float>::infinity();
        // local maximum
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            if ((mask & bit) && data[i] > max_val) {
                max_bit = bit;
                max_val = data[i];
            }
            asm("shl.b32 %0, %1, 1;\n" : "=r"(bit) : "r"(bit));
        }

        if (k == 0) {
            PRAGMA_UNROLL
            for (int i = 0; i < items_per_thread; ++i) {
                data[i] *= kLog2e;
            }
        }

        int   g_max_ei  = ei;
        float g_max_val = max_val;
        if constexpr (threads_per_token > 1) {
            // global maximum
            PRAGMA_UNROLL
            for (int m = threads_per_token / 2; m >= 1; m /= 2) {
                g_max_val = fmaxf(g_max_val, __shfl_xor_sync((uint32_t)-1, g_max_val, m));
            }
            // tie breaking
            const auto active = __ballot_sync((uint32_t)-1, max_val == g_max_val);
            g_max_ei          = __ffs(active >> (unsigned)warp_ti_offset) - 1;
        }
        if (k == 0) {
            max_logit = g_max_val;
        }
        if (ei == g_max_ei) {
            mask -= max_bit;
            ++count;
        }
    };

    run(0);

    for (int k = 1; k < top_k; ++k) {
        run(k);
    }

    mask = ~mask;

    int used[items_per_thread];
    {
        unsigned bit = 1;
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            used[i] = (mask & bit) > 0;
            asm("shl.b32 %0, %1, 1;\n" : "=r"(bit) : "r"(bit));
        }
    }

    PRAGMA_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        if (!norm_topk || used[i]) {
            data[i] = exp2f(data[i] - max_logit);
            sum_prob += data[i];
        }
    }

    PRAGMA_UNROLL
    for (int m = threads_per_token / 2; m >= 1; m /= 2) {
        sum_prob += __shfl_xor_sync((uint32_t)-1, sum_prob, m);
    }

    sum_prob = fdividef(1.f, sum_prob);

    using WarpScan = cub::WarpScan<int, threads_per_token>;
    __shared__ typename WarpScan::TempStorage temp_storage[tokens_per_cta];

    int idx{};
    WarpScan{temp_storage[bti]}.ExclusiveSum(count, idx);

    PRAGMA_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        if (used[i]) {
            smem.shared_exp_id[idx][bti] = idxs[i];
            smem.shared_scales[idx][bti] = data[i] * sum_prob;
            ++idx;
        }
    }

    PRAGMA_UNROLL
    for (int i = 0; i < max_tiles * max_expert_num; i += block_dim) {
        int e                   = (i + threadIdx.x) % max_expert_num;
        int t                   = (i + threadIdx.x) / max_expert_num;
        smem.shared_accum[t][e] = 0;
    }

    __syncthreads();

    constexpr int k_per_thread = cdiv(max_top_k, threads_per_token);

    const int bti2 = threadIdx.x % tokens_per_cta;
    const int ei2  = threadIdx.x / tokens_per_cta;
    const int ti2  = blockIdx.x * tokens_per_cta + bti2;

    PRAGMA_UNROLL
    for (int i = 0; i < k_per_thread; ++i) {
        const int   idx       = ei2 * k_per_thread + i;
        const int   expert_id = smem.shared_exp_id[idx][bti2];
        const float scale     = smem.shared_scales[idx][bti2];

        if (ti2 < token_num && idx < top_k) {
            masks[expert_id * token_num_padded + ti2] = idx;
            scales[idx * token_num + ti2]             = scale;
            atomicAdd(&smem.shared_accum[ti2 >> log_tile][expert_id], 1);

            // printf("%d %d %f\n", idx, expert_id, scale);
        }
    }

    __syncthreads();

    for (int i = 0; i < max_expert_num * max_tiles; i += block_dim) {
        int t = (threadIdx.x + i) % max_tiles;
        int e = (threadIdx.x + i) / max_tiles;
        if (e < expert_num && t < tiles) {
            atomicAdd(accum + e * tiles + t, smem.shared_accum[t][e]);
        }
    }
}

template<int N>
inline constexpr std::integral_constant<int, N> _Int{};

void invokeMoeGate_V2(int*         f2n,            // [e*n]  -> n
                      int*         en2f,           // [e,n] -> n*e
                      int*         offsets,        // [E+1]
                      float*       scales,         // [e,n]
                      void*        masks,          // [E,n]
                      int*         accum,          // [E]
                      const float* logits,         // [e,n]
                      int          tokens,         //  n
                      int          tokens_padded,  //  round_up(n, 4)
                      int          experts,        //  E
                      int          experts_per_token,
                      bool         norm_topk,
                      cudaStream_t st)
{
    constexpr int base_log_tile = 9;

    int log_tile = base_log_tile;
    while (((tokens_padded + (1 << log_tile) - 1) >> log_tile) > kMoeGateMaxTiles) {
        ++log_tile;
    }
    const int tiles = ceil_div(tokens_padded, 1 << log_tile);

    // std::cout << log_tile << " " << tiles << "\n";

    auto invoke = [&](auto max_expert_num, auto top_k, auto items_per_thread) {
        constexpr int thrs_per_tok = max_expert_num.value / items_per_thread.value;
        constexpr int threads      = 256;
        const int     blocks       = ceil_div(tokens, threads / thrs_per_tok);

        cudaMemsetAsync(masks, -1, sizeof(int8_t) * experts * tokens_padded, st);

        MoeGateKernel_v8<max_expert_num.value, top_k.value, items_per_thread.value, threads, 4>
            <<<blocks, threads, 0, st>>>(  //
                scales,
                (int8_t*)masks,
                accum,
                logits,
                log_tile,
                tiles,
                tokens,
                tokens_padded,
                experts,
                experts_per_token,
                norm_topk);
    };

    auto fail = [&] {
        std::cerr << "unsupported moe config: expert_num=" << experts << ", top_k=" << experts_per_token << "\n";
        std::abort();
    };

    if (experts <= 8) {
        if (experts_per_token <= 2) {
            invoke(_Int<8>, _Int<2>, _Int<8>);
        }
        else {
            invoke(_Int<8>, _Int<8>, _Int<8>);
        }
    }
    else if (experts <= 64) {
        if (experts_per_token <= 4) {
            invoke(_Int<64>, _Int<4>, _Int<16>);
        }
        else if (experts_per_token <= 8) {
            invoke(_Int<64>, _Int<8>, _Int<16>);
        }
        else {
            fail();
        }
    }
    else {
        fail();
    }

    {
        constexpr int threads = (1 << base_log_tile) / kMoeGateVecSize;
        const dim3    blocks(tiles, experts + 1);

        MoeScanKernel_v2<threads><<<blocks, threads, 0, st>>>(f2n,  //
                                                              en2f,
                                                              offsets,
                                                              (int8_t*)masks,
                                                              accum,
                                                              log_tile,
                                                              tiles,
                                                              tokens,
                                                              tokens_padded,
                                                              experts);
    }
}

template<int vec_size, int block_dim, class T>
__global__ void MoeGatherKernel(T*         dst,  // [e*n, d]
                                const T*   src,  // [  n, d]
                                const int* f2n,  // [e*n] :: e*n -> n
                                int        dims)
{
    using Vec        = Array<T, vec_size>;
    const int64_t bi = blockIdx.x;

    auto src_ptr = (const Vec*)src + dims * f2n[bi];
    auto dst_ptr = (/* */ Vec*)dst + dims * bi;
    for (int i = threadIdx.x; i < dims; i += block_dim) {
        Vec v;
        Ldg(v, src_ptr[i].data());
        Store(dst_ptr[i].data(), v);
    }
}

template<class T>
void invokeMoeGather(T* dst, const T* src, const int* f2n, int tokens, int experts_per_token, int dims, cudaStream_t st)
{
    constexpr int threads  = 256;
    constexpr int vec_size = 16 / sizeof(T);
    MoeGatherKernel<vec_size, threads><<<tokens * experts_per_token, threads, 0, st>>>(  //
        dst,
        src,
        f2n,
        dims / vec_size);
}

template void invokeMoeGather(uint16_t*, const uint16_t*, const int*, int, int, int, cudaStream_t);

template<int vec_size, int exp_k, int block_dim, class T>
__global__ void MoeReduceKernel(T*           dst,         // [  n, d]
                                const T*     src,         // [e*n, d]
                                const float* scales,      // [  e, n]
                                const int*   en2f,        // [  e, n] :: (e,n) -> e*n
                                const float* dst_scales,  // [n]
                                int          dims,
                                int          tokens)
{
    using Vec = Array<T, vec_size>;

    const int64_t ti = blockIdx.x;

    auto dst_ptr = (Vec*)dst + dims * ti;

    float dst_scale = 0;
    if (dst_scales) {
        dst_scale = dst_scales[ti];
        dst_scale = fdividef(1.f, 1.f + expf(-dst_scale));
    }

    // Should be warp uniforms
    const Vec* src_ptr[exp_k];
    float      scale[exp_k];
    PRAGMA_UNROLL
    for (int e = 0; e < exp_k; ++e) {
        src_ptr[e] = (const Vec*)src + dims * en2f[e * tokens + ti];
        scale[e]   = scales ? scales[e * tokens + ti] : 1.f;
    }

    for (int i = threadIdx.x; i < dims; i += block_dim) {
        Array<float, vec_size> accum{};
        if (dst_scales) {
            Vec v;
            Ldg(v, dst_ptr[i].data());
            using namespace ops;
            accum = cast<float>(v) * dst_scale;
        }
        PRAGMA_UNROLL
        for (int e = 0; e < exp_k; ++e) {
            Vec v;
            Ldg(v, src_ptr[e][i].data());
            using namespace ops;
            const auto x = cast<float>(v) * scale[e];
            accum        = accum + x;
        }
        Store(dst_ptr[i].data(), cast<T>(accum));
    }
}

template<class T>
void invokeMoeReduce(T*           dst,
                     const T*     src,
                     const float* scales,
                     const int*   en2f,
                     const float* dst_scales,
                     int          tokens,
                     int          experts_per_token,
                     int          dims,
                     cudaStream_t st)
{
    // std::cout << __PRETTY_FUNCTION__ << std::endl;

    const auto invoke = [&](auto e) {
        constexpr int threads     = 256;
        constexpr int vec_size    = 16 / sizeof(T);
        constexpr int exp_per_tok = decltype(e)::value;
        MoeReduceKernel<vec_size, exp_per_tok, threads><<<tokens, threads, 0, st>>>(  //
            dst,
            src,
            scales,
            en2f,
            dst_scales,
            dims / vec_size,
            tokens);
    };

    switch (experts_per_token) {
        case 1:
            return invoke(std::integral_constant<int, 1>{});
        case 2:
            return invoke(std::integral_constant<int, 2>{});
        case 4:
            return invoke(std::integral_constant<int, 4>{});
        case 6:
            return invoke(std::integral_constant<int, 6>{});
        case 8:
            return invoke(std::integral_constant<int, 8>{});
        default:
            fprintf(stderr, "Unsupported experts_per_token %d\n", experts_per_token);
            std::abort();
    }
}

template void invokeMoeReduce(half*, const half*, const float*, const int*, const float*, int, int, int, cudaStream_t);
#ifdef ENABLE_BF16
template void
invokeMoeReduce(nv_bfloat16*, const nv_bfloat16*, const float*, const int*, const float*, int, int, int, cudaStream_t);
#endif

std::vector<int> SampleUniform(int token_num, int expert_num, int exp_per_tok, std::mt19937& g)
{
    std::vector<int> idxs((size_t)token_num * exp_per_tok);
    std::vector<int> r(expert_num);
    std::iota(r.begin(), r.end(), 0);
    auto it = idxs.begin();
    for (int i = 0; i < token_num; ++i) {
        it = std::sample(r.cbegin(), r.cend(), it, exp_per_tok, g);
    }
    return idxs;
}

std::vector<int> SampleBalanced(int token_num, int expert_num, int exp_per_tok, std::mt19937& g)
{
    assert(exp_per_tok <= expert_num);
    std::vector<int> idxs((size_t)token_num * exp_per_tok);
    std::vector<int> q;

    std::vector<int> r(expert_num);
    std::iota(r.begin(), r.end(), 0);

    auto it = idxs.begin();
    for (int i = 0; i < token_num; ++i) {
        if ((int)q.size() < exp_per_tok) {
            const int k = q.size();
            // prepend the experts: [xxx] -> [yyy | xxx]
            q.insert(q.begin(), r.cbegin(), r.cend());
            // move duplicated experts to the front: [yyy | xxx] -> [xxx' | yyy' | xxx]
            int p = 0;
            std::for_each(q.cend() - k, q.cend(), [&](auto x) { std::swap(q[p++], q[x]); });
            // shuffle unique experts yyy'
            std::shuffle(q.begin() + p, q.end() - k, g);
        }
        it = std::copy(q.end() - exp_per_tok, q.end(), it);
        // remove used experts [xxx' | yyy' | xxx ] -> [xxx' | zzz]
        q.resize(q.size() - exp_per_tok);
        // alias [xxx] <- [xxx' | zzz]
    }
    assert(it == idxs.end());

    // shuffle to decorrelate adjacent tokens
    r.resize(token_num);
    std::iota(r.begin(), r.end(), 0);
    std::shuffle(r.begin(), r.end(), g);
    std::vector<int> ret(idxs.size());
    it = ret.begin();
    for (const auto& i : r) {
        it = std::copy_n(idxs.begin() + i * exp_per_tok, exp_per_tok, it);
    }
    assert(it == ret.end());
    return ret;
}

}  // namespace turbomind
