// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_pipeline_primitives.h>

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"

namespace turbomind {

template<int top_k, int block_dim>
__global__ void MoeGateKernel_V2(float*       scales,  // [e,n]
                                 int*         masks,   // [E,n], padded
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

template<int block_dim>
__global__ void MoeScanKernel_V2(int*       f2n,      // [e*n]
                                 int*       en2f,     // [e,n]
                                 int*       offsets,  // [E+1]
                                 int*       masks,    // [E,n], padded
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

    using Vec = Array<int, vec_size>;

    const int tile_id = blockIdx.x;
    const int ei      = blockIdx.y;

    const int global_tile_id = ei * tiles + tile_id;

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
        fill(data, -1);
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

void invokeMoeGate_V2(int*         f2n,            // [e*n]  -> n
                      int*         en2f,           // [e,n] -> n*e
                      int*         offsets,        // [E+1]
                      float*       scales,         // [e,n]
                      int*         masks,          // [E,n]
                      int*         accum,          // [E]
                      const float* logits,         // [e,n]
                      int          tokens,         //  n
                      int          tokens_padded,  //  round_up(n, 4)
                      int          experts,        //  E
                      int          experts_per_token,
                      cudaStream_t st)
{
    constexpr int base_log_tile = 9;

    int log_tile = base_log_tile;
    while (((tokens_padded + (1 << log_tile) - 1) >> log_tile) > kMoeGateMaxTiles) {
        ++log_tile;
    }
    const int tiles = ceil_div(tokens_padded, 1 << log_tile);

    // std::cout << log_tile << " " << tiles << "\n";

    {
        constexpr int threads = 128;
        const int     blocks  = ceil_div(tokens, threads);

        auto invoke = [&](auto e) {
            static constexpr int top_k = decltype(e)::value;
            MoeGateKernel_V2<top_k, threads><<<blocks, threads, 0, st>>>(  //
                scales,
                masks,
                accum,
                logits,
                log_tile,
                tiles,
                tokens,
                tokens_padded,
                experts);
        };

        switch (experts_per_token) {
            case 2:
                invoke(std::integral_constant<int, 2>{});
                break;
            // case 4:
            //     invoke(std::integral_constant<int, 4>{});
            //     break;
            default:
                std::cerr << __FILE__ << ":" << __LINE__ << " Not implemented. " << std::endl;
                std::abort();
        }
    }

    // return;

    {
        // Check: tiles * experts <= threads

        constexpr int threads = (1 << base_log_tile) / kMoeGateVecSize;
        const dim3    blocks(tiles, experts + 1);
        MoeScanKernel_V2<threads><<<blocks, threads, 0, st>>>(f2n,  //
                                                              en2f,
                                                              offsets,
                                                              masks,
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
__global__ void MoeReduceKernel(T*           dst,     // [  n, d]
                                const T*     src,     // [e*n, d]
                                const float* scales,  // [  e, n]
                                const int*   en2f,    // [  e, n] :: (e,n) -> e*n
                                int          dims,
                                int          tokens)
{
    using Vec = Array<T, vec_size>;

    const int64_t ti = blockIdx.x;

    auto dst_ptr = (Vec*)dst + dims * ti;

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
            dims / vec_size,
            tokens);
    };

    switch (experts_per_token) {
        case 1:
            return invoke(std::integral_constant<int, 1>{});
        case 2:
            return invoke(std::integral_constant<int, 2>{});
        // case 4:
        //     return invoke(std::integral_constant<int, 4>{});
        // case 6:
        //     return invoke(std::integral_constant<int, 6>{});
        default:
            fprintf(stderr, "Unsupported experts_per_token %d\n", experts_per_token);
            std::abort();
    }
}

template void invokeMoeReduce(half*, const half*, const float*, const int*, int, int, int, cudaStream_t);
#ifdef ENABLE_BF16
template void invokeMoeReduce(nv_bfloat16*, const nv_bfloat16*, const float*, const int*, int, int, int, cudaStream_t);
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