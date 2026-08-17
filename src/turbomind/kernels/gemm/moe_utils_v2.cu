// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_scan.cuh>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

// Classic scan: each (tile,expert) CTA reduces accum[0 .. ei*tiles+tile_id).
// Cheap when E*tiles is small; quadratic cost dominates at E=2560.
template<int block_dim, class Mask>
__global__ void MoeScanKernel_v2_classic(int*       f2n,
                                         int*       f2E,
                                         int*       en2f,
                                         int*       offsets,
                                         Mask*      masks,
                                         const int* accum,
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
    using Vec              = Array<Mask, vec_size>;

    const int tile_id = blockIdx.x;
    const int ei      = blockIdx.y;

    const int  global_tile_id = ei * tiles + tile_id;
    const bool is_valid       = global_tile_id <= experts * tiles;

    int vacc = 0;
    for (int i = threadIdx.x; i < global_tile_id; i += block_dim) {
        if (is_valid && i < global_tile_id) {
            vacc += accum[i];
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

    const int token_vecs      = tokens_padded / vec_size;
    const int tile_size       = 1 << log_tile;
    const int tile_vec_size   = tile_size / vec_size;
    const int tile_vec_beg    = tile_id * tile_vec_size;
    const int tile_vec_end    = std::min(tile_vec_beg + tile_vec_size, token_vecs);
    const int tile_vec_padded = tile_vec_beg + round_up(tile_vec_size, block_dim);

    auto mask_ptr = (Vec*)masks + ei * token_vecs;
    for (int vi = tile_vec_beg + threadIdx.x; vi < tile_vec_padded; vi += block_dim) {
        const bool pred = vi < tile_vec_end;
        Vec        data;
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
                const int flat_id           = prefix[i] + offset;
                const int ti                = vi * vec_size + i;
                f2n[flat_id]                = ti;
                f2E[flat_id]                = ei;
                en2f[data[i] * tokens + ti] = flat_id;
            }
        }
        offset += block_sum;
    }
}

// Hierarchical prefix phase 1: multi-CTA over expert slabs.
// Each CTA owns `block_dim` experts (1 expert / thread). In-place exclusive
// scan of accum[e, 0..tiles) and write expert totals into offsets[e].
template<int block_dim>
__global__ void MoeScanPrefixPhase1Kernel(int* accum,    // [E,tiles] counts -> excl tile prefixes
                                          int* offsets,  // [E+1]; [0..E) = expert totals after phase1
                                          int  tiles,
                                          int  experts)
{
    const int ei = static_cast<int>(blockIdx.x) * block_dim + static_cast<int>(threadIdx.x);
    if (ei >= experts) {
        return;
    }

    int* row = accum + ei * tiles;
    int  sum = 0;
    // tiles <= kMoeGateMaxTiles (16).
    for (int t = 0; t < tiles; ++t) {
        const int v = row[t];
        row[t]      = sum;
        sum += v;
    }
    offsets[ei] = sum;
}

// Hierarchical prefix phase 2: one CTA exclusive-scans expert totals.
template<int block_dim>
__global__ void MoeScanPrefixPhase2Kernel(int* offsets,  // [E+1]; [0..E) totals in, excl out
                                          int  experts)
{
    static_assert(block_dim % 32 == 0, "BLOCK_SCAN_WARP_SCANS requires warp-multiple block_dim");
    using BlockScan = cub::BlockScan<int, block_dim, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ int                             shared_running;

    if (threadIdx.x == 0) {
        shared_running = 0;
    }
    __syncthreads();

    for (int base = 0; base < experts; base += block_dim) {
        const int i = base + threadIdx.x;
        int       v = (i < experts) ? offsets[i] : 0;
        int       prefix{};
        int       block_sum{};
        BlockScan{temp_storage}.ExclusiveSum(v, prefix, block_sum);
        __syncthreads();
        if (i < experts) {
            offsets[i] = prefix + shared_running;
        }
        if (threadIdx.x == 0) {
            shared_running += block_sum;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        offsets[experts] = shared_running;
    }
}

// Large-E scan: O(1) offset from precomputed expert/tile exclusive prefixes.
template<int block_dim, class Mask>
__global__ void MoeScanKernel_v2(int*       f2n,      // [e*n]
                                 int*       f2E,      // [e*n]
                                 int*       en2f,     // [e,n]
                                 int*       offsets,  // [E+1] exclusive expert offsets
                                 Mask*      masks,    // [E,n], padded
                                 const int* accum,    // [E,tiles] exclusive tile prefixes
                                 int        log_tile,
                                 int        tiles,
                                 int        tokens,
                                 int        tokens_padded,
                                 int        experts)
{
    static_assert(block_dim % 32 == 0, "BLOCK_SCAN_WARP_SCANS requires warp-multiple block_dim");
    using BlockScan = cub::BlockScan<int, block_dim, cub::BLOCK_SCAN_WARP_SCANS>;

    __shared__ typename BlockScan::TempStorage temp_storage;

    constexpr int vec_size = kMoeGateVecSize;

    using Vec = Array<Mask, vec_size>;

    const int tile_id = blockIdx.x;
    const int ei      = blockIdx.y;

    // offsets[ei] = sum of all prior experts; accum[ei*tiles+tile_id] = excl prefix within expert.
    int offset = offsets[ei] + accum[ei * tiles + tile_id];

    const int token_vecs = tokens_padded / vec_size;

    const int tile_size     = 1 << log_tile;
    const int tile_vec_size = tile_size / vec_size;

    const int tile_vec_beg    = tile_id * tile_vec_size;
    const int tile_vec_end    = std::min(tile_vec_beg + tile_vec_size, token_vecs);
    const int tile_vec_padded = tile_vec_beg + round_up(tile_vec_size, block_dim);

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

        BlockScan{temp_storage}.ExclusiveSum(prefix, prefix, block_sum);
        __syncthreads();

        PRAGMA_UNROLL
        for (int i = 0; i < vec_size; ++i) {
            if (pred && data[i] >= 0) {
                const int flat_id = prefix[i] + offset;
                const int ti      = vi * vec_size + i;
                f2n[flat_id]      = ti;
                f2E[flat_id]      = ei;
                // No ti is generated for padded tokens so we are safe
                en2f[data[i] * tokens + ti] = flat_id;
            }
        }

        offset += block_sum;
    }
}

static void launchMoeScan_V2(int*         f2n,
                             int*         f2E,
                             int*         en2f,
                             int*         offsets,
                             int8_t*      masks,
                             int*         accum,
                             int          log_tile,
                             int          tiles,
                             int          tokens,
                             int          tokens_padded,
                             int          experts,
                             int          base_log_tile,
                             cudaStream_t st)
{
    constexpr int threads = (1 << 9) / kMoeGateVecSize;
    TM_CHECK_EQ(base_log_tile, 9);

    // Size split: classic reduce when E*tiles is modest (one launch; hierarchical
    // prefix overhead can lose at small tile counts). Hierarchical when the
    // classic O((E*tiles)^2) reduce dominates (large-E and enough tiles).
    const bool use_hierarchical = (experts > 512) && (tiles >= 4);
    if (!use_hierarchical) {
        const dim3 blocks(tiles, experts + 1);
        MoeScanKernel_v2_classic<threads><<<blocks, threads, 0, st>>>(
            f2n, f2E, en2f, offsets, masks, accum, log_tile, tiles, tokens, tokens_padded, experts);
        return;
    }

    // Multi-CTA prefix: phase1 parallelizes expert-row excl scans; phase2 scans totals.
    constexpr int prefix_threads = 256;
    const int     phase1_blocks  = (experts + prefix_threads - 1) / prefix_threads;
    MoeScanPrefixPhase1Kernel<prefix_threads><<<phase1_blocks, prefix_threads, 0, st>>>(accum, offsets, tiles, experts);
    MoeScanPrefixPhase2Kernel<prefix_threads><<<1, prefix_threads, 0, st>>>(offsets, experts);

    const dim3 blocks(tiles, experts);
    MoeScanKernel_v2<threads><<<blocks, threads, 0, st>>>(
        f2n, f2E, en2f, offsets, masks, accum, log_tile, tiles, tokens, tokens_padded, experts);
}

template<int NBits>
struct MoeGateBitset {
    static constexpr int kWords = (NBits + 31) / 32;
    uint32_t             words[kWords];

    __device__ void fill_ones()
    {
        PRAGMA_UNROLL
        for (int w = 0; w < kWords; ++w) {
            words[w] = 0xffffffffu;
        }
        // Clear unused high bits in the last word so invert/test stay well-defined.
        constexpr int rem = NBits % 32;
        if constexpr (rem != 0) {
            words[kWords - 1] = (1u << rem) - 1u;
        }
    }

    __device__ bool test(int i) const
    {
        return (words[i >> 5] >> (i & 31)) & 1u;
    }

    __device__ void clear(int i)
    {
        words[i >> 5] &= ~(1u << (i & 31));
    }

    __device__ void invert_active()
    {
        // After top-k clears, remaining 1-bits are unselected; invert to selected.
        PRAGMA_UNROLL
        for (int w = 0; w < kWords; ++w) {
            words[w] = ~words[w];
        }
        constexpr int rem = NBits % 32;
        if constexpr (rem != 0) {
            words[kWords - 1] &= (1u << rem) - 1u;
        }
    }
};

template<int max_expert_num, int max_top_k, int tokens_per_cta>
struct MoeGateV8SharedStorage {
    // A CTA's tokens_per_cta consecutive tokens span at most 2 log_tile buckets
    // (tile_size >= 512, tokens_per_cta <= 32). Dense [kMoeGateMaxTiles][E] was
    // ~164KB at E=2560 (1 CTA/SM); 2 rows is ~20KB and restores smem-batched
    // histogram flush (avoids global-atomic contention on small-E).
    static constexpr int local_tiles = 2;
    int                  shared_accum[local_tiles][max_expert_num + 1];
    float                shared_scales[max_top_k][tokens_per_cta];
    int                  shared_exp_id[max_top_k][tokens_per_cta];
};

template<int  max_expert_num,
         int  max_top_k,
         int  items_per_thread,
         int  block_dim,
         int  access_size,
         bool fuse_mask_clear,
         bool is_routing,
         class Mask>
__global__ void MoeGateKernel_v8(float*       scales,      // routing: [e,n], direct: [n,e]
                                 Mask*        masks,       // routing: [E,n] int8, padded, direct: [n,e] int32
                                 int*         accum,       // routing: [E,tiles], direct: nullptr
                                 const float* logits,      // [n,E]
                                 const bool*  token_mask,  // [n]; invalid tokens route nowhere
                                 int          log_tile,
                                 int          tiles,
                                 int          token_num,
                                 int          token_num_padded,
                                 int          expert_num,
                                 int          top_k,
                                 bool         softmax,
                                 bool         norm_topk,
                                 float        routed_scale)
{
    constexpr int threads_per_token = max_expert_num / items_per_thread;  // 8
    constexpr int tokens_per_cta    = block_dim / threads_per_token;

    using SharedStorage = MoeGateV8SharedStorage<max_expert_num, max_top_k, tokens_per_cta>;

    __shared__ SharedStorage smem;

    // We use warp-level primitives for reduction
    static_assert(threads_per_token <= 32);
    static_assert((threads_per_token & (threads_per_token - 1)) == 0);
    static_assert(items_per_thread % access_size == 0);
    static_assert(max_expert_num == threads_per_token * items_per_thread);

    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int ti = thread_idx / threads_per_token;
    const int ei = thread_idx % threads_per_token;

    const int bti = threadIdx.x / threads_per_token;

    const int warp_ti = threadIdx.x % WARP_SIZE / threads_per_token;

    float data[items_per_thread];

    PRAGMA_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        data[i] = -std::numeric_limits<float>::infinity();
    }
    if (ti < token_num) {
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; i += access_size) {
            const int e = ei * items_per_thread + i;
            if (e < expert_num) {
                Ldg((Array<float, access_size>&)data[i], &logits[ti * expert_num + e]);
            }
        }
    }

    MoeGateBitset<items_per_thread> mask;
    mask.fill_ones();
    float max_logit{};
    int   count{};

    const int warp_ti_offset = warp_ti * threads_per_token;

    auto run = [&](int k) {
        int   max_i   = -1;
        float max_val = -std::numeric_limits<float>::infinity();
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            if (mask.test(i) && data[i] > max_val) {
                max_i   = i;
                max_val = data[i];
            }
        }

        int   g_max_ei  = ei;
        float g_max_val = max_val;
        if constexpr (threads_per_token > 1) {
            PRAGMA_UNROLL
            for (int m = threads_per_token / 2; m >= 1; m /= 2) {
                g_max_val = fmaxf(g_max_val, __shfl_xor_sync((uint32_t)-1, g_max_val, m));
            }
            const auto active = __ballot_sync((uint32_t)-1, max_val == g_max_val);
            g_max_ei          = __ffs(active >> (unsigned)warp_ti_offset) - 1;
        }
        if (k == 0) {
            max_logit = g_max_val;
        }
        if (ei == g_max_ei) {
            if (max_i >= 0) {
                mask.clear(max_i);
            }
            ++count;
        }
    };

    run(0);
    for (int k = 1; k < top_k; ++k) {
        run(k);
    }

    mask.invert_active();

    float sum_prob{};

    if (softmax) {
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            if (!norm_topk || mask.test(i)) {
                data[i] = expf(data[i] - max_logit);
                sum_prob += data[i];
            }
        }
        PRAGMA_UNROLL
        for (int m = threads_per_token / 2; m >= 1; m /= 2) {
            sum_prob += __shfl_xor_sync((uint32_t)-1, sum_prob, m);
        }
        sum_prob = fdividef(1.f, sum_prob);
    }
    else {
        sum_prob = 1.f;
    }

    using WarpScan = cub::WarpScan<int, threads_per_token>;
    __shared__ typename WarpScan::TempStorage temp_storage[tokens_per_cta];

    int idx{};
    WarpScan{temp_storage[bti]}.ExclusiveSum(count, idx);

    if constexpr (!is_routing) {
        const int end = idx + count;
        if (ti < token_num) {
            if (token_mask[ti]) {
                PRAGMA_UNROLL
                for (int i = 0; i < items_per_thread; ++i) {
                    if (mask.test(i)) {
                        const int dst = ti * top_k + idx;
                        masks[dst]    = ei * items_per_thread + i;
                        scales[dst]   = data[i] * sum_prob * routed_scale;
                        ++idx;
                    }
                }
            }
            // Preserve MoeA2AGate's handling of unselectable logits and overwrite
            // every output slot for masked tokens with skip sentinels.
            for (; idx < end; ++idx) {
                const int dst = ti * top_k + idx;
                masks[dst]    = -1;
                scales[dst]   = 0.f;
            }
        }
        return;
    }
    PRAGMA_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        if (mask.test(i)) {
            smem.shared_exp_id[idx][bti] = ei * items_per_thread + i;
            smem.shared_scales[idx][bti] = data[i] * sum_prob;
            ++idx;
        }
    }

    constexpr int local_tiles = SharedStorage::local_tiles;
    const int     ti_base     = blockIdx.x * tokens_per_cta;
    const int     tile0       = ti_base >> log_tile;

    // Optional in-kernel mask clear (host skips memset when fuse_mask_clear).
    // Cheap for mid/small E; for E=2560 the E*tok/CTA stores regress large-T.
    if constexpr (fuse_mask_clear) {
        for (int i = threadIdx.x; i < expert_num * tokens_per_cta; i += block_dim) {
            const int e  = i / tokens_per_cta;
            const int t  = i % tokens_per_cta;
            const int ti = ti_base + t;
            if (ti < token_num_padded) {
                masks[e * token_num_padded + ti] = Mask{-1};
            }
        }
    }

    PRAGMA_UNROLL
    for (int i = 0; i < local_tiles * max_expert_num; i += block_dim) {
        const int e = (i + threadIdx.x) % max_expert_num;
        const int t = (i + threadIdx.x) / max_expert_num;
        if (t < local_tiles) {
            smem.shared_accum[t][e] = 0;
        }
    }

    __syncthreads();

    constexpr int k_per_thread = cdiv(max_top_k, threads_per_token);

    const int bti2 = threadIdx.x % tokens_per_cta;
    const int ei2  = threadIdx.x / tokens_per_cta;
    const int ti2  = ti_base + bti2;

    // Invalid tokens write no masks/scales/accum entries, so the scan compacts
    // nothing for them and the expert group GEMM never sees them. This also
    // keeps garbage logits (e.g. NaN in stale rows) from indexing `masks` with
    // an uninitialized `shared_exp_id` entry.
    const bool token_valid = ti2 < token_num && token_mask[ti2];

    PRAGMA_UNROLL
    for (int i = 0; i < k_per_thread; ++i) {
        const int idx = ei2 * k_per_thread + i;
        if (token_valid && idx < top_k) {
            const int   expert_id = smem.shared_exp_id[idx][bti2];
            const float scale     = smem.shared_scales[idx][bti2];

            masks[expert_id * token_num_padded + ti2] = idx;
            scales[idx * token_num + ti2]             = scale * routed_scale;
            atomicAdd(&smem.shared_accum[(ti2 >> log_tile) - tile0][expert_id], 1);
        }
    }

    __syncthreads();

    const int tile1         = (ti_base + tokens_per_cta - 1) >> log_tile;
    const int n_local_tiles = tile1 - tile0 + 1;

    for (int i = 0; i < max_expert_num * local_tiles; i += block_dim) {
        const int t = (threadIdx.x + i) % local_tiles;
        const int e = (threadIdx.x + i) / local_tiles;
        if (e < expert_num && t < n_local_tiles) {
            const int v = smem.shared_accum[t][e];
            if (v) {
                atomicAdd(accum + e * tiles + (tile0 + t), v);
            }
        }
    }
}

template<int N>
inline constexpr std::integral_constant<int, N> _Int{};

void launchMoeGate_V8(float*       scales,
                      void*        masks,
                      bool         is_routing,
                      int*         accum,
                      const float* logits,
                      const bool*  token_mask,
                      int          log_tile,
                      int          tiles,
                      int          tokens,
                      int          tokens_padded,
                      int          experts,
                      int          experts_per_token,
                      bool         softmax,
                      bool         norm_topk,
                      float        routed_scale,
                      cudaStream_t st)
{
    TM_CHECK(token_mask);

    auto invoke = [&](auto max_expert_num,  //
                      auto top_k,
                      auto items_per_thread,
                      auto vec_size,
                      auto fuse_mask_clear,
                      auto routing) {
        constexpr bool kIsRouting = decltype(routing)::value;
        using Mask                = std::conditional_t<kIsRouting, int8_t, int>;

        constexpr int thrs_per_tok   = max_expert_num.value / items_per_thread.value;
        constexpr int threads        = 256;
        constexpr int tokens_per_cta = threads / thrs_per_tok;
        // Fuse mask clear for E<=512 (launch tax). Keep host memset for E=2560
        // where E*tok/CTA stores regress large-token cases.
        constexpr bool kFuseMaskClear = fuse_mask_clear.value != 0;
        const int      blocks         = ceil_div(kFuseMaskClear ? tokens_padded : tokens, tokens_per_cta);

        auto* kernel = MoeGateKernel_v8<max_expert_num.value,
                                        top_k.value,
                                        items_per_thread.value,
                                        threads,
                                        vec_size.value,
                                        kFuseMaskClear,
                                        kIsRouting,
                                        Mask>;

        if constexpr (kIsRouting && !kFuseMaskClear) {
            cudaMemsetAsync(masks, -1, sizeof(Mask) * experts * tokens_padded, st);
        }

        kernel<<<blocks, threads, 0, st>>>(scales,
                                           static_cast<Mask*>(masks),
                                           accum,
                                           logits,
                                           token_mask,
                                           log_tile,
                                           tiles,
                                           tokens,
                                           tokens_padded,
                                           experts,
                                           experts_per_token,
                                           softmax,
                                           norm_topk,
                                           routed_scale);
        return true;
    };

    auto dispatch = [&](auto routing) {
        // fuse_mask_clear=_Int<1> for E<=512; _Int<0> keeps host memset for E=2560.
        if (experts <= 8) {
            if (experts_per_token <= 2) {
                return invoke(_Int<8>, _Int<2>, _Int<8>, _Int<4>, _Int<1>, routing);
            }
            else {
                return invoke(_Int<8>, _Int<8>, _Int<8>, _Int<4>, _Int<1>, routing);
            }
        }
        else if (experts <= 64) {
            if (experts_per_token <= 4) {
                return invoke(_Int<64>, _Int<4>, _Int<16>, _Int<4>, _Int<1>, routing);
            }
            else if (experts_per_token <= 8) {
                return invoke(_Int<64>, _Int<8>, _Int<16>, _Int<4>, _Int<1>, routing);
            }
        }
        else if (experts <= 128) {
            if (experts_per_token <= 8) {
                return invoke(_Int<128>, _Int<8>, _Int<16>, _Int<4>, _Int<1>, routing);
            }
        }
        else if (experts <= 160) {
            if (experts_per_token <= 8) {
                return invoke(_Int<160>, _Int<8>, _Int<10>, _Int<2>, _Int<1>, routing);
            }
        }
        else if (experts <= 256) {
            if (experts_per_token <= 8) {
                return invoke(_Int<256>, _Int<8>, _Int<16>, _Int<4>, _Int<1>, routing);
            }
        }
        else if (experts <= 512) {
            if (experts_per_token <= 10) {
                return invoke(_Int<512>, _Int<10>, _Int<16>, _Int<4>, _Int<1>, routing);
            }
        }
        else if (experts <= 2560) {
            if (experts_per_token <= 8) {
                // ~20KB smem (2-row hist); static smem fits H200 default carveout.
                return invoke(_Int<2560>, _Int<8>, _Int<80>, _Int<4>, _Int<0>, routing);
            }
        }
        return false;
    };

    auto dispatch_mode = [&] {
        if (is_routing) {
            return dispatch(std::true_type{});
        }
        else {
            return dispatch(std::false_type{});
        }
    };

    if (!softmax && norm_topk) {
        // norm top-k is part of softmax impl
        TM_LOG_FATAL("unsupported moe config: softmax={} norm_topk={}", softmax, norm_topk);
    }

    const bool success = dispatch_mode();
    TM_CHECK(success) << "unsupported moe config: expert_num=" << experts << ", top_k=" << experts_per_token
                      << ", softmax=" << softmax << ", norm_topk=" << norm_topk;

    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeMoeGate_V2(float*       topk_weights,
                      int*         topk_indices,
                      const float* logits,
                      const bool*  token_mask,
                      int          tokens,
                      int          experts,
                      int          experts_per_token,
                      bool         softmax,
                      bool         norm_topk,
                      float        routed_scale,
                      cudaStream_t st)
{
    TM_CHECK_GE(tokens, 0);
    if (tokens == 0) {
        return;
    }

    launchMoeGate_V8(topk_weights,
                     topk_indices,
                     false,
                     nullptr,
                     logits,
                     token_mask,
                     0,
                     0,
                     tokens,
                     tokens,
                     experts,
                     experts_per_token,
                     softmax,
                     norm_topk,
                     routed_scale,
                     st);
}

void invokeMoeGate_V2(int*         f2n,            // [e*n] -> n
                      int*         f2E,            // [e*n] -> local E
                      int*         en2f,           // [e,n] -> n*e
                      int*         offsets,        // [local E+1]
                      float*       scales,         // [e,n]
                      void*        masks,          // [E,n]
                      int*         accum,          // [E,tiles]
                      const float* logits,         // [n,E]
                      const bool*  token_mask,     // [n]; invalid tokens route nowhere
                      int          tokens,         //  n
                      int          tokens_padded,  //  round_up(n, 4)
                      int          experts,        //  E
                      int          experts_per_token,
                      int          local_expert_offset,
                      int          local_expert_num,
                      bool         softmax,
                      bool         norm_topk,
                      float        routed_scale,
                      cudaStream_t st)
{
    TM_CHECK(token_mask);

    constexpr int base_log_tile = 9;

    int log_tile = base_log_tile;
    while (((tokens_padded + (1 << log_tile) - 1) >> log_tile) > kMoeGateMaxTiles) {
        ++log_tile;
    }
    const int tiles = ceil_div(tokens_padded, 1 << log_tile);

    launchMoeGate_V8(scales,
                     masks,
                     true,
                     accum,
                     logits,
                     token_mask,
                     log_tile,
                     tiles,
                     tokens,
                     tokens_padded,
                     experts,
                     experts_per_token,
                     softmax,
                     norm_topk,
                     routed_scale,
                     st);

    launchMoeScan_V2(f2n,
                     f2E,
                     en2f,
                     offsets,
                     (int8_t*)masks + local_expert_offset * tokens_padded,
                     accum + local_expert_offset * tiles,
                     log_tile,
                     tiles,
                     tokens,
                     tokens_padded,
                     local_expert_num,
                     base_log_tile,
                     st);
    TM_CUDA_CHECK(cudaGetLastError());
}

// noaux_tc: scores = scoring_func(logits), scores_for_choice = scores + correction_bias,
// top-k on scores_for_choice, weights from scores; optionally renormalize; apply routed_scale.
template<int  max_expert_num,
         int  max_top_k,
         int  items_per_thread,
         int  block_dim,
         int  access_size,
         bool fuse_mask_clear,
         bool is_routing,
         class Mask>
__global__ void MoeGateNoAuxTCKernel_V2(float*       scales,
                                        Mask*        masks,
                                        int*         accum,
                                        const float* logits,
                                        const bool*  token_mask,
                                        const float* correction_bias,
                                        int          log_tile,
                                        int          tiles,
                                        int          token_num,
                                        int          token_num_padded,
                                        int          expert_num,
                                        int          top_k,
                                        bool         use_sigmoid,
                                        bool         norm_topk,
                                        float        routed_scale)
{
    constexpr int threads_per_token = max_expert_num / items_per_thread;
    constexpr int tokens_per_cta    = block_dim / threads_per_token;

    using SharedStorage = MoeGateV8SharedStorage<max_expert_num, max_top_k, tokens_per_cta>;
    __shared__ SharedStorage smem;

    static_assert(threads_per_token <= 32);
    static_assert((threads_per_token & (threads_per_token - 1)) == 0);
    static_assert(items_per_thread % access_size == 0);
    static_assert(max_expert_num == threads_per_token * items_per_thread);

    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int ti = thread_idx / threads_per_token;
    const int ei = thread_idx % threads_per_token;

    const int bti = threadIdx.x / threads_per_token;

    const int warp_ti = threadIdx.x % WARP_SIZE / threads_per_token;

    float data[items_per_thread];
    float data_s[items_per_thread];

    PRAGMA_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        data[i]   = -std::numeric_limits<float>::infinity();
        data_s[i] = -std::numeric_limits<float>::infinity();
    }

    if (ti < token_num) {
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; i += access_size) {
            const int e = ei * items_per_thread + i;
            if (e < expert_num) {
                Ldg((Array<float, access_size>&)data[i], &logits[ti * expert_num + e]);
                if (correction_bias) {
                    Ldg((Array<float, access_size>&)data_s[i], correction_bias + e);
                }
            }
        }
    }

    if (use_sigmoid) {
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            const float s = fdividef(1.f, 1.f + expf(-data[i]));
            data[i]       = s;
            data_s[i]     = correction_bias ? data_s[i] + s : s;
        }
    }
    else {
        float max_logit = -std::numeric_limits<float>::infinity();

        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            max_logit = fmaxf(max_logit, data[i]);
        }
        PRAGMA_UNROLL
        for (int m = threads_per_token / 2; m >= 1; m /= 2) {
            max_logit = fmaxf(max_logit, __shfl_xor_sync((uint32_t)-1, max_logit, m));
        }

        float sum_prob = 0.f;
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            data[i] = expf(data[i] - max_logit);
            sum_prob += data[i];
        }
        PRAGMA_UNROLL
        for (int m = threads_per_token / 2; m >= 1; m /= 2) {
            sum_prob += __shfl_xor_sync((uint32_t)-1, sum_prob, m);
        }
        const float inv_sum = fdividef(1.f, sum_prob);

        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            const float s = data[i] * inv_sum;
            data[i]       = s;
            data_s[i]     = correction_bias ? data_s[i] + s : s;
        }
    }
    MoeGateBitset<items_per_thread> mask;
    mask.fill_ones();

    int count{};

    const int warp_ti_offset = warp_ti * threads_per_token;

    auto run = [&] {
        int   max_i   = -1;
        float max_val = -std::numeric_limits<float>::infinity();
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            if (mask.test(i) && data_s[i] > max_val) {
                max_i   = i;
                max_val = data_s[i];
            }
        }

        int   g_max_ei  = ei;
        float g_max_val = max_val;
        if constexpr (threads_per_token > 1) {
            PRAGMA_UNROLL
            for (int m = threads_per_token / 2; m >= 1; m /= 2) {
                g_max_val = fmaxf(g_max_val, __shfl_xor_sync((uint32_t)-1, g_max_val, m));
            }
            const auto active = __ballot_sync((uint32_t)-1, max_val == g_max_val);
            g_max_ei          = __ffs(active >> (unsigned)warp_ti_offset) - 1;
        }
        if (ei == g_max_ei) {
            if (max_i >= 0) {
                mask.clear(max_i);
            }
            ++count;
        }
    };

    run();
    for (int k = 1; k < top_k; ++k) {
        run();
    }

    mask.invert_active();

    float sum_prob{};
    if (norm_topk) {
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            if (mask.test(i)) {
                sum_prob += data[i];
            }
        }
        PRAGMA_UNROLL
        for (int m = threads_per_token / 2; m >= 1; m /= 2) {
            sum_prob += __shfl_xor_sync((uint32_t)-1, sum_prob, m);
        }
        sum_prob = fdividef(1.f, sum_prob);
    }
    else {
        sum_prob = 1.f;
    }

    using WarpScan = cub::WarpScan<int, threads_per_token>;
    __shared__ typename WarpScan::TempStorage temp_storage[tokens_per_cta];

    int idx{};
    WarpScan{temp_storage[bti]}.ExclusiveSum(count, idx);

    if constexpr (!is_routing) {
        const int end = idx + count;
        if (ti < token_num) {
            if (token_mask[ti]) {
                PRAGMA_UNROLL
                for (int i = 0; i < items_per_thread; ++i) {
                    if (mask.test(i)) {
                        const int dst = ti * top_k + idx;
                        masks[dst]    = ei * items_per_thread + i;
                        scales[dst]   = data[i] * sum_prob * routed_scale;
                        ++idx;
                    }
                }
            }
            for (; idx < end; ++idx) {
                const int dst = ti * top_k + idx;
                masks[dst]    = -1;
                scales[dst]   = 0.f;
            }
        }
        return;
    }

    PRAGMA_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        if (mask.test(i)) {
            smem.shared_exp_id[idx][bti] = ei * items_per_thread + i;
            smem.shared_scales[idx][bti] = data[i] * sum_prob;
            ++idx;
        }
    }

    constexpr int local_tiles = SharedStorage::local_tiles;
    const int     ti_base     = blockIdx.x * tokens_per_cta;
    const int     tile0       = ti_base >> log_tile;

    if constexpr (fuse_mask_clear) {
        for (int i = threadIdx.x; i < expert_num * tokens_per_cta; i += block_dim) {
            const int e  = i / tokens_per_cta;
            const int t  = i % tokens_per_cta;
            const int ti = ti_base + t;
            if (ti < token_num_padded) {
                masks[e * token_num_padded + ti] = Mask{-1};
            }
        }
    }

    PRAGMA_UNROLL
    for (int i = 0; i < local_tiles * max_expert_num; i += block_dim) {
        const int e = (i + threadIdx.x) % max_expert_num;
        const int t = (i + threadIdx.x) / max_expert_num;
        if (t < local_tiles) {
            smem.shared_accum[t][e] = 0;
        }
    }

    __syncthreads();

    constexpr int k_per_thread = cdiv(max_top_k, threads_per_token);

    const int bti2 = threadIdx.x % tokens_per_cta;
    const int ei2  = threadIdx.x / tokens_per_cta;
    const int ti2  = ti_base + bti2;

    const bool token_valid = ti2 < token_num && token_mask[ti2];

    PRAGMA_UNROLL
    for (int i = 0; i < k_per_thread; ++i) {
        const int idx = ei2 * k_per_thread + i;
        if (token_valid && idx < top_k) {
            const int   expert_id = smem.shared_exp_id[idx][bti2];
            const float scale     = smem.shared_scales[idx][bti2];

            masks[expert_id * token_num_padded + ti2] = idx;
            scales[idx * token_num + ti2]             = scale * routed_scale;
            atomicAdd(&smem.shared_accum[(ti2 >> log_tile) - tile0][expert_id], 1);
        }
    }

    __syncthreads();

    const int tile1         = (ti_base + tokens_per_cta - 1) >> log_tile;
    const int n_local_tiles = tile1 - tile0 + 1;

    for (int i = 0; i < max_expert_num * local_tiles; i += block_dim) {
        const int t = (threadIdx.x + i) % local_tiles;
        const int e = (threadIdx.x + i) / local_tiles;
        if (e < expert_num && t < n_local_tiles) {
            const int v = smem.shared_accum[t][e];
            if (v) {
                atomicAdd(accum + e * tiles + (tile0 + t), v);
            }
        }
    }
}

static void launchMoeGateNoAuxTCKernel_V2(float*       scales,
                                          void*        masks,
                                          bool         is_routing,
                                          int*         accum,
                                          const float* logits,
                                          const bool*  token_mask,
                                          const float* correction_bias,
                                          int          log_tile,
                                          int          tiles,
                                          int          tokens,
                                          int          tokens_padded,
                                          int          experts,
                                          int          experts_per_token,
                                          bool         norm_topk,
                                          float        routed_scale,
                                          bool         use_sigmoid,
                                          cudaStream_t st)
{
    TM_CHECK_GE(tokens, 0);
    if (tokens == 0) {
        return;
    }
    TM_CHECK(token_mask);

    auto invoke = [&](auto max_expert_num,  //
                      auto top_k,
                      auto items_per_thread,
                      auto vec_size,
                      auto fuse_mask_clear,
                      auto routing) {
        constexpr bool kIsRouting = decltype(routing)::value;
        using Mask                = std::conditional_t<kIsRouting, int8_t, int>;

        constexpr int  thrs_per_tok   = max_expert_num.value / items_per_thread.value;
        constexpr int  threads        = 256;
        constexpr int  tokens_per_cta = threads / thrs_per_tok;
        constexpr bool kFuseMaskClear = fuse_mask_clear.value != 0;
        const int      blocks         = ceil_div(kFuseMaskClear ? tokens_padded : tokens, tokens_per_cta);

        auto* kernel = MoeGateNoAuxTCKernel_V2<max_expert_num.value,
                                               top_k.value,
                                               items_per_thread.value,
                                               threads,
                                               vec_size.value,
                                               kFuseMaskClear,
                                               kIsRouting,
                                               Mask>;

        if constexpr (kIsRouting && !kFuseMaskClear) {
            cudaMemsetAsync(masks, -1, sizeof(Mask) * experts * tokens_padded, st);
        }

        kernel<<<blocks, threads, 0, st>>>(scales,
                                           static_cast<Mask*>(masks),
                                           accum,
                                           logits,
                                           token_mask,
                                           correction_bias,
                                           log_tile,
                                           tiles,
                                           tokens,
                                           tokens_padded,
                                           experts,
                                           experts_per_token,
                                           use_sigmoid,
                                           norm_topk,
                                           routed_scale);
        return true;
    };

    auto dispatch = [&](auto routing) {
        if (experts == 64 && experts_per_token == 4) {
            return invoke(_Int<64>, _Int<4>, _Int<16>, _Int<4>, _Int<1>, routing);
        }
        if (experts == 160 && experts_per_token == 8) {
            return invoke(_Int<160>, _Int<8>, _Int<10>, _Int<2>, _Int<1>, routing);
        }
        return false;
    };

    auto dispatch_mode = [&] {
        if (is_routing) {
            return dispatch(std::true_type{});
        }
        else {
            return dispatch(std::false_type{});
        }
    };

    const bool success = dispatch_mode();
    TM_CHECK(success) << "unsupported noaux_tc config: expert_num=" << experts << ", top_k=" << experts_per_token;
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeMoeGate_NoAuxTC(float*       topk_weights,
                           int*         topk_indices,
                           const float* logits,
                           const bool*  token_mask,
                           const float* correction_bias,
                           int          tokens,
                           int          experts,
                           int          experts_per_token,
                           bool         norm_topk,
                           float        routed_scale,
                           bool         use_sigmoid,
                           cudaStream_t stream)
{
    launchMoeGateNoAuxTCKernel_V2(topk_weights,
                                  topk_indices,
                                  false,
                                  nullptr,
                                  logits,
                                  token_mask,
                                  correction_bias,
                                  0,
                                  0,
                                  tokens,
                                  tokens,
                                  experts,
                                  experts_per_token,
                                  norm_topk,
                                  routed_scale,
                                  use_sigmoid,
                                  stream);
}

void invokeMoeGate_NoAuxTC(int*         f2n,
                           int*         f2E,
                           int*         en2f,
                           int*         offsets,
                           float*       scales,
                           void*        masks,
                           int*         accum,
                           const float* logits,
                           const bool*  token_mask,
                           const float* correction_bias,
                           int          tokens,
                           int          tokens_padded,
                           int          experts,
                           int          exp_per_tok,
                           int          local_expert_offset,
                           int          local_expert_num,
                           bool         norm_topk_prob,
                           float        routed_scale,
                           bool         use_sigmoid,
                           cudaStream_t st)
{
    constexpr int base_log_tile = 9;
    int           log_tile      = base_log_tile;
    while (((tokens_padded + (1 << log_tile) - 1) >> log_tile) > kMoeGateMaxTiles) {
        ++log_tile;
    }
    const int tiles = ceil_div(tokens_padded, 1 << log_tile);

    cudaMemsetAsync(accum, 0, sizeof(int) * experts * kMoeGateMaxTiles, st);

    launchMoeGateNoAuxTCKernel_V2(scales,
                                  masks,
                                  true,
                                  accum,
                                  logits,
                                  token_mask,
                                  correction_bias,
                                  log_tile,
                                  tiles,
                                  tokens,
                                  tokens_padded,
                                  experts,
                                  exp_per_tok,
                                  norm_topk_prob,
                                  routed_scale,
                                  use_sigmoid,
                                  st);

    launchMoeScan_V2(f2n,
                     f2E,
                     en2f,
                     offsets,
                     (int8_t*)masks + local_expert_offset * tokens_padded,
                     accum + local_expert_offset * tiles,
                     log_tile,
                     tiles,
                     tokens,
                     tokens_padded,
                     local_expert_num,
                     base_log_tile,
                     st);

    TM_CUDA_CHECK(cudaGetLastError());
}

template<int vec_size, int block_dim, class T>
__global__ void MoeGatherKernel(T*         dst,  // [e*n, d]
                                const T*   src,  // [  n, d]
                                const int* f2n,  // [e*n] :: e*n -> n
                                const int* num_valid_tokens,
                                int        dims)
{
    if (num_valid_tokens && blockIdx.x >= __ldg(num_valid_tokens)) {
        return;
    }

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

void invokeMoeDispatch(Ref<Tensor>   out_,
                       const Tensor& src,
                       const int*    f2n,
                       int           num_worst_tokens,
                       const int*    num_valid_tokens,
                       cudaStream_t  st)
{
    auto& out    = out_.get();
    auto  invoke = [&](auto t) {
        using T                = decltype(t);
        const int     dim      = src.shape(1);
        constexpr int threads  = 256;
        constexpr int vec_size = 16 / sizeof(T);
        // f2n/out have num_worst_tokens rows; num_valid_tokens limits rows that read f2n.
        MoeGatherKernel<vec_size, threads><<<num_worst_tokens, threads, 0, st>>>(  //
            (T*)out.raw_data(),
            (const T*)src.raw_data(),
            f2n,
            num_valid_tokens,
            dim / vec_size);
        TM_CUDA_CHECK(cudaGetLastError());
    };
    TM_CHECK_EQ(src.dtype(), out.dtype());
    if (num_worst_tokens == 0) {
        return;
    }
    const auto elem_size = byte_size(src.dtype());
    if (elem_size == sizeof(uint16_t)) {
        invoke(uint16_t{});
    }
    else if (elem_size == sizeof(uint8_t)) {
        invoke(uint8_t{});
    }
    else {
        TM_LOG_FATAL("unsupported data type: {}", src.dtype());
    }
}

template<class T>
__global__ void MoeDispatchScalesNonaligned(
    T* dst, const T* src, int dst_stride, int src_stride, const int* f2n, const int* num_valid_tokens, int dim)
{
    const int bi = blockIdx.x;
    if (num_valid_tokens && bi >= __ldg(num_valid_tokens)) {
        return;
    }

    const int ti = f2n[bi];

    if (threadIdx.x < dim) {
        dst[threadIdx.x * dst_stride + bi] = src[threadIdx.x * src_stride + ti];
    }
}

void invokeMoeDispatchScales(Ref<Tensor>   out_,
                             const Tensor& src,
                             const int*    f2n,
                             int           num_worst_tokens,
                             const int*    num_valid_tokens,
                             cudaStream_t  st)
{
    using T                 = float;
    constexpr int alignment = 16 / sizeof(T);

    const int dim = src.shape(0);

    // Keep the scale layout aligned to num_worst_tokens; num_valid_tokens limits rows that read f2n.
    const int size         = num_worst_tokens;
    const int aligned_size = round_up<int>(size, alignment);

    auto& out = out_.get();

    if (!out) {
        out = Tensor_<T>{{{dim, size}, {aligned_size, 1}}, kDEVICE};
    }
    else {
        TM_CHECK(std::make_tuple(dim, size) == out.shapes(0, 1));
        TM_CHECK(out.stride(1) == 1);
        TM_CHECK(out.stride(0) % alignment == 0);
    }

    TM_CHECK_LE(dim, 1024);
    if (size == 0) {
        return;
    }
    const int threads = round_up<int>(dim, WARP_SIZE);
    const int blocks  = size;

    MoeDispatchScalesNonaligned<<<blocks, threads, 0, st>>>((T*)out.raw_data(),  //
                                                            (const T*)src.raw_data(),
                                                            out.stride(0),
                                                            src.stride(0),
                                                            f2n,
                                                            num_valid_tokens,
                                                            dim);

    TM_CUDA_CHECK(cudaGetLastError());
}

template<int vec_size, int exp_k, bool has_bias, int block_dim, class T>
__global__ void MoeReduceKernel(T*           dst,         // [  n, d]
                                const T*     src,         // [e*n, d]
                                const T*     bias,        // [  E, d]
                                const float* scales,      // [  e, n]
                                const int*   en2f,        // [  e, n] :: (e,n) -> e*n
                                const int*   f2E,         // [  e* n]
                                const float* dst_scales,  // [n]
                                int          dim,
                                int          tokens,
                                T            bscale,
                                float        dst_scale)
{
    if constexpr (TURBOMIND_ARCH_DTYPE_GUARD(data_type_v<T>)) {
        const int64_t ti = blockIdx.x;

        dst += (int64_t)dim * ti;

        if (dst_scales) {
            const float scale = dst_scales[ti];
            dst_scale *= fdividef(1.f, 1.f + expf(-scale));
        }

        // Should be warp uniforms
        const T* src_[exp_k]{};
        const T* bias_[exp_k]{};

        float scale[exp_k]{};

        PRAGMA_UNROLL
        for (int e = 0; e < exp_k; ++e) {
            int fid = __ldg(&en2f[e * tokens + ti]);
            if (fid >= 0) {
                src_[e] = src + (int64_t)dim * fid;
                if constexpr (has_bias) {
                    bias_[e] = bias + __ldg(&f2E[fid]) * (int64_t)dim;
                }
                scale[e] = scales ? __ldg(&scales[e * tokens + ti]) : 1.f;
            }
        }

        using Vec = Array<T, vec_size>;

        for (int i = threadIdx.x * vec_size; i < dim; i += block_dim * vec_size) {
            Array<float, vec_size> accum{};
            if (dst_scale) {
                Vec v;
                Load(v, &dst[i]);
                using namespace ops;
                accum = cast<float>(v) * dst_scale;
            }
            PRAGMA_UNROLL
            for (int e = 0; e < exp_k; ++e) {
                if (src_[e] == nullptr) {
                    continue;
                }
                Vec v;
                Load(v, src_[e] + i);
                using namespace ops;
                if constexpr (has_bias) {
                    Vec b;
                    Load(b, bias_[e] + i);
                    PRAGMA_UNROLL
                    for (int i = 0; i < vec_size; ++i) {
                        v[i] = __hfma(b[i], bscale, v[i]);
                    }
                }
                const auto x = cast<float>(v) * scale[e];
                accum        = accum + x;
            }
            Store(&dst[i], cast<T>(accum));
        }
    }
}

template<bool has_bias, class T>
void invokeMoeReduce(T*           dst,
                     const T*     src,
                     const T*     bias,
                     const float* scales,
                     const int*   en2f,
                     const int*   f2E,
                     const float* dst_scales,
                     int          tokens,
                     int          experts_per_token,
                     int          dim,
                     T            bscale,
                     float        dst_scale,
                     cudaStream_t st)
{
    const auto invoke = [&](auto e) {
        constexpr int threads     = 256;
        constexpr int vec_size    = 16 / sizeof(T);
        constexpr int exp_per_tok = decltype(e)::value;
        MoeReduceKernel<vec_size, exp_per_tok, has_bias, threads><<<tokens, threads, 0, st>>>(  //
            dst,
            src,
            bias,
            scales,
            en2f,
            f2E,
            dst_scales,
            dim,
            tokens,
            bscale,
            dst_scale);
        TM_CUDA_CHECK(cudaGetLastError());
    };

    switch (experts_per_token) {
        case 1:
            invoke(std::integral_constant<int, 1>{});
            break;
        case 2:
            invoke(std::integral_constant<int, 2>{});
            break;
        case 4:
            invoke(std::integral_constant<int, 4>{});
            break;
        case 6:
            invoke(std::integral_constant<int, 6>{});
            break;
        case 8:
            invoke(std::integral_constant<int, 8>{});
            break;
        case 10:
            invoke(std::integral_constant<int, 10>{});
            break;
        default:
            fprintf(stderr, "Unsupported experts_per_token %d\n", experts_per_token);
            std::abort();
    }
}

void invokeMoeCombine(Ref<Tensor>   out_,
                      const Tensor& src,
                      const Tensor& bias,
                      const float*  scales,
                      const int*    en2f,
                      const int*    f2E,
                      const float*  dst_scales,
                      int           experts_per_token,
                      float         bscale,
                      float         dst_scale,
                      cudaStream_t  st)
{
    auto& out = out_.get();

    const int tokens = out.shape(0);
    TM_CHECK_EQ(src.shape(0), tokens * experts_per_token);

    auto invoke = [&](auto has_bias, auto t) {
        using T = decltype(t);
        invokeMoeReduce<has_bias.value>(out.data<T>(),
                                        src.data<T>(),
                                        bias.data_or((T*)nullptr),
                                        scales,
                                        en2f,
                                        f2E,
                                        dst_scales,
                                        tokens,
                                        experts_per_token,
                                        src.shape(1),
                                        (T)bscale,
                                        dst_scale,
                                        st);
    };

    auto dispatch_dtype = [&](auto t) {
        if (bias) {
            TM_CHECK_NOTNULL(f2E);
            invoke(std::true_type{}, t);
        }
        else {
            invoke(std::false_type{}, t);
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(src.dtype(), dispatch_dtype);
}

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

template<int max_expert_num, int items_per_thread, int access_size>
__global__ void MoeSoftmaxMaskTopKGroups(float* logits, int token_num, int expert_num, int top_k)
{
    constexpr int threads_per_token = max_expert_num / items_per_thread;

    static_assert((threads_per_token & (threads_per_token - 1)) == 0);
    static_assert(items_per_thread % access_size == 0);

    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int ti = thread_idx / threads_per_token;
    const int ei = thread_idx % threads_per_token;

    float data[items_per_thread];
    PRAGMA_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        data[i] = -std::numeric_limits<float>::infinity();
    }
    // max logit in the group
    float max_val = -std::numeric_limits<float>::infinity();
    if (ti < token_num) {
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; i += access_size) {
            const int e = ei * items_per_thread + i;  // blocked partition
            if (e < expert_num) {
                Ldg((Array<float, access_size>&)data[i], &logits[ti * expert_num + e]);
                PRAGMA_UNROLL
                for (int c = 0; c < access_size; ++c) {
                    max_val = fmaxf(max_val, data[i + c]);
                }
            }
        }
    }

    const int warp_ti        = threadIdx.x % WARP_SIZE / threads_per_token;
    const int warp_ti_offset = warp_ti * threads_per_token;

    bool  alive     = false;
    float max_logit = 0;

    for (int k = 0; k < top_k; ++k) {
        int   g_max_ei  = ei;
        float g_max_val = max_val;
        PRAGMA_UNROLL
        for (int m = threads_per_token / 2; m >= 1; m /= 2) {
            g_max_val = fmaxf(g_max_val, __shfl_xor_sync((uint32_t)-1, g_max_val, m));
        }
        // tie breaking
        const auto active = __ballot_sync((uint32_t)-1, max_val == g_max_val);
        g_max_ei          = __ffs(active >> (unsigned)warp_ti_offset) - 1;
        if (k == 0) {
            max_logit = g_max_val;
        }
        if (ei == g_max_ei) {
            alive   = true;
            max_val = -std::numeric_limits<float>::infinity();
        }
    }

    float sum_prob{};

    PRAGMA_NO_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        data[i] = expf(data[i] - max_logit);
        sum_prob += data[i];
    }

    PRAGMA_UNROLL
    for (int m = threads_per_token / 2; m >= 1; m /= 2) {
        sum_prob += __shfl_xor_sync((uint32_t)-1, sum_prob, m);
    }

    // mask dead logits
    sum_prob = alive ? fdividef(1.f, sum_prob) : 0;

    PRAGMA_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        data[i] *= sum_prob;
    }

    if (ti < token_num) {
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; i += access_size) {
            const int e = ei * items_per_thread + i;
            if (e < expert_num) {
                Store(&logits[ti * expert_num + e], (Array<float, access_size>&)data[i]);
            }
        }
    }
}

void invokeMoeSoftmaxMaskTopKGroups(
    float* logits, int token_num, int expert_num, int group_size, int top_k, cudaStream_t st)
{
    auto invoke = [&](auto max_expert_num, auto items_per_thread, auto vec_size) {
        constexpr int thrs_per_tok = max_expert_num.value / items_per_thread.value;
        constexpr int threads      = 256;
        const int     blocks       = ceil_div(token_num, threads / thrs_per_tok);
        MoeSoftmaxMaskTopKGroups<max_expert_num.value, items_per_thread.value, vec_size.value>
            <<<blocks, threads, 0, st>>>(logits, token_num, expert_num, top_k);
        TM_CUDA_CHECK(cudaGetLastError());
    };

    if (expert_num == 160 && group_size == 20) {
        invoke(_Int<160>, _Int<20>, _Int<4>);
        return;
    }

    std::cerr << __FILE__ << "(" << __LINE__ << "): unsupported moe config: expert_num=" << expert_num
              << ", group_size=" << group_size << "\n";
    std::abort();
}

}  // namespace turbomind
