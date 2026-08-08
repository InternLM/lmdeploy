// Copyright (c) OpenMMLab. All rights reserved.
//
// Merged CUDA kernels for the unified Qwen ViT (Qwen2-VL / Qwen2.5-VL / Qwen3.5).
// Sections, in order:
//   1. QKV preprocessing (bias + RoPE fuse)        — all variants
//   2. Spatial-merge index mapping                  — all variants
//   3. Learned pos-embed bilinear interpolation     — Qwen3.5
//   4. 2D rotary position-embedding table           — all variants
//   5. mrope position ids                           — all variants
//   6. Window attention reordering                  — Qwen2.5

#include "src/turbomind/models/qwenvit/qwenvit_kernels.h"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <type_traits>

namespace turbomind {

namespace {

// `num_grids` is tiny (usually 1..a few) so a linear scan is fine.
// Shared by the pos-embed interpolation and rotary-embedding kernels.
__device__ inline int find_grid(const int* offsets, int num_grids, int pos)
{
    int g = 0;
    for (int i = 1; i < num_grids; ++i) {
        if (offsets[i * 2 + 1] <= pos) {
            g = i;
        }
        else {
            break;
        }
    }
    return g;
}

// ------------------------------------------------------------------------------------
// 1. QKV preprocessing
// ------------------------------------------------------------------------------------

constexpr int kWarpsPerBlock = 4;

// Per-head_dim launch traits. Adding a new head_dim is a single specialization here.
template<int HD>
struct HeadConfig;

template<>
struct HeadConfig<64> {
    static constexpr int kVecSize      = 8;
    static constexpr int kHeadsPerWarp = 4;
};

template<>
struct HeadConfig<72> {
    static constexpr int kVecSize      = 8;
    static constexpr int kHeadsPerWarp = 3;
};

template<>
struct HeadConfig<128> {
    static constexpr int kVecSize      = 8;  // 128 / 8 = 16 vec/head
    static constexpr int kHeadsPerWarp = 2;  // 16 * 2 = 32 == WARP_SIZE
};

template<int VecSize, typename T>
__device__ __forceinline__ void add_bias(Array<float, VecSize>& x, const T* bias)
{
    Array<T, VecSize> b;
    Ldg(b, bias);
    using namespace ops;
    x = x + cast<float>(b);
}

// Rotate adjacent (x[2k], x[2k+1]) pairs using cos/sin packed as
// rope[2k]=cos, rope[2k+1]=sin (see fastRotaryPosEmbKernel below).
template<int VecSize, typename T>
__device__ __forceinline__ void apply_rope_pair(Array<float, VecSize>& x, const Array<T, VecSize>& rope)
{
    auto cs = cast<float>(rope);
    PRAGMA_UNROLL
    for (int i = 0; i < VecSize; i += 2) {
        const float x0 = x[i];
        const float x1 = x[i + 1];
        const float c  = cs[i];
        const float s  = cs[i + 1];
        x[i]           = c * x0 - s * x1;
        x[i + 1]       = c * x1 + s * x0;
    }
}

// Load `src`, add `bias`, apply RoPE, store to `dst` (may equal `src` for in-place).
template<int VecSize, typename T>
__device__ __forceinline__ void fuse_bias_rope_store(T* dst, const T* src, const T* bias, const Array<T, VecSize>& rope)
{
    Array<T, VecSize> x_vec;
    Load(x_vec, src);
    auto x = cast<float>(x_vec);
    add_bias(x, bias);
    apply_rope_pair(x, rope);
    Store(dst, cast<T>(x));
}

// V-path: bias only, no RoPE.
template<int VecSize, typename T>
__device__ __forceinline__ void fuse_bias_store(T* dst, const T* src, const T* bias)
{
    Array<T, VecSize> x_vec;
    Load(x_vec, src);
    auto x = cast<float>(x_vec);
    add_bias(x, bias);
    Store(dst, cast<T>(x));
}

template<typename T, int HD>
__global__ __launch_bounds__(kWarpsPerBlock* WARP_SIZE) void prepareQKVKernel(T* __restrict__ qkv,
                                                                              T* __restrict__ kv,
                                                                              const T* __restrict__ bias,
                                                                              const T* __restrict__ rotary_pos_emb,
                                                                              const int* __restrict__ mapped_idx,
                                                                              int token_num,
                                                                              int local_head_num,
                                                                              int head_group_num,
                                                                              int rope_head_dim)
{
    using Cfg                   = HeadConfig<HD>;
    constexpr int kVecSize      = Cfg::kVecSize;
    constexpr int kHeadsPerWarp = Cfg::kHeadsPerWarp;
    constexpr int kVecPerHead   = HD / kVecSize;
    static_assert(HD % kVecSize == 0);
    static_assert(kVecPerHead * kHeadsPerWarp <= WARP_SIZE);

    const int warp_id   = threadIdx.x / WARP_SIZE;
    const int lane      = threadIdx.x - warp_id * WARP_SIZE;
    const int head_slot = lane / kVecPerHead;
    if (head_slot >= kHeadsPerWarp) {
        return;
    }

    const int global_warp = blockIdx.x * kWarpsPerBlock + warp_id;
    const int total_warps = token_num * head_group_num;
    if (global_warp >= total_warps) {
        return;
    }

    const int token_idx  = global_warp / head_group_num;
    const int head_group = global_warp - token_idx * head_group_num;
    const int head_idx   = head_group * kHeadsPerWarp + head_slot;
    if (head_idx >= local_head_num) {
        return;
    }

    const int vec_idx = lane - head_slot * kVecPerHead;
    const int di      = vec_idx * kVecSize;

    // QKV per-token layout: [Q_heads | K_heads | V_heads], head_num == kv_head_num for ViT.
    const int64_t qkv_stride = (int64_t)local_head_num * 3 * HD;
    T* const      q_ptr      = qkv + (int64_t)token_idx * qkv_stride + head_idx * HD + di;
    const T*      k_ptr      = q_ptr + (int64_t)local_head_num * HD;
    const T*      v_ptr      = k_ptr + (int64_t)local_head_num * HD;

    const T* q_bias = bias + head_idx * HD + di;
    const T* k_bias = q_bias + local_head_num * HD;
    const T* v_bias = k_bias + local_head_num * HD;

    // K/V destination in transposed [kv_head, 2, token, head_dim] layout.
    T* const k_dst = kv + ((int64_t)head_idx * 2 * token_num + token_idx) * HD + di;
    T* const v_dst = k_dst + (int64_t)token_num * HD;

    // rope[token, di] is shared between Q and K — load once, reuse twice.
    // When HD > rope_head_dim, padded di-slices have zero Q/K, so loading a
    // zero rope_vec there is correct (and avoids OOB on the [N, rope_head_dim]
    // buffer). kVecSize is aligned to rope_head_dim so each vec is fully in or
    // fully out of the rope range.
    Array<T, kVecSize> rope_vec{};
    if (di < rope_head_dim) {
        Ldg(rope_vec, rotary_pos_emb + (int64_t)mapped_idx[token_idx] * rope_head_dim + di);
    }

    fuse_bias_rope_store<kVecSize>(q_ptr, q_ptr, q_bias, rope_vec);  // Q: in-place
    fuse_bias_rope_store<kVecSize>(k_dst, k_ptr, k_bias, rope_vec);  // K: transposed
    fuse_bias_store<kVecSize>(v_dst, v_ptr, v_bias);                 // V: transposed, no RoPE
}

template<typename T>
void dispatchPrepareQKV(T*           qkv,
                        T*           kv,
                        const T*     qkv_bias,
                        const T*     rotary_pos_emb,
                        const int*   mapped_idx,
                        int          token_num,
                        int          local_head_num,
                        int          head_dim,
                        int          rope_head_dim,
                        cudaStream_t stream)
{
    auto invoke = [&](auto hd_c) {
        constexpr int HD = decltype(hd_c)::value;
        using Cfg        = HeadConfig<HD>;

        // Each vec_size-wide load must lie entirely in or out of the rope range.
        TM_CHECK(rope_head_dim % Cfg::kVecSize == 0)
            << "rope_head_dim (" << rope_head_dim << ") must be a multiple of kVecSize (" << Cfg::kVecSize << ")";
        TM_CHECK(rope_head_dim <= HD) << "rope_head_dim (" << rope_head_dim << ") cannot exceed head_dim (" << HD
                                      << ")";

        const int head_group_num = (local_head_num + Cfg::kHeadsPerWarp - 1) / Cfg::kHeadsPerWarp;
        const int total_warps    = token_num * head_group_num;
        dim3      grid((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
        prepareQKVKernel<T, HD><<<grid, kWarpsPerBlock * WARP_SIZE, 0, stream>>>(
            qkv, kv, qkv_bias, rotary_pos_emb, mapped_idx, token_num, local_head_num, head_group_num, rope_head_dim);
    };

    switch (head_dim) {
        case 64:
            return invoke(std::integral_constant<int, 64>{});
        case 72:
            return invoke(std::integral_constant<int, 72>{});
        case 128:
            return invoke(std::integral_constant<int, 128>{});
        default:
            TM_LOG_FATAL("unsupported Qwen ViT head_dim for qkv preprocess: {}", head_dim);
    }
}

// ------------------------------------------------------------------------------------
// 2. Spatial-merge index mapping
// ------------------------------------------------------------------------------------

__global__ void buildMappedIdxKernel(int* mapped_idx, int token_offset, int natural_offset, int t, int h, int w, int S)
{
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = t * h * w;
    if (idx >= total) {
        return;
    }

    const int hw                   = h * w;
    const int merge_unit           = S * S;
    const int local                = idx % hw;
    const int group                = local / merge_unit;
    const int inner                = local - group * merge_unit;
    const int group_cols           = w / S;
    const int h_outer              = group / group_cols;
    const int w_outer              = group - h_outer * group_cols;
    const int h_inner              = inner / S;
    const int w_inner              = inner - h_inner * S;
    const int natural_idx          = (h_outer * S + h_inner) * w + (w_outer * S + w_inner);
    mapped_idx[token_offset + idx] = natural_offset + natural_idx;
}

__global__ void
buildMappedIdxBatchedKernel(int* mapped_idx, const int* grid_thws, const int* grid_offsets, int num_grids, int S)
{
    const int grid_id = blockIdx.x;
    if (grid_id >= num_grids) {
        return;
    }

    const int t              = grid_thws[grid_id * 3];
    const int h              = grid_thws[grid_id * 3 + 1];
    const int w              = grid_thws[grid_id * 3 + 2];
    const int token_offset   = grid_offsets[grid_id * 2];
    const int natural_offset = grid_offsets[grid_id * 2 + 1];
    const int total          = t * h * w;
    const int hw             = h * w;
    const int merge_unit     = S * S;
    const int group_cols     = w / S;

    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int local                = idx % hw;
        const int group                = local / merge_unit;
        const int inner                = local - group * merge_unit;
        const int h_outer              = group / group_cols;
        const int w_outer              = group - h_outer * group_cols;
        const int h_inner              = inner / S;
        const int w_inner              = inner - h_inner * S;
        const int natural_idx          = (h_outer * S + h_inner) * w + (w_outer * S + w_inner);
        mapped_idx[token_offset + idx] = natural_offset + natural_idx;
    }
}

// ------------------------------------------------------------------------------------
// 3. Learned pos-embed bilinear interpolation (Qwen3.5)
// ------------------------------------------------------------------------------------

template<typename T>
__device__ inline T from_float(float x);

template<>
__device__ inline half from_float<half>(float x)
{
    return __float2half(x);
}

#ifdef ENABLE_BF16
template<>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float x)
{
    return __float2bfloat16(x);
}
#endif

template<typename T>
__global__ void fastPosEmbedIdxWeightKernel(
    int* idx_out, T* weight_out, const int* grid_thws, const int* grid_offsets, int num_grids, int total_n, int G)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total_n) {
        return;
    }

    const int g      = find_grid(grid_offsets, num_grids, pos);
    const int grid_h = grid_thws[g * 3 + 1];
    const int grid_w = grid_thws[g * 3 + 2];
    const int local  = pos - grid_offsets[g * 2 + 1];
    const int i      = local / grid_w;
    const int j      = local % grid_w;

    // torch.linspace(0, G-1, n) uses the halfway-symmetric formulation so
    // that both endpoints are exact:
    //   step      = (end - start) / (n - 1)
    //   halfway   = n / 2
    //   out[i<hw] = start + step * i
    //   out[i>=hw]= end   - step * (n - 1 - i)
    // For n == 1 the single element is `start` (== 0 here); the formula
    // below collapses to 0 since hw_h == 0 is bypassed via grid_h > 1.
    const float end    = (float)(G - 1);
    const float step_h = (grid_h > 1) ? end / (float)(grid_h - 1) : 0.f;
    const float step_w = (grid_w > 1) ? end / (float)(grid_w - 1) : 0.f;

    const int hw_h = grid_h / 2;
    const int hw_w = grid_w / 2;

    const float h_val = (grid_h == 1) ? 0.f : ((i < hw_h) ? step_h * (float)i : end - step_h * (float)(grid_h - 1 - i));
    const float w_val = (grid_w == 1) ? 0.f : ((j < hw_w) ? step_w * (float)j : end - step_w * (float)(grid_w - 1 - j));

    // torch.Tensor.int() truncates toward zero; h_val, w_val are non-negative
    // and bounded above by G-1, so (int) cast is in [0, G-1].
    const int h_floor = (int)h_val;
    const int w_floor = (int)w_val;
    const int h_ceil  = min(h_floor + 1, G - 1);
    const int w_ceil  = min(w_floor + 1, G - 1);

    const float dh = h_val - (float)h_floor;
    const float dw = w_val - (float)w_floor;

    const int base_h      = h_floor * G;
    const int base_h_ceil = h_ceil * G;

    Array<int, 4> idx;
    idx[0] = base_h + w_floor;
    idx[1] = base_h + w_ceil;
    idx[2] = base_h_ceil + w_floor;
    idx[3] = base_h_ceil + w_ceil;

    Array<T, 4> weight;
    weight[0] = from_float<T>((1.f - dh) * (1.f - dw));
    weight[1] = from_float<T>((1.f - dh) * dw);
    weight[2] = from_float<T>(dh * (1.f - dw));
    weight[3] = from_float<T>(dh * dw);

    const int out_base = pos * 4;
    Store(idx_out + out_base, idx);
    Store(weight_out + out_base, weight);
}

template<typename T, int N>
__device__ Array<float, N> roundToStorageDtype(Array<float, N> x)
{
    return cast<float>(cast<T>(x));
}

template<int vec_size, typename T>
__global__ void fusedPosEmbedMergeKernel(T*         hidden_states,
                                         const T*   pos_embeds,
                                         const T*   pos_embed_weights,
                                         const int* mapped_idx,
                                         const T*   bias,
                                         int        hidden,
                                         int        vdim)
{
    const int index  = blockIdx.x;
    const int mapped = mapped_idx[index];  // same address for all threads in block -> L1 broadcast

    Array<T, 4> w4;
    Ldg(w4, pos_embed_weights + mapped * 4);

    const int row_off = index * hidden;
    const int pe_row0 = mapped * 4 * hidden;

    using namespace ops;
    for (int d = threadIdx.x; d < vdim; d += blockDim.x) {
        Array<float, vec_size> pos{};
        Array<T, vec_size>     tmp;
        Load(tmp, hidden_states + row_off + d * vec_size);
        auto hidden_acc = cast<float>(tmp);

        if (bias) {
            Ldg(tmp, bias + d * vec_size);
            hidden_acc = roundToStorageDtype<T>(hidden_acc + cast<float>(tmp));
        }
        PRAGMA_UNROLL
        for (int k = 0; k < 4; ++k) {
            Ldg(tmp, pos_embeds + pe_row0 + k * hidden + d * vec_size);
            pos = pos + cast<float>(tmp * w4[k]);
        }
        const auto out = hidden_acc + roundToStorageDtype<T>(pos);
        Store(hidden_states + row_off + d * vec_size, cast<T>(out));
    }
}

// ------------------------------------------------------------------------------------
// 4. 2D rotary position-embedding table
// ------------------------------------------------------------------------------------

template<typename T>
__global__ void fastRotaryPosEmbKernel(T*         cos_sin_out,
                                       const int* grid_thws,
                                       const int* grid_offsets,
                                       int        num_grids,
                                       int        total_hw,
                                       int        head_dim,
                                       float      scale)  // -log2(theta) / (head_dim/4)
{
    const int pair_count = head_dim / 2;  // e.g. 36
    const int freq_half  = head_dim / 4;  // e.g. 18

    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int pos    = tid / pair_count;
    const int pair_k = tid % pair_count;
    if (pos >= total_hw) {
        return;
    }

    const int g      = find_grid(grid_offsets, num_grids, pos);
    const int grid_w = grid_thws[g * 3 + 2];
    const int local  = pos - grid_offsets[g * 2 + 1];
    const int i      = local / grid_w;  // h_coord
    const int j      = local % grid_w;  // w_coord

    // Pairs [0, freq_half) rotate in h; pairs [freq_half, 2*freq_half) rotate in w.
    const int   freq_idx = pair_k % freq_half;
    const int   coord    = (pair_k < freq_half) ? i : j;
    const float inv_freq = exp2f((float)freq_idx * scale);

    float c, s;
    sincosf((float)coord * inv_freq, &s, &c);

    Array<T, 2> cs{(T)c, (T)s};
    Store(cos_sin_out + (size_t)pos * head_dim + pair_k * 2, cs);
}

// ------------------------------------------------------------------------------------
// 5. mrope position ids
// ------------------------------------------------------------------------------------

constexpr int kMropeBlock = 128;

__global__ void mropeScatterKernel(int* pos_ids, const MropeSegment* __restrict__ segs)
{
    const MropeSegment s       = segs[blockIdx.x];
    const int          local_k = blockIdx.y * blockDim.x + threadIdx.x;
    if (local_k >= s.n_tok) {
        return;
    }
    int* dst = pos_ids + 3 * (s.dst_offset + local_k);
    if (s.h2 == 0) {  // text run
        const int p = s.base_pos + local_k;
        dst[0]      = p;
        dst[1]      = p;
        dst[2]      = p;
    }
    else {  // image run - grid math uses the original (un-clipped) k
        const int k  = s.k_offset + local_k;
        const int hw = s.h2 * s.w2;
        dst[0]       = s.base_pos + k / hw;
        dst[1]       = s.base_pos + (k / s.w2) % s.h2;
        dst[2]       = s.base_pos + k % s.w2;
    }
}

// ------------------------------------------------------------------------------------
// 6. Window attention reordering (Qwen2.5)
// ------------------------------------------------------------------------------------

template<int vec_size, typename T>
__global__ void windowReorderKernel(T*         out,
                                    const T*   in,
                                    const int* window_idx,
                                    int64_t    out_stride,
                                    int64_t    in_stride,
                                    int        merge_unit,
                                    int        group_count,
                                    int        dim)
{
    const int dst_group = blockIdx.x;
    const int inner     = blockIdx.y;
    const int di        = (threadIdx.x + blockIdx.z * blockDim.x) * vec_size;
    if (di >= dim) {
        return;
    }

    const int src_group = window_idx[dst_group];
    using Vec           = Array<T, vec_size>;
    Vec x;
    Load(x, in + ((int64_t)src_group * merge_unit + inner) * in_stride + di);
    Store(out + ((int64_t)dst_group * merge_unit + inner) * out_stride + di, x);
}

template<int vec_size, typename T>
__global__ void reverseWindowKernel(
    T* out, const T* in, const int* window_idx, int64_t out_stride, int64_t in_stride, int group_count, int dim)
{
    const int src_group = blockIdx.x;
    const int di        = (threadIdx.x + blockIdx.y * blockDim.x) * vec_size;
    if (di >= dim) {
        return;
    }

    const int dst_group = window_idx[src_group];
    using Vec           = Array<T, vec_size>;
    Vec x;
    Load(x, in + (int64_t)src_group * in_stride + di);
    Store(out + (int64_t)dst_group * out_stride + di, x);
}

__global__ void buildWindowMappedIdxKernel(
    int* window_mapped_idx, const int* mapped_idx, const int* window_idx, int merge_unit, int total)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int dst_group    = idx / merge_unit;
    const int inner        = idx - dst_group * merge_unit;
    const int src_group    = window_idx[dst_group];
    window_mapped_idx[idx] = mapped_idx[src_group * merge_unit + inner];
}

}  // namespace

// =====================================================================================
// Public entry points
// =====================================================================================

void invokeQwenVitPrepareQKV(void*        qkv,
                             void*        kv,
                             const void*  qkv_bias,
                             const void*  rotary_pos_emb,
                             const int*   mapped_idx,
                             DataType     dtype,
                             int          token_num,
                             int          local_head_num,
                             int          head_dim,
                             int          rope_head_dim,
                             cudaStream_t stream)
{
    if (token_num == 0) {
        return;
    }

    auto invoke = [&](auto t) {
        using T = decltype(t);
        dispatchPrepareQKV((T*)qkv,
                           (T*)kv,
                           (const T*)qkv_bias,
                           (const T*)rotary_pos_emb,
                           mapped_idx,
                           token_num,
                           local_head_num,
                           head_dim,
                           rope_head_dim,
                           stream);
    };

    TM_DISPATCH_PRIMARY_DTYPES(dtype, invoke);
}

void invokeQwenVitBuildMappedIdx(int*         mapped_idx,
                                 int          token_offset,
                                 int          natural_offset,
                                 int          t,
                                 int          h,
                                 int          w,
                                 int          spatial_merge_size,
                                 cudaStream_t stream)
{
    if (t * h * w == 0) {
        return;
    }

    const int total   = t * h * w;
    const int threads = 256;
    buildMappedIdxKernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
        mapped_idx, token_offset, natural_offset, t, h, w, spatial_merge_size);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeQwenVitBuildMappedIdx(int*         mapped_idx,
                                 const int*   grid_thws,
                                 const int*   grid_offsets,
                                 int          num_grids,
                                 int          spatial_merge_size,
                                 cudaStream_t stream)
{
    if (num_grids == 0) {
        return;
    }

    const int threads = 256;
    buildMappedIdxBatchedKernel<<<num_grids, threads, 0, stream>>>(
        mapped_idx, grid_thws, grid_offsets, num_grids, spatial_merge_size);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeFastPosEmbedIdxWeight(int*         idx_out,
                                 void*        weight_out,
                                 DataType     dtype,
                                 const int*   grid_thws,
                                 const int*   grid_offsets,
                                 int          num_grids,
                                 int          total_n,
                                 int          num_grid_per_side,
                                 cudaStream_t stream)
{
    if (total_n <= 0 || num_grids <= 0) {
        return;
    }
    const int block = 256;
    const int grid  = (total_n + block - 1) / block;

    auto invoke = [&](auto t) {
        using T = decltype(t);
        fastPosEmbedIdxWeightKernel<T><<<grid, block, 0, stream>>>(
            idx_out, (T*)weight_out, grid_thws, grid_offsets, num_grids, total_n, num_grid_per_side);
    };
    TM_DISPATCH_PRIMARY_DTYPES(dtype, invoke);
}

void invokeFusedPosEmbedMerge(void*        hidden_states,
                              const void*  pos_embeds,
                              const void*  pos_embed_weights,
                              const int*   mapped_idx,
                              const void*  bias,
                              int          batch,
                              int          hidden,
                              DataType     dtype,
                              cudaStream_t stream)
{
    if (batch <= 0) {
        return;
    }

    const dim3 grid(batch);
    const dim3 block(128);

    auto invoke = [&](auto t) {
        using T                = decltype(t);
        constexpr int vec_size = sizeof(uint4) / sizeof(T);
        TM_CHECK(hidden % vec_size == 0);
        fusedPosEmbedMergeKernel<vec_size, T><<<grid, block, 0, stream>>>((T*)hidden_states,
                                                                          (const T*)pos_embeds,
                                                                          (const T*)pos_embed_weights,
                                                                          mapped_idx,
                                                                          (const T*)bias,
                                                                          hidden,
                                                                          hidden / vec_size);
    };
    TM_DISPATCH_PRIMARY_DTYPES(dtype, invoke);
}

void invokeQwenVitRotaryPosEmb(void*        cos_sin,
                               DataType     dtype,
                               const int*   grid_thws,
                               const int*   grid_offsets,
                               int          num_grids,
                               int          total_hw,
                               int          head_dim,
                               float        theta,
                               cudaStream_t stream)
{
    if (total_hw <= 0 || num_grids <= 0 || head_dim <= 0) {
        return;
    }
    TM_CHECK(head_dim % 4 == 0) << "head_dim must be divisible by 4, got " << head_dim;

    const int   total = total_hw * (head_dim / 2);
    const int   block = 256;
    const int   grid  = (total + block - 1) / block;
    const float scale = -log2f(theta) / (float)(head_dim / 4);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        fastRotaryPosEmbKernel<T>
            <<<grid, block, 0, stream>>>((T*)cos_sin, grid_thws, grid_offsets, num_grids, total_hw, head_dim, scale);
    };
    TM_DISPATCH_PRIMARY_DTYPES(dtype, invoke);
}

void invokeMropePositionIds(
    int* pos_ids, const MropeSegment* segments, int num_segments, int max_seg_len, cudaStream_t stream)
{
    if (num_segments <= 0 || max_seg_len <= 0) {
        return;
    }
    const int  tiles = (max_seg_len + kMropeBlock - 1) / kMropeBlock;
    const dim3 grid((unsigned)num_segments, (unsigned)tiles);
    mropeScatterKernel<<<grid, kMropeBlock, 0, stream>>>(pos_ids, segments);
}

void invokeQwenVitWindowReorder(
    Tensor& out, const Tensor& in, const int* window_idx, int merge_unit, int group_count, cudaStream_t stream)
{
    if (group_count == 0) {
        return;
    }

    const int dim     = in.shape(1);
    const int threads = 256;

    auto invoke = [&](auto t) {
        using T               = decltype(t);
        constexpr int max_vec = sizeof(uint4) / sizeof(T);

        int best_vec_size = 1;
        for (int v = max_vec; v >= 1; v >>= 1) {
            if (dim % v == 0 && in.stride(0) % v == 0 && out.stride(0) % v == 0) {
                best_vec_size = v;
                break;
            }
        }

        auto launch = [&](auto vec_size_c) {
            constexpr int vec_size = decltype(vec_size_c)::value;
            const dim3    grid(group_count, merge_unit, cdiv(dim, threads * vec_size));
            windowReorderKernel<vec_size, T><<<grid, threads, 0, stream>>>(
                out.data<T>(), in.data<T>(), window_idx, out.stride(0), in.stride(0), merge_unit, group_count, dim);
        };

        switch (best_vec_size) {
            case 8:
                return launch(std::integral_constant<int, 8>{});
            case 4:
                return launch(std::integral_constant<int, 4>{});
            case 2:
                return launch(std::integral_constant<int, 2>{});
            default:
                return launch(std::integral_constant<int, 1>{});
        }
    };
    TM_DISPATCH_PRIMARY_DTYPES(in.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeQwenVitReverseWindow(
    Tensor& out, const Tensor& in, const int* window_idx, int group_count, cudaStream_t stream)
{
    if (group_count == 0) {
        return;
    }

    const int dim     = in.shape(1);
    const int threads = 256;

    auto invoke = [&](auto t) {
        using T               = decltype(t);
        constexpr int max_vec = sizeof(uint4) / sizeof(T);

        int best_vec_size = 1;
        for (int v = max_vec; v >= 1; v >>= 1) {
            if (dim % v == 0 && in.stride(0) % v == 0 && out.stride(0) % v == 0) {
                best_vec_size = v;
                break;
            }
        }

        auto launch = [&](auto vec_size_c) {
            constexpr int vec_size = decltype(vec_size_c)::value;
            const dim3    grid(group_count, cdiv(dim, threads * vec_size));
            reverseWindowKernel<vec_size, T><<<grid, threads, 0, stream>>>(
                out.data<T>(), in.data<T>(), window_idx, out.stride(0), in.stride(0), group_count, dim);
        };

        switch (best_vec_size) {
            case 8:
                return launch(std::integral_constant<int, 8>{});
            case 4:
                return launch(std::integral_constant<int, 4>{});
            case 2:
                return launch(std::integral_constant<int, 2>{});
            default:
                return launch(std::integral_constant<int, 1>{});
        }
    };
    TM_DISPATCH_PRIMARY_DTYPES(in.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeQwenVitBuildWindowMappedIdx(int*         window_mapped_idx,
                                       const int*   mapped_idx,
                                       const int*   window_idx,
                                       int          merge_unit,
                                       int          group_count,
                                       cudaStream_t stream)
{
    if (group_count == 0) {
        return;
    }

    const int total   = group_count * merge_unit;
    const int threads = 256;
    buildWindowMappedIdxKernel<<<cdiv(total, threads), threads, 0, stream>>>(
        window_mapped_idx, mapped_idx, window_idx, merge_unit, total);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
