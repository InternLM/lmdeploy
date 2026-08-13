// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/internvit/internvit_kernels.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/utils/cuda_utils.h"

#include "cub/block/block_reduce.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <type_traits>

namespace turbomind {

namespace {

struct SumPair {
    float s{};
    float sq{};

    __device__ SumPair operator+(const SumPair& other) const
    {
        return {s + other.s, sq + other.sq};
    }
};

template<typename T>
__global__ void patchIm2ColKernel(T*       out,
                                  const T* input,
                                  int      channels,
                                  int      image_h,
                                  int      image_w,
                                  int      patch_h,
                                  int      patch_w,
                                  int      grid_w,
                                  int      patch_area,
                                  int      patch_in_dim,
                                  int      num_patches)
{
    const int batch = blockIdx.x;
    const int patch = blockIdx.y;
    const int row   = batch * num_patches + patch;
    const int ph    = patch / grid_w;
    const int pw    = patch - ph * grid_w;

    for (int k = threadIdx.x; k < patch_in_dim; k += blockDim.x) {
        const int c   = k / patch_area;
        const int rem = k - c * patch_area;
        const int ih  = rem / patch_w;
        const int iw  = rem - ih * patch_w;

        const int64_t src =
            ((int64_t)batch * channels + c) * image_h * image_w + (ph * patch_h + ih) * image_w + pw * patch_w + iw;
        out[(int64_t)row * patch_in_dim + k] = input[src];
    }
}

template<typename T, int vec_size>
__global__ void addEmbeddingsVecKernel(T*       out,
                                       const T* patch,
                                       const T* patch_bias,
                                       const T* cls_token,
                                       const T* pos,
                                       int      seq_len,
                                       int      num_patches,
                                       int      tiles)
{
    const int vec_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int token  = blockIdx.y;
    const int batch  = blockIdx.z;
    if (vec_id >= tiles) {
        return;
    }

    Array<T, vec_size> pos_vec;
    Array<T, vec_size> out_vec;
    Array<T, vec_size> bias_vec{};
    Load(pos_vec, pos + (token * tiles + vec_id) * vec_size);

    if (token == 0) {
        Load(out_vec, cls_token + vec_id * vec_size);
    }
    else {
        Load(out_vec, patch + (((int64_t)batch * num_patches + token - 1) * tiles + vec_id) * vec_size);
        if (patch_bias) {
            Load(bias_vec, patch_bias + vec_id * vec_size);
        }
    }

    using namespace ops;
    Store(out + (((int64_t)batch * seq_len + token) * tiles + vec_id) * vec_size,
          cast<T>(cast<float>(out_vec) + cast<float>(bias_vec) + cast<float>(pos_vec)));
}

template<typename T, int vec_size, int block_dim>
__global__ void preRMSNormKernel(float* sums, const T* qkv, int token_num, int local_dim, int qkv_dim)
{
    const int token = blockIdx.x;
    const int part  = blockIdx.y;  // 0: q, 1: k
    const int base  = token * qkv_dim + part * local_dim;

    using namespace ops;
    float sum = 0.f;
    for (int d = threadIdx.x * vec_size; d < local_dim; d += block_dim * vec_size) {
        Array<T, vec_size> qkv_vec;
        Load(qkv_vec, qkv + base + d);

        const auto qkv_float = cast<float>(qkv_vec);
        const auto sq        = qkv_float * qkv_float;

        PRAGMA_UNROLL
        for (int i = 0; i < vec_size; ++i) {
            sum += sq[i];
        }
    }

    using BlockReduce = cub::BlockReduce<float, block_dim>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    sum = BlockReduce{temp_storage}.Sum(sum);

    if (threadIdx.x == 0) {
        sums[part * token_num + token] = sum;
    }
}

template<typename T, int vec_size>
__global__ void postRMSNormKernel(T*           qkv,
                                  const float* sums,
                                  const T*     q_weight,
                                  const T*     k_weight,
                                  int          token_num,
                                  int          local_dim,
                                  int          qkv_dim,
                                  int          hidden_dim,
                                  int          tiles,
                                  float        eps)
{
    const int token      = blockIdx.x;
    const int tile_block = blockIdx.y;
    const int part       = blockIdx.z;
    const int vec_id     = tile_block * blockDim.x + threadIdx.x;
    if (vec_id >= tiles) {
        return;
    }

    const int d    = vec_id * vec_size;
    const int base = token * qkv_dim + part * local_dim + d;

    Array<T, vec_size> qkv_vec;
    Array<T, vec_size> weight_vec;
    Load(qkv_vec, qkv + base);
    Ldg(weight_vec, (part == 0 ? q_weight : k_weight) + d);

    const float inv = rsqrtf(sums[part * token_num + token] / hidden_dim + eps);

    using namespace ops;
    Store(qkv + base, cast<T>((cast<float>(qkv_vec) * cast<float>(weight_vec)) * inv));
}

template<typename T, int head_dim, int vec_size, int warps_per_block>
__global__ void prepareQKVKernel(T* kv, const T* qkv, int token_num, int local_head_num)
{
    static_assert(head_dim % vec_size == 0);
    constexpr int kVecPerHead   = head_dim / vec_size;
    constexpr int kHeadsPerWarp = WARP_SIZE / kVecPerHead;
    static_assert(kVecPerHead * kHeadsPerWarp == WARP_SIZE);

    const int warp_id   = threadIdx.x / WARP_SIZE;
    const int lane_id   = threadIdx.x - warp_id * WARP_SIZE;
    const int head_slot = lane_id / kVecPerHead;
    const int vec_id    = lane_id - head_slot * kVecPerHead;

    const int token = blockIdx.x * warps_per_block + warp_id;
    if (token >= token_num) {
        return;
    }

    const int head_group = blockIdx.y;
    const int part       = blockIdx.z;
    const int head       = head_group * kHeadsPerWarp + head_slot;
    if (head >= local_head_num) {
        return;
    }

    const int     local_dim  = local_head_num * head_dim;
    const int     qkv_dim    = 3 * local_dim;
    const int     offset     = vec_id * vec_size;
    const int     src_offset = (part == 0 ? local_dim : 2 * local_dim) + head * head_dim + offset;
    const int64_t dst        = (((int64_t)head * 2 + part) * token_num + token) * head_dim + offset;

    Array<T, vec_size> qkv_vec;
    Load(qkv_vec, qkv + token * qkv_dim + src_offset);
    Store(kv + dst, qkv_vec);
}

// residual <- residual + (branch_output + optional branch_bias) * branch_scale
template<typename T, int vec_size>
__global__ void residualScaleKernel(T* __restrict__ residual,
                                    const T* __restrict__ branch_output,
                                    const T* __restrict__ branch_scale,
                                    const T* __restrict__ branch_bias,
                                    int hidden_dim,
                                    int tiles)
{
    const int token  = blockIdx.x;
    const int vec_id = blockIdx.y * blockDim.x + threadIdx.x;
    if (vec_id >= tiles) {
        return;
    }

    const int d = vec_id * vec_size;
    residual += (int64_t)token * hidden_dim + d;
    branch_output += (int64_t)token * hidden_dim + d;
    branch_scale += d;
    if (branch_bias) {
        branch_bias += d;
    }

    Array<T, vec_size> residual_vec;
    Array<T, vec_size> branch_vec;
    Array<T, vec_size> branch_scale_vec;
    Array<T, vec_size> branch_bias_vec{};

    Load(residual_vec, residual);
    Load(branch_vec, branch_output);
    Ldg(branch_scale_vec, branch_scale);
    if (branch_bias) {
        Ldg(branch_bias_vec, branch_bias);
    }

    using namespace ops;
    Store(
        residual,
        cast<T>(cast<float>(residual_vec) + cast<float>(branch_vec + branch_bias_vec) * cast<float>(branch_scale_vec)));
}

// residual <- residual + (branch_output + optional branch_bias) * branch_scale
// hidden_states <- LayerNorm(residual) * norm_weight + optional norm_bias
template<typename T, int vec_size, int block_dim>
__global__ void residualScaleLayerNormKernel(T*       hidden_states,
                                             T*       residual,
                                             const T* branch_output,
                                             const T* branch_scale,
                                             const T* branch_bias,
                                             const T* norm_weight,
                                             const T* norm_bias,
                                             int      hidden_dim,
                                             float    eps)
{
    const int token = blockIdx.x;
    const int di    = threadIdx.x * vec_size;

    residual += (int64_t)token * hidden_dim;
    branch_output += (int64_t)token * hidden_dim;
    hidden_states += (int64_t)token * hidden_dim;

    Array<float, vec_size> sum_v{};
    Array<float, vec_size> sq_v{};
    Array<T, vec_size>     residual_vec;
    Array<T, vec_size>     branch_vec;
    Array<T, vec_size>     branch_scale_vec;
    Array<T, vec_size>     branch_bias_vec{};

    using namespace ops;
    for (int i = di; i < hidden_dim; i += block_dim * vec_size) {
        Load(residual_vec, residual + i);
        Load(branch_vec, branch_output + i);
        Ldg(branch_scale_vec, branch_scale + i);
        if (branch_bias) {
            Ldg(branch_bias_vec, branch_bias + i);
        }

        residual_vec = cast<T>(cast<float>(residual_vec)
                               + cast<float>(branch_vec + branch_bias_vec) * cast<float>(branch_scale_vec));
        Store(residual + i, residual_vec);

        const auto residual_float = cast<float>(residual_vec);
        sum_v                     = sum_v + residual_float;
        sq_v                      = sq_v + residual_float * residual_float;
    }

    SumPair pair{};
    PRAGMA_UNROLL
    for (int i = 0; i < vec_size; ++i) {
        pair.s += sum_v[i];
        pair.sq += sq_v[i];
    }

    using BlockReduce = cub::BlockReduce<SumPair, block_dim>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    pair = BlockReduce{temp_storage}.Sum(pair);

    __shared__ float shared_mean;
    __shared__ float shared_inv_std;

    if (threadIdx.x == 0) {
        const float inv_dim = 1.f / hidden_dim;
        const float mean    = pair.s * inv_dim;
        const float var     = fmaxf(pair.sq * inv_dim - mean * mean, 0.f);
        shared_mean         = mean;
        shared_inv_std      = rsqrtf(var + eps);
    }

    __syncthreads();

    const float mean    = shared_mean;
    const float inv_std = shared_inv_std;

    Array<T, vec_size> weight_vec;
    Array<T, vec_size> bias_vec{};
    for (int i = di; i < hidden_dim; i += block_dim * vec_size) {
        Load(residual_vec, residual + i);
        Ldg(weight_vec, norm_weight + i);
        if (norm_bias) {
            Ldg(bias_vec, norm_bias + i);
        }

        Array<float, vec_size> out_vec;
        PRAGMA_UNROLL
        for (int j = 0; j < vec_size; ++j) {
            out_vec[j] = (static_cast<float>(residual_vec[j]) - mean) * inv_std * static_cast<float>(weight_vec[j])
                         + static_cast<float>(bias_vec[j]);
        }
        Store(hidden_states + i, cast<T>(out_vec));
    }
}

// residual <- residual + (branch_output + optional branch_bias) * branch_scale
// hidden_states <- RMSNorm(residual) * norm_weight
template<typename T, int vec_size, int block_dim>
__global__ void residualScaleRMSNormKernel(T*       hidden_states,
                                           T*       residual,
                                           const T* branch_output,
                                           const T* branch_scale,
                                           const T* branch_bias,
                                           const T* norm_weight,
                                           int      hidden_dim,
                                           float    eps)
{
    const int token = blockIdx.x;
    const int di    = threadIdx.x * vec_size;

    residual += (int64_t)token * hidden_dim;
    branch_output += (int64_t)token * hidden_dim;
    hidden_states += (int64_t)token * hidden_dim;

    Array<float, vec_size> sq_v{};
    Array<T, vec_size>     residual_vec;
    Array<T, vec_size>     branch_vec;
    Array<T, vec_size>     branch_scale_vec;
    Array<T, vec_size>     branch_bias_vec{};

    using namespace ops;
    for (int i = di; i < hidden_dim; i += block_dim * vec_size) {
        Load(residual_vec, residual + i);
        Load(branch_vec, branch_output + i);
        Ldg(branch_scale_vec, branch_scale + i);
        if (branch_bias) {
            Ldg(branch_bias_vec, branch_bias + i);
        }

        residual_vec = cast<T>(cast<float>(residual_vec)
                               + cast<float>(branch_vec + branch_bias_vec) * cast<float>(branch_scale_vec));
        Store(residual + i, residual_vec);

        const auto residual_float = cast<float>(residual_vec);
        sq_v                      = sq_v + residual_float * residual_float;
    }

    float sum{};
    PRAGMA_UNROLL
    for (int i = 0; i < vec_size; ++i) {
        sum += sq_v[i];
    }

    using BlockReduce = cub::BlockReduce<float, block_dim>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    sum = BlockReduce{temp_storage}.Sum(sum);

    __shared__ float shared_inv_rms;
    if (threadIdx.x == 0) {
        shared_inv_rms = rsqrtf(sum / hidden_dim + eps);
    }

    __syncthreads();

    const float inv_rms = shared_inv_rms;

    using namespace ops;
    Array<T, vec_size> weight_vec;
    for (int i = di; i < hidden_dim; i += block_dim * vec_size) {
        Load(residual_vec, residual + i);
        Ldg(weight_vec, norm_weight + i);
        Store(hidden_states + i, cast<T>(cast<float>(residual_vec) * inv_rms * cast<float>(weight_vec)));
    }
}

template<typename T, int vec_size>
__global__ void pixelShuffleKernel(T* __restrict__ out,
                                   const T* __restrict__ hidden,
                                   int grid_size,
                                   int out_grid,
                                   int hidden_dim,
                                   int seq_len,
                                   int tiles)
{
    const int vec_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_id >= tiles) {
        return;
    }

    const int token = blockIdx.y;
    const int batch = blockIdx.z;
    const int ow    = token / out_grid;
    const int oh    = token - ow * out_grid;

    const int     c        = vec_id * vec_size;
    const int     in_token = 1 + (ow * 2) * grid_size + oh * 2;
    const int64_t in0      = ((int64_t)batch * seq_len + in_token) * hidden_dim + c;
    const int64_t out0     = ((int64_t)batch * out_grid * out_grid + token) * (hidden_dim * 4) + c;

    Array<T, vec_size> v;
    Load(v, hidden + in0);
    Store(out + out0, v);
    Load(v, hidden + in0 + hidden_dim);
    Store(out + out0 + hidden_dim, v);
    Load(v, hidden + in0 + (int64_t)grid_size * hidden_dim);
    Store(out + out0 + 2 * hidden_dim, v);
    Load(v, hidden + in0 + ((int64_t)grid_size + 1) * hidden_dim);
    Store(out + out0 + 3 * hidden_dim, v);
}

}  // namespace

void invokeInternVitPatchify(Tensor&       patches,
                             const Tensor& pixel_values,
                             int           batch_size,
                             int           channels,
                             int           image_h,
                             int           image_w,
                             int           patch_h,
                             int           patch_w,
                             cudaStream_t  stream)
{
    TM_CHECK_EQ(patches.ndim(), 2);
    TM_CHECK_EQ(pixel_values.ndim(), 4);
    TM_CHECK_EQ(patches.dtype(), pixel_values.dtype());

    const int grid_h       = image_h / patch_h;
    const int grid_w       = image_w / patch_w;
    const int num_patches  = grid_h * grid_w;
    const int patch_area   = patch_h * patch_w;
    const int patch_in_dim = channels * patch_area;

    auto invoke = [&](auto t) {
        using T = decltype(t);
        const dim3 grid(batch_size, num_patches);
        const dim3 block(256);
        patchIm2ColKernel<T><<<grid, block, 0, stream>>>((T*)patches.raw_data(),
                                                         (const T*)pixel_values.raw_data(),
                                                         channels,
                                                         image_h,
                                                         image_w,
                                                         patch_h,
                                                         patch_w,
                                                         grid_w,
                                                         patch_area,
                                                         patch_in_dim,
                                                         num_patches);
    };
    TM_DISPATCH_PRIMARY_DTYPES(patches.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeInternVitAddEmbeddings(Tensor&       hidden,
                                  const Tensor& patch_embeds,
                                  const Tensor& patch_bias,
                                  const Tensor& cls_token,
                                  const Tensor& position_embeddings,
                                  int           batch_size,
                                  int           num_patches,
                                  int           hidden_dim,
                                  cudaStream_t  stream)
{
    constexpr int Vec      = 4;
    constexpr int kThreads = 256;
    const int     seq_len  = num_patches + 1;
    const int     tiles    = hidden_dim / Vec;
    const dim3    grid((tiles + kThreads - 1) / kThreads, seq_len, batch_size);

    TM_CHECK_EQ(hidden_dim % Vec, 0);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        addEmbeddingsVecKernel<T, Vec>
            <<<grid, kThreads, 0, stream>>>((T*)hidden.raw_data(),
                                            (const T*)patch_embeds.raw_data(),
                                            patch_bias ? (const T*)patch_bias.raw_data() : nullptr,
                                            (const T*)cls_token.raw_data(),
                                            (const T*)position_embeddings.raw_data(),
                                            seq_len,
                                            num_patches,
                                            tiles);
    };
    TM_DISPATCH_PRIMARY_DTYPES(hidden.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeInternVitPreRMSNorm(Tensor& sums, const Tensor& qkv, int local_dim, cudaStream_t stream)
{
    TM_CHECK_EQ(sums.dtype(), kFloat);
    constexpr int kThreads  = 64;
    const int     token_num = qkv.shape(0);
    const int     qkv_dim   = qkv.shape(1);

    auto invoke = [&](auto t) {
        using T                = decltype(t);
        constexpr int kVecSize = sizeof(uint4) / sizeof(T);
        TM_CHECK_EQ(local_dim % kVecSize, 0);
        preRMSNormKernel<T, kVecSize, kThreads><<<dim3(token_num, 2), kThreads, 0, stream>>>(
            (float*)sums.raw_data(), (const T*)qkv.raw_data(), token_num, local_dim, qkv_dim);
    };
    TM_DISPATCH_PRIMARY_DTYPES(qkv.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeInternVitPostRMSNorm(Tensor&       qkv,
                                const Tensor& sums,
                                const Tensor& q_weight,
                                const Tensor& k_weight,
                                int           local_dim,
                                int           hidden_dim,
                                float         eps,
                                cudaStream_t  stream)
{
    constexpr int kThreads  = 256;
    const int     token_num = qkv.shape(0);
    const int     qkv_dim   = qkv.shape(1);

    TM_CHECK_EQ(q_weight.size(), local_dim);
    TM_CHECK_EQ(k_weight.size(), local_dim);

    auto invoke = [&](auto t) {
        using T                = decltype(t);
        constexpr int kVecSize = sizeof(uint4) / sizeof(T);
        TM_CHECK_EQ(local_dim % kVecSize, 0);
        const int  tiles       = local_dim / kVecSize;
        const int  tile_blocks = (tiles + kThreads - 1) / kThreads;
        const dim3 grid(token_num, tile_blocks, 2);
        postRMSNormKernel<T, kVecSize><<<grid, kThreads, 0, stream>>>((T*)qkv.raw_data(),
                                                                      (const float*)sums.raw_data(),
                                                                      (const T*)q_weight.raw_data(),
                                                                      (const T*)k_weight.raw_data(),
                                                                      token_num,
                                                                      local_dim,
                                                                      qkv_dim,
                                                                      hidden_dim,
                                                                      tiles,
                                                                      eps);
    };
    TM_DISPATCH_PRIMARY_DTYPES(qkv.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeInternVitPrepareQKV(Tensor& kv, const Tensor& qkv, int local_head_num, int head_dim, cudaStream_t stream)
{
    auto invoke = [&](auto t) {
        using T       = decltype(t);
        auto dispatch = [&](auto head_dim_c) {
            constexpr int kHeadDim       = decltype(head_dim_c)::value;
            constexpr int kVecSize       = sizeof(uint4) / sizeof(T);
            constexpr int kVecPerHead    = kHeadDim / kVecSize;
            constexpr int kHeadsPerWarp  = WARP_SIZE / kVecPerHead;
            constexpr int kWarpsPerBlock = 4;
            static_assert(kVecPerHead * kHeadsPerWarp == WARP_SIZE);

            const int  token_num      = qkv.shape(0);
            const int  head_group_num = (local_head_num + kHeadsPerWarp - 1) / kHeadsPerWarp;
            const dim3 grid((token_num + kWarpsPerBlock - 1) / kWarpsPerBlock, head_group_num, 2);
            const dim3 block(kWarpsPerBlock * WARP_SIZE);
            prepareQKVKernel<T, kHeadDim, kVecSize, kWarpsPerBlock>
                <<<grid, block, 0, stream>>>((T*)kv.raw_data(), (const T*)qkv.raw_data(), token_num, local_head_num);
        };

        if (head_dim == 64) {
            dispatch(std::integral_constant<int, 64>{});
        }
        else if (head_dim == 128) {
            dispatch(std::integral_constant<int, 128>{});
        }
        else {
            TM_LOG_FATAL("unsupported InternVit PrepareQKV head_dim: {}", head_dim);
        }
    };
    TM_DISPATCH_PRIMARY_DTYPES(qkv.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeInternVitResidualScaleNorm(Tensor&       hidden_states,
                                      Tensor&       residual,
                                      const Tensor& branch_output,
                                      const Tensor& branch_scale,
                                      const Tensor& branch_bias,
                                      const Tensor& norm_weight,
                                      const Tensor& norm_bias,
                                      float         eps,
                                      NormType      norm_type,
                                      cudaStream_t  stream)
{
    TM_CHECK_EQ(residual.ndim(), 2);
    TM_CHECK_EQ(branch_output.size(), residual.size());
    TM_CHECK_EQ(branch_output.dtype(), residual.dtype());

    const int hidden_dim = residual.shape(1);
    const int token_num  = residual.shape(0);

    TM_CHECK_EQ(branch_scale.size(), hidden_dim);

    if (norm_type != NormType::kNone) {
        TM_CHECK(norm_weight);
        TM_CHECK_EQ(norm_weight.size(), hidden_dim);
    }

    auto invoke = [&](auto t) {
        using T                    = decltype(t);
        constexpr int kNormThreads = 512;
        constexpr int kVecSize     = sizeof(uint4) / sizeof(T);
        TM_CHECK_EQ(hidden_dim % kVecSize, 0);

        if (norm_type == NormType::kNone) {
            const int  kThreads = hidden_dim <= 1024 ? 128 : 256;
            const int  tiles    = hidden_dim / kVecSize;
            const dim3 grid(token_num, (tiles + kThreads - 1) / kThreads);
            residualScaleKernel<T, kVecSize>
                <<<grid, kThreads, 0, stream>>>((T*)residual.raw_data(),
                                                (const T*)branch_output.raw_data(),
                                                (const T*)branch_scale.raw_data(),
                                                branch_bias ? (const T*)branch_bias.raw_data() : nullptr,
                                                hidden_dim,
                                                tiles);
        }
        else if (norm_type == NormType::kLayerNorm) {
            residualScaleLayerNormKernel<T, kVecSize, kNormThreads>
                <<<token_num, kNormThreads, 0, stream>>>((T*)hidden_states.raw_data(),
                                                         (T*)residual.raw_data(),
                                                         (const T*)branch_output.raw_data(),
                                                         (const T*)branch_scale.raw_data(),
                                                         branch_bias ? (const T*)branch_bias.raw_data() : nullptr,
                                                         (const T*)norm_weight.raw_data(),
                                                         norm_bias ? (const T*)norm_bias.raw_data() : nullptr,
                                                         hidden_dim,
                                                         eps);
        }
        else if (norm_type == NormType::kRMSNorm) {
            residualScaleRMSNormKernel<T, kVecSize, kNormThreads>
                <<<token_num, kNormThreads, 0, stream>>>((T*)hidden_states.raw_data(),
                                                         (T*)residual.raw_data(),
                                                         (const T*)branch_output.raw_data(),
                                                         (const T*)branch_scale.raw_data(),
                                                         branch_bias ? (const T*)branch_bias.raw_data() : nullptr,
                                                         (const T*)norm_weight.raw_data(),
                                                         hidden_dim,
                                                         eps);
        }
        else {
            TM_LOG_FATAL("unsupported InternVit residual norm type: {}", (int)norm_type);
        }
    };
    TM_DISPATCH_PRIMARY_DTYPES(residual.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeInternVitPixelShuffle(Tensor& output, const Tensor& hidden, int grid_size, cudaStream_t stream)
{
    // kVecSize=4 was faster than 16-byte vectors in pixel-shuffle benchmarks.
    constexpr int kVecSize = 4;
    constexpr int kThreads = 128;

    const int out_grid   = grid_size / 2;
    const int seq_len    = grid_size * grid_size + 1;
    const int hidden_dim = hidden.shape(1);
    const int batch_size = output.shape(0) / (out_grid * out_grid);
    const int tiles      = hidden_dim / kVecSize;

    auto invoke = [&](auto t) {
        using T = decltype(t);
        const dim3 grid((tiles + kThreads - 1) / kThreads, out_grid * out_grid, batch_size);
        pixelShuffleKernel<T, kVecSize><<<grid, kThreads, 0, stream>>>(
            (T*)output.raw_data(), (const T*)hidden.raw_data(), grid_size, out_grid, hidden_dim, seq_len, tiles);
    };
    TM_DISPATCH_PRIMARY_DTYPES(output.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
