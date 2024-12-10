// Copyright (c) OpenMMLab. All rights reserved.
#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/rotary_emb.h"
#include <cub/device/device_scan.cuh>
#include <map>

namespace turbomind {

__inline__ __device__ float compute_default_parameters(float base, float dim, int di, float factor)
{
    float scale_factor = -log2f(base) / dim;
    float inv_freq     = exp2f(di * scale_factor) * factor;
    return inv_freq;
}

struct CosSinDefault {
    __device__ float2 operator()(float base, float dim, float di, float factor, float p)
    {
        float c, s;
        float inv_freq = compute_default_parameters(base, dim, di, factor);
        sincosf(p * inv_freq, &s, &c);
        return {c, s};
    }
};

struct CosSinLlama3 {
    InnerLlama3RopeParam param_;
    CosSinLlama3(InnerLlama3RopeParam param): param_(param) {}
    __device__ float2 operator()(float base, float dim, float di, float factor, float p)
    {
        float c, s;
        float inv_freq = compute_default_parameters(base, dim, di, factor);
        auto  smooth   = fmaxf(0.f, fminf(1.f, param_.alpha * inv_freq - param_.beta));
        inv_freq       = (1 - smooth) * inv_freq * param_.inv_scaling_factor + smooth * inv_freq;
        sincosf(p * inv_freq, &s, &c);
        return {c, s};
    }
};

struct CosSinYarn {
    InnerYarnRopeParam param_;
    CosSinYarn(InnerYarnRopeParam param): param_(param) {}
    __device__ float2 operator()(float base, float dim, float di, float factor, float p)
    {
        float c, s;
        float inv_freq = compute_default_parameters(base, dim, di, factor);
        float alpha    = 2 * di * param_.ramp_inv_factor_div_2 - param_.ramp_inv_factor_mul_min;
        alpha          = fmaxf(0.f, fminf(1.f, alpha));
        inv_freq       = inv_freq - inv_freq * alpha * param_.inv_scaling_factor;
        sincosf(p * inv_freq, &s, &c);
        c *= param_.attention_factor;
        s *= param_.attention_factor;
        return {c, s};
    }
};

template<typename T, int items_per_thread, class Func>
__global__ void rotaryEmbedding(const float rope_base, int token_num, int dim, float factor, Func func, T* cos_sin)
{
    int thread_id      = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_per_tok = dim / items_per_thread;
    int pi             = thread_id / thread_per_tok;
    int di             = thread_id % thread_per_tok * items_per_thread;
    if (pi >= token_num || di >= dim) {
        return;
    }

    Array<T, items_per_thread> cs;
    for (int i = 0; i < items_per_thread; i += 2) {
        float2 v  = func(rope_base, dim, di + i, factor, pi);
        cs[i]     = (T)v.x;
        cs[i + 1] = (T)v.y;
    }
    Store(&cos_sin[dim * pi + di], cs);
}

__global__ void computeQ2B(int* q2b, int* q_len, int token_num, int batch_size)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < token_num; i += blockDim.x * gridDim.x) {
        q2b[i] = 0;
    }
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    if (b < batch_size) {
        q2b[q_len[b]] = b;
    }
}

template<int iterms_per_thread>
__global__ void computeQ2P(const int* q2b, const int* q_len, const int* k_len, int token_num, int* q2p)
{
    Array<int, iterms_per_thread> p;

    size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    size_t index     = thread_id * iterms_per_thread;
    for (int i = 0; i < iterms_per_thread; ++i) {
        int qi = index + i;
        if (qi < token_num) {
            int bid         = q2b[qi] + 1;
            int history_len = (k_len[bid] - k_len[bid - 1]) - (q_len[bid] - q_len[bid - 1]);
            int pi          = history_len + qi - q_len[bid - 1];
            p[i]            = pi;
        }
    }
    if (index < token_num) {
        Store(&q2p[index], p);
    }
}

template<typename T, int items_per_thread>
__global__ void rotaryEmbeddingDynamic(
    const int* q2b, const int* q2p, const float* rope_base, int token_num, int dim, float factor, T* cos_sin)
{
    int thread_id      = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_per_tok = dim / items_per_thread;
    int qi             = thread_id / thread_per_tok;
    int di             = thread_id % thread_per_tok * items_per_thread;
    if (qi >= token_num || di >= dim) {
        return;
    }

    int   bid  = q2b[qi] + 1;
    float base = rope_base[bid - 1];
    float ti   = (float)q2p[qi];

    Array<T, items_per_thread> cs;
    float                      c, s;
    for (int i = 0; i < items_per_thread; i += 2) {
        float inv_freq = compute_default_parameters(base, dim, di + i, factor);
        sincosf(ti * inv_freq, &s, &c);
        cs[i]     = (T)c;
        cs[i + 1] = (T)s;
    }
    Store(&cos_sin[dim * qi + di], cs);
}

RopeType GetRoPEType(const std::string& type)
{
    std::map<std::string, RopeType> lookup = {{"", RopeType::kDefault},
                                              {"linear", RopeType::kLinear},
                                              {"dynamic", RopeType::kDynamic},
                                              {"yarn", RopeType::kYarn},
                                              {"llama3", RopeType::kLlama3}};
    return lookup.at(type);
}

template<typename T>
void RotaryEmbeddingV2<T>::freeBuffer()
{
    allocator_->free((void**)&cos_sin_);
    allocator_->free((void**)&q2b_);
    allocator_->free((void**)&q2p_);
    allocator_->free((void**)&d_temp_storage_);
    cached_len_         = 0;
    temp_storage_bytes_ = 0;
}

template<typename T>
void RotaryEmbeddingV2<T>::allocateBuffer(int token_num)
{
    int token_num_padded = round_up(token_num, 4);
    q2p_                 = (int*)allocator_->reMalloc(q2p_, sizeof(int) * token_num_padded);
    q2b_                 = (int*)allocator_->reMalloc(q2b_, sizeof(int) * token_num);
    cub::DeviceScan::InclusiveScan(
        nullptr, temp_storage_bytes_, (int*)nullptr, (int*)nullptr, cub::Max{}, token_num, stream_);
    d_temp_storage_ = allocator_->reMalloc(d_temp_storage_, temp_storage_bytes_);
}

template<typename T>
RotaryEmbeddingV2<T>::RotaryEmbeddingV2(const AttentionParam& param,
                                        int                   session_len,
                                        cudaStream_t          stream,
                                        IAllocator*           allocator):
    stream_(stream), allocator_(allocator)
{
    type_      = param.rope.type;
    dim_       = param.rope.dim;
    rope_base_ = param.rope.base;

    switch (type_) {
        case RopeType::kDefault:
        case RopeType::kDynamic:
            break;
        case RopeType::kLinear:
            inv_factor_ = 1.0f / param.rope.factor;
            break;
        case RopeType::kYarn: {
            const double PI                  = 3.14159265358979323846;
            auto         find_correction_dim = [&](float num_rotations) {
                return (param.rope.dim * std::log(param.rope.max_position_embeddings / (num_rotations * 2 * PI)))
                       / (2 * std::log(param.rope.base));
            };
            auto find_correction_range = [&](float low_rot, float high_rot, float& low, float& high) {
                low  = std::floor(find_correction_dim(low_rot));
                high = std::ceil(find_correction_dim(high_rot));
                low  = std::max(low, 0.f);
                high = std::min(high, param.rope.dim - 1.f);
            };
            float low, high;
            find_correction_range(param.rope.yarn.beta_fast, param.rope.yarn.beta_slow, low, high);
            if (low == high) {
                high += 0.01f;
            }
            yarn_.ramp_inv_factor_div_2   = 1.0 / (high - low) / 2.0;
            yarn_.ramp_inv_factor_mul_min = 1.0 / (high - low) * low;
            yarn_.inv_scaling_factor      = (1 - 1.0 / param.rope.factor);
            yarn_.attention_factor        = param.rope.yarn.attention_factor;
            break;
        }
        case RopeType::kLlama3: {
            // clang-format off
            /* The [llama3 rope](https://github.com/huggingface/transformers/blob/5f4ee98a7ade33e1c54fdd6181d04ee7b426b392/src/transformers/modeling_rope_utils.py#L298)
            * used by llama3.1 equals to the following equation, given the precommuted parameters as:
            ```C++
            inv_scaling_factor = 1 / factor;
            inv_diff_freq_factor = 1 / (high_freq_factor - low_freq_factor);
            alpha = old_context_len / (2 * PI) * inv_diff_freq_factor;
            beta = low_freq_factor * inv_diff_freq_factor
            ```
            */
            // clang-format on
            const double PI            = 3.14159265358979323846;
            float inv_diff_freq_factor = 1.0 / (param.rope.llama3.high_freq_factor - param.rope.llama3.low_freq_factor);
            llama3_.inv_scaling_factor = 1.0 / param.rope.factor;
            llama3_.alpha = param.rope.llama3.original_max_position_embeddings / (2 * PI) * inv_diff_freq_factor;
            llama3_.beta  = param.rope.llama3.low_freq_factor * inv_diff_freq_factor;
            break;
        }
        default:
            FT_CHECK(0);
            break;
    }

    cos_sin_ = (T*)allocator_->reMalloc(cos_sin_, sizeof(T) * session_len * dim_);
    allocateBuffer(session_len);
    computeCache(session_len);
}

template<typename T>
void RotaryEmbeddingV2<T>::computeCache(int session_len)
{
    const int items_per_thread = 8;
    const int block            = 256;
    const int grid             = (dim_ / items_per_thread * session_len + block - 1) / block;

    switch (type_) {
        case RopeType::kDefault:
        case RopeType::kLinear:
        case RopeType::kDynamic:
            rotaryEmbedding<T, items_per_thread>
                <<<grid, block, 0, stream_>>>(rope_base_, session_len, dim_, inv_factor_, CosSinDefault{}, cos_sin_);
            break;
        case RopeType::kLlama3:
            rotaryEmbedding<T, items_per_thread><<<grid, block, 0, stream_>>>(
                rope_base_, session_len, dim_, inv_factor_, CosSinLlama3{llama3_}, cos_sin_);
            break;
        case RopeType::kYarn:
            rotaryEmbedding<T, items_per_thread>
                <<<grid, block, 0, stream_>>>(rope_base_, session_len, dim_, inv_factor_, CosSinYarn{yarn_}, cos_sin_);
            break;
        default:
            FT_CHECK(0);
    }

    cached_len_ = session_len;
}

template<typename T>
void RotaryEmbeddingV2<T>::updateCache(const RotaryEmbeddingV2Param& params)
{
    if (type_ == RopeType::kDynamic) {
        cos_sin_                   = (T*)allocator_->reMalloc(cos_sin_, sizeof(T) * params.token_num * dim_);
        const int items_per_thread = 8;
        const int block            = 256;
        const int grid             = (dim_ / items_per_thread * params.token_num + block - 1) / block;
        rotaryEmbeddingDynamic<T, items_per_thread>
            <<<grid, block, 0, stream_>>>(q2b_, q2p_, params.rope_theta, params.token_num, dim_, inv_factor_, cos_sin_);
    }
    else {
        int sess_len = 0;
        for (int i = 0; i < params.batch_size; ++i) {
            sess_len = std::max(sess_len, (params.h_k_len[i + 1] - params.h_k_len[i]));
        }
        if (sess_len > cached_len_) {
            cos_sin_ = (T*)allocator_->reMalloc(cos_sin_, sizeof(T) * sess_len * dim_);
            computeCache(sess_len);
        }
    }
}

template<typename T>
void RotaryEmbeddingV2<T>::updateMapping(const RotaryEmbeddingV2Param& params)
{
    // q2b
    {
        const size_t block = 512;
        const size_t grid  = (params.token_num + block - 1) / block;
        FT_CHECK(block * grid >= (size_t)params.batch_size);
        computeQ2B<<<grid, block, 0, stream_>>>(q2b_, params.q_len, params.token_num, params.batch_size);
        if (params.dc_size != params.batch_size) {
            cub::DeviceScan::InclusiveScan(
                d_temp_storage_, temp_storage_bytes_, q2b_, q2b_, cub::Max{}, params.token_num, stream_);
        }
    }

    // q2p
    {
        const int    iterms_per_thread = 4;
        const size_t block             = 256;
        const int    tokens_per_block  = block * iterms_per_thread;
        const size_t grid              = (params.token_num + tokens_per_block - 1) / tokens_per_block;
        computeQ2P<iterms_per_thread>
            <<<grid, block, 0, stream_>>>(q2b_, params.q_len, params.k_len, params.token_num, q2p_);
    }
}

template<typename T>
void RotaryEmbeddingV2<T>::forward(const RotaryEmbeddingV2Param& params)
{
    allocateBuffer(params.token_num);
    updateMapping(params);
    updateCache(params);
}

#ifdef ENABLE_FP32
template class RotaryEmbeddingV2<float>;
#endif
template class RotaryEmbeddingV2<half>;
#ifdef ENABLE_BF16
template class RotaryEmbeddingV2<__nv_bfloat16>;
#endif  // ENABLE_BF16

}  // namespace turbomind
