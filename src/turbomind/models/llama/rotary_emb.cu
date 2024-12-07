// Copyright (c) OpenMMLab. All rights reserved.
#include "src/turbomind/models/llama/rotary_emb.h"
#include <map>

namespace turbomind {

__device__ int get_batch_id(int qi, int* q_len, int batch_size)
{
    int result{};
    int end = (batch_size + blockDim.x - 1) / blockDim.x * blockDim.x;
    for (int i = threadIdx.x; i < end; i += blockDim.x) {
        int  prefix_sum = (i < batch_size) ? q_len[i + 1] : q_len[batch_size];
        auto count      = __syncthreads_count(prefix_sum > qi);
        if (count != 0) {
            result = i / blockDim.x * blockDim.x + blockDim.x - count + 1;
            break;
        }
    }
    return result;
}

__inline__ __device__ float compute_default_parameters(float base, float dim, int di, float factor)
{
    float scale_factor = -log2f(base) / dim;
    float inv_freq     = exp2f(di * scale_factor) * factor;
    return inv_freq;
}

template<typename T>
__global__ void rotaryEmbeddingDefault(
    const float* rope_base, int* q_len, int* k_len, int token_num, int batch_size, int dim, float factor, T* cos_sin)
{
    int qi = blockIdx.x;
    int di = threadIdx.x;

    int   bid         = get_batch_id(qi, q_len, batch_size);
    int   history_len = (k_len[bid] - k_len[bid - 1]) - (q_len[bid] - q_len[bid - 1]);
    float base        = rope_base[bid - 1];
    float ti          = history_len + qi - q_len[bid - 1];

    float inv_freq = compute_default_parameters(base, dim, di * 2, factor);
    float c, s;
    sincosf(ti * inv_freq, &s, &c);
    cos_sin[dim * qi + 2 * di]     = (T)c;
    cos_sin[dim * qi + 2 * di + 1] = (T)s;
}

template<typename T>
__global__ void rotaryEmbeddingLlama3(const float* rope_base,
                                      int*         q_len,
                                      int*         k_len,
                                      int          token_num,
                                      int          batch_size,
                                      int          dim,
                                      float        inv_scaling_factor,
                                      float        alpha,
                                      float        beta,
                                      T*           cos_sin)
{
    int qi = blockIdx.x;
    int di = threadIdx.x;

    int   bid         = get_batch_id(qi, q_len, batch_size);
    int   history_len = (k_len[bid] - k_len[bid - 1]) - (q_len[bid] - q_len[bid - 1]);
    float base        = rope_base[bid - 1];
    float ti          = history_len + qi - q_len[bid - 1];

    float inv_freq = compute_default_parameters(base, dim, di * 2, 1.0f);
    auto  smooth   = fmaxf(0.f, fminf(1.f, alpha * inv_freq - beta));
    inv_freq       = (1 - smooth) * inv_freq * inv_scaling_factor + smooth * inv_freq;
    float c, s;
    sincosf(ti * inv_freq, &s, &c);
    cos_sin[dim * qi + 2 * di]     = (T)c;
    cos_sin[dim * qi + 2 * di + 1] = (T)s;
}

template<typename T>
__global__ void rotaryEmbeddingYarn(const float* rope_base,
                                    int*         q_len,
                                    int*         k_len,
                                    int          token_num,
                                    int          batch_size,
                                    int          dim,
                                    float        ramp_inv_factor_div_2,
                                    float        ramp_inv_factor_mul_min,
                                    float        inv_scaling_factor,
                                    float        attention_scaling,
                                    T*           cos_sin)
{
    int qi = blockIdx.x;
    int di = threadIdx.x;

    int   bid         = get_batch_id(qi, q_len, batch_size);
    int   history_len = (k_len[bid] - k_len[bid - 1]) - (q_len[bid] - q_len[bid - 1]);
    float base        = rope_base[bid - 1];
    float ti          = history_len + qi - q_len[bid - 1];

    float inv_freq = compute_default_parameters(base, dim, di * 2, 1.0f);
    float alpha    = 2 * di * ramp_inv_factor_div_2 - ramp_inv_factor_mul_min;
    alpha          = fmaxf(0.f, fminf(1.f, alpha));
    inv_freq       = inv_freq - inv_freq * alpha * inv_scaling_factor;

    float c, s;
    sincosf(ti * inv_freq, &s, &c);
    c *= attention_scaling;
    s *= attention_scaling;
    cos_sin[dim * qi + 2 * di]     = (T)c;
    cos_sin[dim * qi + 2 * di + 1] = (T)s;
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
}

template<typename T>
void RotaryEmbeddingV2<T>::allocateBuffer(size_t token_num)
{
    cos_sin_ = (T*)allocator_->reMalloc(cos_sin_, sizeof(T) * token_num * dim_);
}

template<typename T>
RotaryEmbeddingV2<T>::RotaryEmbeddingV2(const AttentionParam& param, cudaStream_t stream, IAllocator* allocator):
    stream_(stream), allocator_(allocator)
{
    type_ = param.rope.type;
    dim_  = param.rope.dim;

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
}

template<typename T>
void RotaryEmbeddingV2<T>::forward(const RotaryEmbeddingV2Param& params)
{
    allocateBuffer(params.token_num);

    const int grid  = params.token_num;
    const int block = dim_ / 2;

    switch (type_) {
        case RopeType::kDefault:
        case RopeType::kLinear:
        case RopeType::kDynamic:
            rotaryEmbeddingDefault<<<grid, block, 0, stream_>>>(params.rope_theta,
                                                                params.q_len,
                                                                params.k_ken,
                                                                params.token_num,
                                                                params.batch_size,
                                                                dim_,
                                                                inv_factor_,
                                                                cos_sin_);
            break;
        case RopeType::kLlama3:
            rotaryEmbeddingLlama3<<<grid, block, 0, stream_>>>(params.rope_theta,
                                                               params.q_len,
                                                               params.k_ken,
                                                               params.token_num,
                                                               params.batch_size,
                                                               dim_,
                                                               llama3_.inv_scaling_factor,
                                                               llama3_.alpha,
                                                               llama3_.beta,
                                                               cos_sin_);
            break;
        case RopeType::kYarn:
            rotaryEmbeddingYarn<<<grid, block, 0, stream_>>>(params.rope_theta,
                                                             params.q_len,
                                                             params.k_ken,
                                                             params.token_num,
                                                             params.batch_size,
                                                             dim_,
                                                             yarn_.ramp_inv_factor_div_2,
                                                             yarn_.ramp_inv_factor_mul_min,
                                                             yarn_.inv_scaling_factor,
                                                             yarn_.attention_factor,
                                                             cos_sin_);
            break;
        default:
            FT_CHECK(0);
    }
}
#ifdef ENABLE_FP32
template class RotaryEmbeddingV2<float>;
#endif
template class RotaryEmbeddingV2<half>;
#ifdef ENABLE_BF16
template class RotaryEmbeddingV2<__nv_bfloat16>;
#endif  // ENABLE_BF16

}  // namespace turbomind
