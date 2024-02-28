// Copyright (c) OpenMMLab. All rights reserved.

#include "test_utils.h"
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>

#define _CG_ABI_EXPERIMENTAL
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include "src/turbomind/kernels/decoder_masked_multihead_attention.h"

namespace turbomind {

cublasHandle_t cublas_handle{};
cudaStream_t   cublas_stream{};

template<typename T>
void Compare(const T* src, const T* ref, size_t stride, int m, int n, bool show, float rtol, float atol)
{
    float asums{};
    float rsums{};
    int   outliers{};
    for (int nn = 0; nn < n; ++nn) {
        float abs_diff_sum{};
        float rel_diff_sum{};
        for (int mm = 0; mm < m; ++mm) {
            auto x = float(src[nn * stride + mm]);
            auto y = float(ref[nn * stride + mm]);
            // if (show) {
            //     std::cout << x << "\t" << y << std::endl;
            // }
            auto abs_diff = std::abs(x - y);
            auto rel_diff = abs_diff / std::abs(y + 1e-6f);
            if (!(abs_diff <= atol + rtol * std::abs(y))) {
                ++outliers;
                if (show) {
                    std::cout << nn << "," << mm << "\t" << x << "\t" << y << std::endl;
                }
            }
            abs_diff_sum += abs_diff;
            rel_diff_sum += rel_diff;
        }
        asums += abs_diff_sum / m;
        rsums += rel_diff_sum / m;
    }
    std::cout << "abs_diff = " << asums / n << " rel_diff = " << rsums / n << " outliers = " << outliers / (float)n
              << std::endl;
}

template void Compare(const half* src, const half* ref, size_t stride, int m, int n, bool show, float rtol, float atol);
template void
Compare(const float* src, const float* ref, size_t stride, int m, int n, bool show, float rtol, float atol);
#if ENABLE_BF16
template void Compare(const nv_bfloat16* src, const nv_bfloat16* ref, size_t stride, int m, int n, bool show, float rtol, float atol);
#endif


void LoadBinary(const std::string& path, size_t size, void* dst)
{
    std::ifstream ifs(path, std::ios::binary | std::ios::in);
    if (!ifs.is_open()) {
        std::cerr << "failed to open " << path << "\n";
        std::abort();
    }
    ifs.seekg(0, ifs.end);
    auto actual_size_in_bytes = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    if (size != actual_size_in_bytes) {
        std::cerr << "[warning] file " << path << " has " << actual_size_in_bytes << " bytes, while " << size
                  << " bytes is requested\n";
    }
    ifs.read((char*)dst, size);
    std::cerr << "[info] " << path << " " << size << "\n";
}

namespace cg = cooperative_groups;

__global__ void curand_init(curandState* state)
{
    auto tid = cg::this_grid().thread_rank();
    curand_init(0xe4c45822e90461ddULL, tid, 0, state + tid);
}

template<typename T>
__global__ void curand_uniform(curandState* state, size_t count, T* result, float scale, float shift)
{
    auto grid = cg::this_grid();
    for (auto i = grid.thread_rank(); i < count; i += grid.size()) {
        float tmp = curand_uniform(state + grid.thread_rank());
        result[i] = T(scale * tmp + shift);
    }
}

template<typename T>
__global__ void curand_normal(curandState* state, size_t count, T* result, float scale, float shift)
{
    auto grid = cg::this_grid();
    for (auto i = grid.thread_rank(); i < count; i += grid.size()) {
        float tmp = curand_normal(state + grid.thread_rank());
        result[i] = T(scale * tmp + shift);
    }
}

__global__ void curand_bytes(curandState* state, size_t count, uint* result)
{
    auto grid = cg::this_grid();
    for (auto i = grid.thread_rank(); i < count; i += grid.size()) {
        result[i] = curand(state + grid.thread_rank());
    }
}

struct RNG::Impl {

    curandState* states{};

    Impl()
    {
        cudaMalloc(&states, sizeof(curandState) * 64 * 64);
        curand_init<<<64, 64>>>(states);
    }

    ~Impl()
    {
        cudaFree(states);
    }

    void GenerateUInt(uint* out, size_t count)
    {
        curand_bytes<<<64, 64>>>(states, count, out);
    }

    template<typename T>
    void GenerateUniform(T* out, size_t count, float scale, float shift)
    {
        curand_uniform<<<64, 64>>>(states, count, out, scale, shift);
    }

    template<typename T>
    void GenerateNormal(T* out, size_t count, float scale, float shift)
    {
        curand_normal<<<64, 64>>>(states, count, out, scale, shift);
    }
};

RNG::RNG(): impl_(std::make_unique<Impl>()) {}

RNG::~RNG() = default;

void RNG::GenerateUInt(uint* out, size_t count)
{
    impl_->GenerateUInt(out, count);
}

template<typename T>
void RNG::GenerateUniform(T* out, size_t count, float scale, float shift)
{
    std::cout << count << std::endl;
    impl_->GenerateUniform(out, count, scale, shift);
}

template<typename T>
void RNG::GenerateNormal(T* out, size_t count, float scale, float shift)
{
    impl_->GenerateNormal(out, count, scale, shift);
}

template void RNG::GenerateUniform(half* out, size_t count, float scale, float shift);
template void RNG::GenerateUniform(float* out, size_t count, float scale, float shift);
#if ENABLE_BF16
template void RNG::GenerateUniform(nv_bfloat16* out, size_t count, float scale, float shift);
#endif

template void RNG::GenerateNormal(half* out, size_t count, float scale, float shift);
template void RNG::GenerateNormal(float* out, size_t count, float scale, float shift);
#if ENABLE_BF16
template void RNG::GenerateNormal(nv_bfloat16* out, size_t count, float scale, float shift);
#endif


template<typename T>
struct SATypeConverter {
    using Type = T;
};

template<>
struct SATypeConverter<half> {
    using Type = uint16_t;
};

template<typename T>
void mmha_ft_reference(const AttentionParams<T>& p,
                       T**                       per_sample_k_cache,
                       T**                       per_sample_v_cache,
                       const int*                sequence_length,
                       int                       max_memory_len,
                       cudaStream_t              st)
{
    using DataType = typename SATypeConverter<T>::Type;

    // Prepare the parameters.
    Masked_multihead_attention_params<DataType> params{};
    params.q_bias = reinterpret_cast<const DataType*>(p.q_bias);
    params.k_bias = reinterpret_cast<const DataType*>(p.k_bias);
    params.v_bias = reinterpret_cast<const DataType*>(p.v_bias);

    // Set the output buffer.
    params.out = reinterpret_cast<DataType*>(p.out);

    // Set the input buffers.
    // [B, nH + kvH, D]
    params.q = reinterpret_cast<const DataType*>(p.q);
    params.k = reinterpret_cast<const DataType*>(p.k);
    params.v = reinterpret_cast<const DataType*>(p.v);

    params.stride   = p.stride;
    params.finished = (bool*)p.finished;

    params.k_cache_per_sample         = reinterpret_cast<DataType**>(per_sample_k_cache);
    params.v_cache_per_sample         = reinterpret_cast<DataType**>(per_sample_v_cache);
    params.kv_cache_per_sample_offset = 0;  // single layer
    params.batch_size                 = p.batch_size;
    params.beam_width                 = 1;
    params.memory_max_len             = max_memory_len;
    params.prefix_prompt_lengths      = 0;
    params.max_prefix_prompt_length   = 0;
    params.length_per_sample          = sequence_length;  // max_input_length + current output length

    for (int i = 0; i < p.batch_size; ++i) {
        params.timestep = std::max(sequence_length[i], params.timestep);
    }

    std::cout << "timestep = " << params.timestep << "\n";

    params.num_heads    = p.num_heads;
    params.num_kv_heads = p.num_kv_heads;

    params.hidden_size_per_head    = p.size_per_head;
    params.rotary_embedding_dim    = p.rotary_embedding_dim;
    params.max_position_embeddings = p.max_position_embeddings;
    params.use_dynamic_ntk         = false;
    params.use_logn_attn           = p.use_logn_attn;

    // Note: keep norm factor (sqrt(K_dim)) when adopting megatron T5 structure (may adjust)
    params.inv_sqrt_dh = 1.F / (sqrtf((float)params.hidden_size_per_head) * 1.f);

    params.int8_mode = 0;

    masked_multihead_attention(params, st);
}

template void mmha_ft_reference(const AttentionParams<half>& params,
                                half**                       per_sample_k_cache,
                                half**                       per_sample_v_cache,
                                const int*                   sequence_length,
                                int                          max_memory_len,
                                cudaStream_t                 st);

}  // namespace turbomind
