/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tests/unittests/gtest_utils.h"
#include "src/turbomind/kernels/gen_relative_pos_bias.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/kernels/unfused_attention_kernels.h"
#include "src/turbomind/utils/memory_utils.h"
#include "src/turbomind/utils/nccl_utils.h"
#include "src/turbomind/utils/Tensor.h"

#include <curand.h>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace turbomind;

namespace {

struct AttentionKernelTestParam {
    size_t batch_size    = 4;
    size_t q_length      = 32;
    size_t k_length      = 32;
    size_t head_num      = 4;
    size_t size_per_head = 32;

    bool   use_fp32_qk_buf      = false;
    size_t rotary_embedding_dim = 0;
    bool   neox_rotary_style    = false;

    float  q_scaling = 1.0f;
};

namespace utils {

#define CHECK_CURAND(cmd) do {                                                      \
    curandStatus_t err = cmd;                                                       \
    if (err != CURAND_STATUS_SUCCESS) {                                             \
        throw std::runtime_error(                                                   \
            fmtstr("[FT][ERROR] curand runtime error: %d", err)); \
    }} while(0)                                                                     \

__global__ void convert_and_copy(half* dst, const float* src, const size_t size)
{
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < size; idx += blockDim.x * gridDim.x) {
        dst[idx] = __float2half(src[idx]);
    }
}

#ifdef ENABLE_BF16
__global__ void convert_and_copy(__nv_bfloat16* dst, const float* src, const size_t size)
{
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < size; idx += blockDim.x * gridDim.x) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}
#endif

template<typename T>
void normal(curandGenerator_t curng, T* buf, size_t size, float mean, float stddev)
{
    float* tmp_buf = nullptr;
    deviceMalloc(&tmp_buf, size);

    // Generate random values in float data type.
    CHECK_CURAND(curandGenerateNormal(curng, tmp_buf, size / 2, mean, stddev));
    sync_check_cuda_error();

    // Convert and copy to the output buffer if it is not of type float.
    dim3 block(512);
    dim3 grid(min(static_cast<int>((size + block.x - 1) / block.x), 256));
    convert_and_copy<<<grid, block>>>(buf, tmp_buf, size);
    cudaDeviceSynchronize();

    deviceFree(tmp_buf);
    sync_check_cuda_error();
}

template<>
void normal(curandGenerator_t curng, float* buf, size_t size, float mean, float stddev)
{
    // Generate random values in float data type.
    CHECK_CURAND(curandGenerateNormal(curng, buf, size / 2, mean, stddev));
    sync_check_cuda_error();
}

template<typename T>
void normal(curandGenerator_t curng, Tensor& tensor, float mean = 0.0f, float stddev = 1.0f)
{
    if (tensor.size() > 0) {
        FT_CHECK(tensor.type == getTensorType<T>());
        normal(curng, tensor.getPtr<T>(), tensor.size(), mean, stddev);
    }
}

__host__ uint32_t pow2_rounddown(uint32_t x)
{
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x >>= 1;
    return x + 1;
}

}  // namespace utils

////////////////////////////
// Reference computation.
////////////////////////////

template<typename T>
inline T safe_add_bias(const T v, const T* bias, const size_t bias_idx)
{
    return bias == nullptr ? v : ::math::add(v, bias[bias_idx]);
}

template<typename T>
void computeQkSoftmax(T*           attn_score,
                      const T*     qk,
                      const T*     attn_mask,
                      const T*     pos_bias,
                      const size_t batch_size,
                      const size_t num_heads,
                      const size_t q_length,
                      const size_t k_length,
                      const T      qk_scale)
{
    // attn_score [batch_size, num_heads, q_length, k_length]
    // qk         [batch_size, num_heads, q_length, k_length]
    // attn_mask  [batch_size, 1, q_length, k_length]
    // pos_bias   [1, num_heads, q_length, k_length]

    // batch, head index.
    for (size_t bhi = 0; bhi < batch_size * num_heads; ++bhi) {
        size_t bi = bhi / num_heads;  // batch index.
        size_t hi = bhi % num_heads;  // head index.
        // The attention mask of the current batch.
        const T* mask = &attn_mask[bi * q_length * k_length];
        // The position bias of the current head.
        const T* head_pos_bias = pos_bias != nullptr ? &pos_bias[hi * q_length * k_length] : nullptr;

        for (size_t qi = 0; qi < q_length; ++qi) {
            float maxval = -FLT_MAX;
            for (size_t ki = 0; ki < k_length; ++ki) {
                size_t qk_idx = qi * k_length + ki;
                if (int(mask[qk_idx]) > 0) {  // mask = 0 or 1.
                    float val = (float)safe_add_bias(::math::mul(qk_scale, qk[qk_idx]), head_pos_bias, qk_idx);
                    if (val > maxval) {
                        maxval = val;
                    }
                }
            }
            float sum = 0.0f;
            for (size_t ki = 0; ki < k_length; ++ki) {
                size_t qk_idx = qi * k_length + ki;
                if (int(mask[qk_idx]) > 0) {  // mask = 0 or 1.
                    float val = (float)safe_add_bias(::math::mul(qk_scale, qk[qk_idx]), head_pos_bias, qk_idx);
                    sum += expf(val - maxval);
                }
            }
            for (size_t ki = 0; ki < k_length; ++ki) {
                size_t qk_idx = qi * k_length + ki;
                if (int(mask[qk_idx]) > 0) {  // mask = 0 or 1.
                    float val = (float)safe_add_bias(::math::mul(qk_scale, qk[qk_idx]), head_pos_bias, qk_idx);
                    attn_score[qk_idx] = static_cast<T>(expf(val - maxval) / (sum + EPSILON));
                }
                else {
                    attn_score[qk_idx] = T(0.0f);
                }
            }
        }

        // Move the data pointers to the next.
        attn_score += q_length * k_length;
        qk         += q_length * k_length;
    }
}

template<typename T>
class AttentionKernelTest : public FtTestBase {

private:
    using FtTestBase::stream;
    using FtTestBase::allocator;

    unsigned long long seed = 31;
    curandGenerator_t  curng;

    Tensor randomAttentionMask(const std::vector<size_t> shape)
    {
        // shape (batch_size, 1, max_input_length, max_input_length + max_prompt_length)

        // Create a attention mask tensor and buffer.
        Tensor attn_mask = createTensor(MEMORY_GPU, getTensorType<T>(), shape);

        // Set the mask values.
        size_t batch_size   = shape[0];
        size_t max_q_length = shape[2];
        size_t max_k_length = shape[3];
        // TODO: Enable prompts.
        size_t max_prompt_length = max_k_length - max_q_length;

        Tensor h_seq_lengths    = createTensor(MEMORY_CPU, TYPE_INT32, {batch_size});
        Tensor h_prompt_lengths = createTensor(MEMORY_CPU, TYPE_INT32, {batch_size});
        initRandomInt(h_seq_lengths.getPtr<int>(), batch_size, max_q_length, max_q_length + 1);
        initRandomInt(h_prompt_lengths.getPtr<int>(), batch_size, 0, max_prompt_length + 1);

        Tensor d_seq_lengths    = createTensor(MEMORY_GPU, TYPE_INT32, {batch_size});
        Tensor d_prompt_lengths = createTensor(MEMORY_GPU, TYPE_INT32, {batch_size});
        copyTensor(d_seq_lengths, h_seq_lengths);
        copyTensor(d_prompt_lengths, h_prompt_lengths);

        // Used gpt_kernels function to build attention mask.
        invokeBuildDecoderAttentionMask(attn_mask.getPtr<T>(),
                                        d_seq_lengths.getPtr<int>(),
                                        d_prompt_lengths.getPtr<int>(),
                                        batch_size,
                                        max_q_length,
                                        max_prompt_length,
                                        stream);
        sync_check_cuda_error();
        return attn_mask;
    };

public:
    void SetUp() override
    {
        FtTestBase::SetUp();
        CHECK_CURAND(curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT));
        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(curng, seed));
    }

    void TearDown() override
    {
        curandDestroyGenerator(curng);
        FtTestBase::TearDown();
    }

    void runTestMaskedSoftmax(AttentionKernelTestParam param, bool is_benchmark = false) {
        DataType dtype = getTensorType<T>();

        std::vector<size_t> qk_shape {param.batch_size, param.head_num, param.q_length, param.k_length};

        bool use_fp32_qk = param.use_fp32_qk_buf && dtype != TYPE_FP32;

        Tensor qk        = createTensor(MEMORY_GPU, dtype, qk_shape);
        Tensor qk_fp32   = use_fp32_qk ? createTensor(MEMORY_GPU, TYPE_FP32, qk_shape) : Tensor();
        Tensor attn_mask = randomAttentionMask({param.batch_size, 1, param.q_length, param.k_length});
        // Input random initialization
        if (param.use_fp32_qk_buf && dtype != TYPE_FP32) {
            utils::normal<float>(curng, qk_fp32);
        }
        else {
            utils::normal<T>(curng, qk);
        }

        // Clone to host for reference computation if needed.
        Tensor h_qk        = is_benchmark ? Tensor() : toHost<T>(qk);
        Tensor h_attn_mask = is_benchmark ? Tensor() : toHost<T>(attn_mask);
        Tensor h_qk_fp32   = is_benchmark ? Tensor() : toHost<float>(qk_fp32);

        T scale = static_cast<T>(1 / sqrtf(param.size_per_head * 1.0f));

        if (param.use_fp32_qk_buf && dtype != TYPE_FP32) {
            MaskedSoftmaxParam<T, float> softmax_param;
            softmax_param.attention_score    = qk.getPtr<T>();
            softmax_param.qk                 = qk_fp32.getPtr<float>();
            softmax_param.attention_mask     = attn_mask.getPtr<T>();
            softmax_param.batch_size         = param.batch_size;
            softmax_param.num_heads          = param.head_num;
            softmax_param.q_length           = param.q_length;
            softmax_param.k_length           = param.k_length;
            softmax_param.qk_scale           = scale;
            invokeMaskedSoftmax(softmax_param, stream);
            sync_check_cuda_error();
        }
        else {
            MaskedSoftmaxParam<T, T> softmax_param;
            softmax_param.attention_score    = qk.getPtr<T>();
            softmax_param.qk                 = qk.getPtr<T>();
            softmax_param.attention_mask     = attn_mask.getPtr<T>();
            softmax_param.batch_size         = param.batch_size;
            softmax_param.num_heads          = param.head_num;
            softmax_param.q_length           = param.q_length;
            softmax_param.k_length           = param.k_length;
            softmax_param.qk_scale           = scale;
            invokeMaskedSoftmax(softmax_param, stream);
            sync_check_cuda_error();
        }

        if (!is_benchmark) {
            if (use_fp32_qk) {
                computeQkSoftmax(h_qk.getPtr<T>(),
                                 h_qk_fp32.getPtr<T>(),
                                 h_attn_mask.getPtr<T>(),
                                 (T*)nullptr,
                                 param.batch_size,
                                 param.head_num,
                                 param.q_length,
                                 param.k_length,
                                 scale);
            }
            else {
                computeQkSoftmax(h_qk.getPtr<T>(),
                                 h_qk.getPtr<T>(),
                                 h_attn_mask.getPtr<T>(),
                                 (T*)nullptr,
                                 param.batch_size,
                                 param.head_num,
                                 param.q_length,
                                 param.k_length,
                                 scale);
            }
            bool passed = checkResult("MaskedSoftmax", qk.getPtr<T>(), h_qk.getPtr<T>(), qk.size());
            EXPECT_TRUE(passed);
        }
    }

    void runTestAlibiMaskedSoftmax(AttentionKernelTestParam param, bool is_benchmark = false) {
        DataType dtype = getTensorType<T>();

        std::vector<size_t> qk_shape {param.batch_size, param.head_num, param.q_length, param.k_length};

        bool use_fp32_qk = param.use_fp32_qk_buf && dtype != TYPE_FP32;

        Tensor qk           = createTensor(MEMORY_GPU, dtype, qk_shape);
        Tensor qk_fp32      = use_fp32_qk ? createTensor(MEMORY_GPU, TYPE_FP32, qk_shape) : Tensor();
        Tensor attn_mask    = randomAttentionMask({param.batch_size, 1, param.q_length, param.k_length});
        Tensor alibi_slopes = createTensor(MEMORY_GPU, dtype, {param.head_num});

        // Input random initialization
        if (param.use_fp32_qk_buf && dtype != TYPE_FP32) {
            utils::normal<float>(curng, qk_fp32);
        }
        else {
            utils::normal<T>(curng, qk);
        }
        invokeBuildAlibiSlopes(alibi_slopes.getPtr<T>(), param.head_num, stream);
        sync_check_cuda_error();

        Tensor h_alibi_slopes = createTensor(MEMORY_CPU, dtype, {param.head_num});
        Tensor h_alibi_bias   = is_benchmark ? Tensor() :
            createTensor(MEMORY_CPU, dtype, {param.head_num, param.q_length, param.k_length});
        // The nearest power of 2 equal to / smaller than num_heads followed by HF's implementation.
        T* alibi_slope_ptr = h_alibi_slopes.getPtr<T>();
        int num_heads_pow2 = utils::pow2_rounddown(param.head_num);
        for (size_t h = 0; h < param.head_num; ++h) {
            // The slope of linear bias of the attention head
            if (h < num_heads_pow2) {
                alibi_slope_ptr[h] = static_cast<T>(powf(powf(0.5f, powf(0.5f, log2f(num_heads_pow2) - 3.f)), h + 1));
            } else {
                alibi_slope_ptr[h] = static_cast<T>(
                    powf(powf(0.5f, powf(0.5f, log2f(num_heads_pow2 << 1) - 3.f)), (h - num_heads_pow2) * 2 + 1));
            }
            if (h_alibi_bias.size() > 0) {
                T* alibi_bias_ptr = h_alibi_bias.getPtr<T>();
                for (size_t qi = 0; qi < param.q_length; ++qi) {
                    for (size_t ki = 0; ki < param.k_length; ++ki) {
                        size_t hqk_idx = (h * param.q_length + qi) * param.k_length + ki;
                        alibi_bias_ptr[hqk_idx] = ::math::mul(alibi_slope_ptr[h], T(0.0f + ki - qi));
                    }
                }
            }
        }
        EXPECT_TRUE(
            checkResult("CheckAlibiSlopes", alibi_slopes.getPtr<T>(), h_alibi_slopes.getPtr<T>(), param.head_num));

        // Clone to host for reference computation if needed.
        Tensor h_qk        = is_benchmark ? Tensor() : toHost<T>(qk);
        Tensor h_attn_mask = is_benchmark ? Tensor() : toHost<T>(attn_mask);
        Tensor h_qk_fp32   = is_benchmark ? Tensor() : toHost<float>(qk_fp32);

        T scale = static_cast<T>(1 / sqrtf(param.size_per_head * 1.0f));

        if (param.use_fp32_qk_buf && dtype != TYPE_FP32) {
            MaskedSoftmaxParam<T, float> softmax_param;
            softmax_param.attention_score    = qk.getPtr<T>();
            softmax_param.qk                 = qk_fp32.getPtr<float>();
            softmax_param.attention_mask     = attn_mask.getPtr<T>();
            softmax_param.linear_bias_slopes = alibi_slopes.getPtr<T>();
            softmax_param.batch_size         = param.batch_size;
            softmax_param.num_heads          = param.head_num;
            softmax_param.q_length           = param.q_length;
            softmax_param.k_length           = param.k_length;
            softmax_param.qk_scale           = scale;
            invokeMaskedSoftmax(softmax_param, stream);
            sync_check_cuda_error();
        }
        else {
            MaskedSoftmaxParam<T, T> softmax_param;
            softmax_param.attention_score    = qk.getPtr<T>();
            softmax_param.qk                 = qk.getPtr<T>();
            softmax_param.attention_mask     = attn_mask.getPtr<T>();
            softmax_param.linear_bias_slopes = alibi_slopes.getPtr<T>();
            softmax_param.batch_size         = param.batch_size;
            softmax_param.num_heads          = param.head_num;
            softmax_param.q_length           = param.q_length;
            softmax_param.k_length           = param.k_length;
            softmax_param.qk_scale           = scale;
            invokeMaskedSoftmax(softmax_param, stream);
            sync_check_cuda_error();
        }

        if (!is_benchmark) {
            if (use_fp32_qk) {
                computeQkSoftmax(h_qk.getPtr<T>(),
                                 h_qk_fp32.getPtr<T>(),
                                 h_attn_mask.getPtr<T>(),
                                 h_alibi_bias.getPtr<T>(),
                                 param.batch_size,
                                 param.head_num,
                                 param.q_length,
                                 param.k_length,
                                 scale);
            }
            else {
                computeQkSoftmax(h_qk.getPtr<T>(),
                                 h_qk.getPtr<T>(),
                                 h_attn_mask.getPtr<T>(),
                                 h_alibi_bias.getPtr<T>(),
                                 param.batch_size,
                                 param.head_num,
                                 param.q_length,
                                 param.k_length,
                                 scale);
            }
            bool passed = checkResult("AlibiMaskedSoftmax", qk.getPtr<T>(), h_qk.getPtr<T>(), qk.size());
            EXPECT_TRUE(passed);
        }
    }
};

TYPED_TEST_SUITE(AttentionKernelTest, SupportTypes);

TYPED_TEST(AttentionKernelTest, MaskedSoftmax_NoPrompt) {
    this->runTestMaskedSoftmax({1, 12, 12, 1, 32, false, 0, false});
}

TYPED_TEST(AttentionKernelTest, MaskedSoftmax_NoPrompt2) {
    // q_length is not multiple of 4.
    this->runTestMaskedSoftmax({1, 11, 11, 4, 32, false, 0, false});
}

TYPED_TEST(AttentionKernelTest, MaskedSoftmax_HasPrompt) {
    this->runTestMaskedSoftmax({1, 12, 24, 2, 32, false, 0, false});
}

TYPED_TEST(AttentionKernelTest, MaskedSoftmax_HasPrompt2) {
    this->runTestMaskedSoftmax({1, 11, 24, 2, 32, false, 0, false});
}

TYPED_TEST(AttentionKernelTest, MaskedSoftmax_LongSequence1024) {
    this->runTestMaskedSoftmax({1, 12, 1024, 2, 32, false, 0, false});
}

TYPED_TEST(AttentionKernelTest, MaskedSoftmax_LongSequence2048) {
    this->runTestMaskedSoftmax({1, 12, 2048, 2, 32, false, 0, false});
}

TYPED_TEST(AttentionKernelTest, MaskedSoftmax_LongSequence3072) {
    this->runTestMaskedSoftmax({1, 12, 3072, 2, 32, false, 0, false});
}

TYPED_TEST(AttentionKernelTest, MaskedSoftmax_LongSequence4096) {
    this->runTestMaskedSoftmax({1, 12, 4096, 2, 32, false, 0, false});
}

TYPED_TEST(AttentionKernelTest, Benchmark_MaskedSoftmax_LongSequence1024) {
    // Assume the bloom 176B model with 8 TP.
    this->runTestMaskedSoftmax({8, 1024, 1024, 14, 128, false, 0, false, true}, true);
}

TYPED_TEST(AttentionKernelTest, Benchmark_MaskedSoftmax_LongSequence2048) {
    // Assume the bloom 176B model with 8 TP.
    this->runTestMaskedSoftmax({8, 2048, 2048, 14, 128, false, 0, false, true}, true);
}

TYPED_TEST(AttentionKernelTest, Benchmark_MaskedSoftmax_LongSequence4096) {
    // Assume the bloom 176B model with 8 TP.
    this->runTestMaskedSoftmax({8, 4096, 4096, 14, 128, false, 0, false, true}, true);
}

TYPED_TEST(AttentionKernelTest, AlibiMaskedSoftmax_ShortSequence1) {
    this->runTestAlibiMaskedSoftmax({1, 12, 12, 4, 32, false, 0, false});
}

TYPED_TEST(AttentionKernelTest, AlibiMaskedSoftmax_ShortSequence2) {
    // q_length is not multiple of 4.
    this->runTestAlibiMaskedSoftmax({1, 11, 11, 4, 32, false, 0, false});
}

TYPED_TEST(AttentionKernelTest, AlibiMaskedSoftmax_ShortSequence_HasPrompt1) {
    this->runTestAlibiMaskedSoftmax({1, 12, 20, 4, 32, false, 0, false});
}

TYPED_TEST(AttentionKernelTest, AlibiMaskedSoftmax_ShortSequence_HasPrompt2) {
    // q_length is not multiple of 4.
    this->runTestAlibiMaskedSoftmax({1, 11, 20, 4, 32, false, 0, false});
}

// Tests for long sentence generation. Assume the bloom 176B model with 8 TP.

TYPED_TEST(AttentionKernelTest, Benchmark_AlibiMaskedSoftmax_LongSequence1024) {
    this->runTestAlibiMaskedSoftmax({8, 1024, 1024, 14, 128, false, 0, false, true}, true);
}

TYPED_TEST(AttentionKernelTest, Benchmark_AlibiMaskedSoftmax_LongSequence2048) {
    this->runTestAlibiMaskedSoftmax({8, 2048, 2048, 14, 128, false, 0, false, true}, true);
}

TYPED_TEST(AttentionKernelTest, Benchmark_AlibiMaskedSoftmax_LongSequence3072) {
    this->runTestAlibiMaskedSoftmax({8, 3072, 3072, 14, 128, false, 0, false, true}, true);
}

TYPED_TEST(AttentionKernelTest, Benchmark_AlibiMaskedSoftmax_LongSequence4096) {
    this->runTestAlibiMaskedSoftmax({4, 4096, 4096, 14, 128, false, 0, false, true}, true);
}

}  // end of namespace
