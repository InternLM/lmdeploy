/*
 * Copyright (c) OpenMMLab. All rights reserved.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "src/turbomind/core/core.h"
#include "src/turbomind/kernels/cross_entropy_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

using namespace turbomind;

namespace {

float LogitValue(int row, int col)
{
    const int mixed = (row * 131 + col * 17 + (col / 97) * 29) % 257;
    return (mixed - 128) * 0.01f + ((row + col) % 4096 == 0 ? 0.25f : 0.f);
}

template<typename T>
float ReferenceCrossEntropyLoss(const Tensor_<T>&   logits_storage,
                                const Tensor_<int>& targets,
                                int                 target_offset,
                                int                 logit_offset,
                                int                 token_num,
                                int                 vocab_size,
                                int                 logits_stride,
                                float               initial_loss)
{
    float loss = initial_loss;

    for (int row = 0; row < token_num; ++row) {
        const int storage_row = logit_offset + row;
        const T*  row_logits  = logits_storage.data() + storage_row * logits_stride;

        float max_val = -INFINITY;
        for (int i = 0; i < vocab_size; ++i) {
            max_val = std::max(max_val, static_cast<float>(row_logits[i]));
        }

        float sum_exp = 0.f;
        for (int i = 0; i < vocab_size; ++i) {
            sum_exp += std::exp(static_cast<float>(row_logits[i]) - max_val);
        }

        const int target = targets.data()[target_offset + row];
        loss += std::log(sum_exp + 1e-9f) + max_val - static_cast<float>(row_logits[target]);
    }

    return loss;
}

void CheckClose(float actual, float expected, const char* case_name, DataType dtype)
{
    const float atol = 5e-3f;
    const float rtol = 5e-4f;
    const float diff = std::abs(actual - expected);
    std::cout << "test_cross_entropy: " << case_name << " dtype=" << dtype << " actual=" << actual
              << " expected=" << expected << " diff=" << diff << std::endl;
    if (diff > atol + rtol * std::abs(expected)) {
        std::cerr << "test_cross_entropy failed: " << case_name << " dtype=" << dtype << " actual=" << actual
                  << " expected=" << expected << " diff=" << diff << std::endl;
        std::exit(1);
    }
}

template<typename T>
void RunAccumulationCase(const char* case_name, int vocab_size, int logits_stride, int token_num = 32)
{
    constexpr int   target_offset = 2;
    constexpr int   logit_offset  = 1;
    const int       total_rows    = logit_offset + token_num + 1;
    const int       target_count  = target_offset + token_num + 2;
    constexpr float initial_loss  = 0.375f;

    auto stream = core::Context::stream().handle();

    Tensor_<T> h_logits_storage{{total_rows, logits_stride}, kCPU};
    Tensor_<T> d_logits_storage{{total_rows, logits_stride}, kDEVICE};

    for (int row = 0; row < total_rows; ++row) {
        for (int col = 0; col < logits_stride; ++col) {
            const float value                                  = col < vocab_size ? LogitValue(row, col) : -99.f;
            h_logits_storage.data()[row * logits_stride + col] = static_cast<T>(value);
        }
    }
    Copy(h_logits_storage, d_logits_storage);

    Tensor_<int> h_targets{{target_count}, kCPU};
    Tensor_<int> d_targets{{target_count}, kDEVICE};
    for (int i = 0; i < target_count; ++i) {
        h_targets.data()[i] = (i * 9973 + 17) % vocab_size;
    }
    h_targets.data()[target_offset + 0] = vocab_size - 1;
    h_targets.data()[target_offset + 1] = vocab_size / 2;
    h_targets.data()[target_offset + 2] = vocab_size / 7;
    Copy(h_targets, d_targets);

    Tensor_<float> h_loss{{1}, kCPU};
    Tensor_<float> d_loss{{1}, kDEVICE};
    h_loss.data()[0] = initial_loss;
    Copy(h_loss, d_loss);

    Tensor logits{d_logits_storage.buffer(), Layout{{total_rows, vocab_size}, {logits_stride, 1}}};
    invokeCrossEntropyLoss(
        d_loss.data(), logits, d_targets.data(), target_offset, logit_offset, token_num, vocab_size, stream);

    Copy(d_loss, h_loss);
    TM_CUDA_CHECK(cudaStreamSynchronize(stream));

    const float expected = ReferenceCrossEntropyLoss(
        h_logits_storage, h_targets, target_offset, logit_offset, token_num, vocab_size, logits_stride, initial_loss);
    CheckClose(h_loss.data()[0], expected, case_name, data_type_v<T>);
}

template<typename T>
void RunZeroTokenCase()
{
    auto stream = core::Context::stream().handle();

    Tensor_<T>     d_logits{{1, 4}, kDEVICE};
    Tensor_<int>   d_targets{{1}, kDEVICE};
    Tensor_<float> h_loss{{1}, kCPU};
    Tensor_<float> d_loss{{1}, kDEVICE};

    h_loss.data()[0] = 7.25f;
    Copy(h_loss, d_loss);

    invokeCrossEntropyLoss(d_loss.data(), d_logits, d_targets.data(), 0, 0, 0, 4, stream);

    Copy(d_loss, h_loss);
    TM_CUDA_CHECK(cudaStreamSynchronize(stream));

    CheckClose(h_loss.data()[0], 7.25f, "zero_token", data_type_v<T>);
}

template<typename T>
void RunDtypeCases()
{
    RunAccumulationCase<T>("llama_vocab_offsets_accumulation", 32000, 32000);
    RunAccumulationCase<T>("qwen_vocab_padded_stride_offsets_accumulation", 151936, 152064);
    RunAccumulationCase<T>("long_prefill_2048_llama_vocab_accumulation", 32000, 32000, 2048);
    RunZeroTokenCase<T>();
}

}  // namespace

int main()
{
    core::ContextGuard ctx{core::Stream::create(), core::Allocator{kCPU}, core::Allocator{kDEVICE}};

    RunDtypeCases<half_t>();
#if ENABLE_BF16
    RunDtypeCases<bfloat16_t>();
#endif

    std::cout << "test_cross_entropy passed" << std::endl;
    return 0;
}
