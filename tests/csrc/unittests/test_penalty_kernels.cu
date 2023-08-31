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

#include <algorithm>  // std::min, std::max
#include <iostream>   // snprintf
#include <math.h>     // expf, log
#include <stdexcept>
#include <stdlib.h>   // rand
#include <string>     // std::string
#include <unordered_map>
#include <vector>     // std::vector

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "gtest_utils.h"
#include "src/turbomind/kernels/penalty_types.h"
#include "src/turbomind/kernels/sampling_penalty_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"

using namespace turbomind;

struct TemperatureTestParam {
    size_t batch_size;
    size_t vocab_size;
    float* temperatures;
    size_t temperatures_size;

    std::string toString()
    {
        return fmtstr("TemperatureTestParam[batch=%ld, vocab=%ld, temperatures=%s]",
                      batch_size,
                      vocab_size,
                      arr2str(temperatures, temperatures_size).c_str());
    }
};

size_t pad_vocab_size(size_t vocab_size, size_t pad = 8)
{
    return (vocab_size + pad - 1) / pad * pad;
}

template<typename T>
void applyRepetitonPenalty(T*           logits,
                           const int*   output_ids,
                           const int*   input_lengths,
                           const float  repetition_penalty,
                           const size_t step,
                           const size_t max_input_length,
                           const size_t batch_size,
                           const size_t vocab_size,
                           const size_t vocab_size_padded)
{
    bool* penalized = new bool[vocab_size];
    for (size_t i = 0; i < batch_size; ++i) {
        std::fill_n(penalized, vocab_size, false);
        size_t length = std::min<int>(step, input_lengths[i]);
        size_t offset = i * vocab_size_padded;
        for (size_t t = 0; t < step; ++t) {
            if (t >= (size_t)input_lengths[i] && t < max_input_length) {
                continue;
            }
            int token_id = output_ids[i + t * batch_size];
            if (!penalized[token_id]) {
                float logit = static_cast<float>(logits[offset + token_id]);
                logits[offset + token_id] =
                    static_cast<T>(logit < 0.0f ? logit * repetition_penalty : logit / repetition_penalty);
                penalized[token_id] = true;
            }
        }
    }
    delete[] penalized;
}

template<typename T>
void batchApplyRepetitonPenalty(T*           logits,
                                const int*   output_ids,
                                const int*   input_lengths,
                                const float* repetition_penalties,
                                const size_t step,
                                const size_t max_input_length,
                                const size_t batch_size,
                                const size_t vocab_size,
                                const size_t vocab_size_padded)
{
    bool* penalized = new bool[vocab_size];
    for (size_t i = 0; i < batch_size; ++i) {
        float repetition_penalty = repetition_penalties[i];
        std::fill_n(penalized, vocab_size, false);
        size_t offset = i * vocab_size_padded;
        for (size_t t = 0; t < step; ++t) {
            if (t >= (size_t)input_lengths[i] && t < max_input_length) {
                continue;
            }
            int token_id = output_ids[i + t * batch_size];
            if (!penalized[token_id]) {
                float logit = static_cast<float>(logits[offset + token_id]);
                logits[offset + token_id] =
                    static_cast<T>(logit < 0.0f ? logit * repetition_penalty : logit / repetition_penalty);
                penalized[token_id] = true;
            }
        }
    }
    delete[] penalized;
}

template<typename T>
void initLogitsAndBias(
    T* logits, T* bias, const size_t batch_size, const size_t vocab_size, const size_t vocab_size_padded)
{
    initRandom(logits, batch_size * vocab_size_padded, -5.0f, 5.0f);
    if (bias != nullptr) {
        initRandom(bias, vocab_size, -5.0f, 5.0f);
    }
    bool is_half = sizeof(T) == 2;
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < vocab_size_padded; ++j) {
            if (j >= vocab_size) {
                logits[i * vocab_size_padded + j] = static_cast<T>(is_half ? -65504.f : -FLT_MAX);
                if (bias != nullptr && i == 0) {
                    bias[j] = (T)0.0f;
                }
            }
        }
    }
}

/////////////////////////////////// Tests //////////////////////////////////////////

template<typename T>
class TemperaturePenaltyTest: public FtTestBase {
protected:
    // Set up test
    size_t batch_size_;
    size_t vocab_size_;
    size_t vocab_size_padded_;

    T* h_logits_;
    T* h_bias_;
    T* d_logits_;
    T* d_bias_;

    float* d_temperatures_;

    void subsetup(TemperatureTestParam param)
    {
        batch_size_        = param.batch_size;
        vocab_size_        = param.vocab_size;
        vocab_size_padded_ = pad_vocab_size(vocab_size_);

        h_logits_ = new T[batch_size_ * vocab_size_padded_];
        h_bias_   = new T[vocab_size_padded_];
        initLogitsAndBias(h_logits_, h_bias_, batch_size_, vocab_size_, vocab_size_padded_);

        d_logits_ = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batch_size_ * vocab_size_padded_));
        d_bias_   = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * vocab_size_padded_));
        cudaAutoCpy(d_logits_, h_logits_, batch_size_ * vocab_size_padded_, stream);
        cudaAutoCpy(d_bias_, h_bias_, vocab_size_padded_, stream);
        if (param.temperatures_size > 1) {
            ASSERT_EQ(param.temperatures_size, param.batch_size) << "Invalid test configuration.";
            d_temperatures_ = reinterpret_cast<float*>(allocator->malloc(sizeof(T) * param.temperatures_size));
            cudaAutoCpy(d_temperatures_, param.temperatures, batch_size_, stream);
        }
    }

    void subteardown()
    {
        delete[] h_logits_;
        delete[] h_bias_;
    }

    void computeReference(T*           logits,
                          const T*     bias,
                          const float* temperatures,
                          const size_t temperatures_size,
                          const size_t batch_size,
                          const size_t vocab_size,
                          const size_t vocab_size_padded)
    {
        for (size_t i = 0; i < batch_size; ++i) {
            float temperature = temperatures_size > 1 ? temperatures[i] : temperatures[0];
            ASSERT_GT(temperature, 0.0f) << "temperature should be positive but got " << temperature;
            for (size_t j = 0; j < vocab_size; ++j) {
                size_t index = i * vocab_size_padded + j;
                float  logit = static_cast<float>(logits[index]);
                if (bias != nullptr) {
                    logit += static_cast<float>(bias[j]);
                }
                logits[index] = static_cast<T>(logit / temperature);
            }
        }
    }

public:
    void runTest(TemperatureTestParam param)
    {
        subsetup(param);
        // Do test
        if (param.temperatures_size == 1) {
            invokeApplyTemperaturePenalty(
                d_logits_, d_bias_, param.temperatures[0], batch_size_, vocab_size_, vocab_size_padded_, stream);
        }
        else {
            invokeBatchApplyTemperaturePenalty(
                d_logits_, d_bias_, d_temperatures_, batch_size_, vocab_size_, vocab_size_padded_, stream);
        }
        computeReference(h_logits_,
                         h_bias_,
                         param.temperatures,
                         param.temperatures_size,
                         batch_size_,
                         vocab_size_,
                         vocab_size_padded_);
        bool passed = checkResult(param.toString(), d_logits_, h_logits_, batch_size_ * vocab_size_padded_);
        EXPECT_TRUE(passed);
        subteardown();
    }

    void runConsistencyTest(TemperatureTestParam param)
    {
        // Set up test
        ASSERT_EQ(param.temperatures_size, 1) << "A consistency test assumes temperatures_size=1";
        subsetup(param);

        // Run a single runtime value case.
        invokeApplyTemperaturePenalty(
            d_logits_, d_bias_, param.temperatures[0], batch_size_, vocab_size_, vocab_size_padded_, stream);

        float  temperature    = param.temperatures[0];
        float* h_temperatures = new float[batch_size_];
        for (size_t i = 0; i < batch_size_; ++i) {
            h_temperatures[i] = temperature;
        }
        d_temperatures_ = reinterpret_cast<float*>(allocator->malloc(sizeof(T) * batch_size_));
        cudaAutoCpy(d_temperatures_, h_temperatures, batch_size_, stream);

        T* d_logits_batch = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batch_size_ * vocab_size_padded_));
        T* d_bias_batch   = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * vocab_size_padded_));
        cudaAutoCpy(d_logits_batch, h_logits_, batch_size_ * vocab_size_padded_, stream);
        cudaAutoCpy(d_bias_batch, h_bias_, vocab_size_padded_, stream);

        invokeBatchApplyTemperaturePenalty(
            d_logits_batch, d_bias_batch, d_temperatures_, batch_size_, vocab_size_, vocab_size_padded_, stream);
        bool passed =
            checkResult(param.toString(), d_logits_, d_logits_batch, batch_size_ * vocab_size_padded_, true, true);
        EXPECT_TRUE(passed);

        // Tear down test
        delete[] h_temperatures;
        subteardown();
    }
};

// Since a compiler doesn't correctly catch the use of a variable inside gtest,
// we carefully suppress a compile warning message.
#pragma nv_diag_suppress 177

TYPED_TEST_SUITE(TemperaturePenaltyTest, FloatAndHalfTypes);

TYPED_TEST(TemperaturePenaltyTest, NoPenalty)
{
    float temperature = 1.0f;
    this->runTest({6, 4, &temperature, 1});
}

TYPED_TEST(TemperaturePenaltyTest, LessThanOne)
{
    float temperature = 0.53f;
    this->runTest({6, 4, &temperature, 1});
}

TYPED_TEST(TemperaturePenaltyTest, GreaterThaneOne)
{
    float temperature = 2.01f;
    this->runTest({6, 4, &temperature, 1});
}

TYPED_TEST(TemperaturePenaltyTest, LargeVocab)
{
    float temperature = 2.01f;
    this->runTest({6, 50001, &temperature, 1});
}

TYPED_TEST(TemperaturePenaltyTest, BatchNoPenalty)
{
    size_t batch_size   = 6;
    float* temperatures = new float[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        temperatures[i] = 1.0f;
    }
    this->runTest({batch_size, 4, temperatures, batch_size});
}

TYPED_TEST(TemperaturePenaltyTest, BatchLessThanOne)
{
    size_t batch_size   = 6;
    float* temperatures = new float[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        temperatures[i] = 0.53f;
    }
    this->runTest({batch_size, 4, temperatures, batch_size});
}

TYPED_TEST(TemperaturePenaltyTest, BatchGreaterThaneOne)
{
    size_t batch_size   = 6;
    float* temperatures = new float[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        temperatures[i] = 2.01f;
    }
    this->runTest({batch_size, 4, temperatures, batch_size});
}

TYPED_TEST(TemperaturePenaltyTest, BatchMixed)
{
    size_t batch_size   = 6;
    float* temperatures = new float[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        temperatures[i] = i % 2 == 0 ? 2.01f : 0.53f;
    }
    this->runTest({batch_size, 4, temperatures, batch_size});
}

TYPED_TEST(TemperaturePenaltyTest, Consistency)
{
    float temperature = 2.01f;
    this->runConsistencyTest({6, 4, &temperature, 1});
}

struct RepetitionPenaltyTestCase {
    size_t                batch_size;
    size_t                vocab_size;
    size_t                max_input_length;
    float*                repetition_penalties;
    size_t                repetition_penalties_size;
    RepetitionPenaltyType repetition_penalty_type;

    std::string toString()
    {
        static const std::unordered_map<RepetitionPenaltyType, std::string> typestr_map{
            {RepetitionPenaltyType::Additive, "additive"},
            {RepetitionPenaltyType::Multiplicative, "multiplicative"},
            {RepetitionPenaltyType::None, "none"}};
        return fmtstr("RepetitionPenaltyTestCase[batch=%ld, vocab=%ld, max_input_length=%ld, "
                      "repetition_penalties=%s, repetition_penalty_type=%s]",
                      batch_size,
                      vocab_size,
                      max_input_length,
                      arr2str(repetition_penalties, repetition_penalties_size).c_str(),
                      typestr_map.at(repetition_penalty_type).c_str());
    }
};

template<typename T>
class RepetitionPenaltyTest: public FtTestBase {
protected:
    // Set up test
    size_t batch_size_;
    size_t vocab_size_;
    size_t vocab_size_padded_;
    size_t max_input_length_;
    size_t sequence_length_;
    size_t step_;

    T*   h_logits_;
    T*   h_bias_;
    int* h_output_ids_;
    int* h_input_lengths_;

    T*   d_logits_;
    T*   d_bias_;
    int* d_output_ids_;
    int* d_input_lengths_;

    float* d_repetition_penalties_;

    void subsetup(RepetitionPenaltyTestCase param)
    {
        batch_size_        = param.batch_size;
        vocab_size_        = param.vocab_size;
        vocab_size_padded_ = pad_vocab_size(vocab_size_);
        max_input_length_  = param.max_input_length;
        sequence_length_   = 2 * max_input_length_;  // input + output
        step_              = sequence_length_ * 0.7;

        h_logits_        = new T[batch_size_ * vocab_size_padded_];
        h_bias_          = new T[vocab_size_padded_];
        h_output_ids_    = new int[sequence_length_ * batch_size_];
        h_input_lengths_ = new int[batch_size_];
        initLogitsAndBias(h_logits_, h_bias_, batch_size_, vocab_size_, vocab_size_padded_);
        initRandomInt(h_output_ids_, sequence_length_ * batch_size_, 0, vocab_size_);
        initRandomInt(h_input_lengths_, batch_size_, 1, max_input_length_);

        d_logits_        = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batch_size_ * vocab_size_padded_));
        d_bias_          = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * vocab_size_padded_));
        d_output_ids_    = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * sequence_length_ * batch_size_));
        d_input_lengths_ = reinterpret_cast<int*>(allocator->malloc(sizeof(int) * batch_size_));

        cudaAutoCpy(d_logits_, h_logits_, batch_size_ * vocab_size_padded_, stream);
        cudaAutoCpy(d_bias_, h_bias_, vocab_size_padded_, stream);
        cudaAutoCpy(d_output_ids_, h_output_ids_, sequence_length_ * batch_size_, stream);
        cudaAutoCpy(d_input_lengths_, h_input_lengths_, batch_size_, stream);
        if (param.repetition_penalties_size > 1) {
            ASSERT_EQ(param.repetition_penalties_size, param.batch_size) << "Invalid test configuration.";
            d_repetition_penalties_ =
                reinterpret_cast<float*>(allocator->malloc(sizeof(T) * param.repetition_penalties_size));
            cudaAutoCpy(d_repetition_penalties_, param.repetition_penalties, batch_size_, stream);
        }
    }

    void subteardown()
    {
        delete[] h_logits_;
        delete[] h_bias_;
        delete[] h_output_ids_;
        delete[] h_input_lengths_;
    }

    void computeReference(T*                          logits,
                          const int*                  output_ids,
                          const int*                  input_lengths,
                          const float*                repetition_penalties,
                          const size_t                repetition_penalties_size,
                          const RepetitionPenaltyType repetition_penalty_type,
                          const size_t                step,
                          const size_t                max_input_length,
                          const size_t                batch_size,
                          const size_t                vocab_size,
                          const size_t                vocab_size_padded)
    {
        bool* penalized = new bool[vocab_size];
        for (size_t i = 0; i < batch_size; ++i) {
            float repetition_penalty =
                repetition_penalties_size > 1 ? repetition_penalties[i] : repetition_penalties[0];

            std::fill_n(penalized, vocab_size, false);
            size_t offset = i * vocab_size_padded;
            for (size_t t = 0; t < step; ++t) {
                if (t >= (size_t)input_lengths[i] && t < max_input_length) {
                    continue;
                }
                int token_id = output_ids[i + t * batch_size];
                if (!penalized[token_id]) {
                    float logit = static_cast<float>(logits[offset + token_id]);
                    switch (repetition_penalty_type) {
                        case RepetitionPenaltyType::Additive:
                            logits[offset + token_id] = static_cast<T>(logit - repetition_penalty);
                            break;
                        case RepetitionPenaltyType::Multiplicative:
                            logits[offset + token_id] =
                                static_cast<T>(logit < 0.0f ? logit * repetition_penalty : logit / repetition_penalty);
                            break;
                        case RepetitionPenaltyType::None:
                            // None. do nothing.
                            break;
                        default:
                            throw std::domain_error("Invalid repetition penalty type.");
                    }
                    penalized[token_id] = true;
                }
            }
        }
        delete[] penalized;
    }

public:
    void runTest(RepetitionPenaltyTestCase param)
    {
        subsetup(param);
        // Do test
        if (param.repetition_penalties_size == 1) {
            invokeApplyRepetitionPenalty(d_logits_,
                                         param.repetition_penalties[0],
                                         nullptr,
                                         d_output_ids_,
                                         batch_size_,
                                         batch_size_,
                                         vocab_size_,
                                         vocab_size_padded_,
                                         d_input_lengths_,
                                         max_input_length_,
                                         step_,
                                         param.repetition_penalty_type,
                                         stream);
        }
        else {
            invokeBatchApplyRepetitionPenalty(d_logits_,
                                              d_repetition_penalties_,
                                              d_output_ids_,
                                              batch_size_,
                                              batch_size_,
                                              vocab_size_padded_,
                                              d_input_lengths_,
                                              max_input_length_,
                                              step_,
                                              param.repetition_penalty_type,
                                              stream);
        }
        computeReference(h_logits_,
                         h_output_ids_,
                         h_input_lengths_,
                         param.repetition_penalties,
                         param.repetition_penalties_size,
                         param.repetition_penalty_type,
                         step_,
                         max_input_length_,
                         batch_size_,
                         vocab_size_,
                         vocab_size_padded_);
        bool passed = checkResult(param.toString(), d_logits_, h_logits_, batch_size_ * vocab_size_padded_);
        EXPECT_TRUE(passed);
        subteardown();
    }

    void runConsistencyTest(RepetitionPenaltyTestCase param)
    {
        // Set up test
        ASSERT_EQ(param.repetition_penalties_size, 1) << "A consistency test assumes repetition_penalties_size=1";
        subsetup(param);

        // Run a single runtime value case.
        invokeApplyRepetitionPenalty(d_logits_,
                                     param.repetition_penalties[0],
                                     nullptr,
                                     d_output_ids_,
                                     batch_size_,
                                     batch_size_,
                                     vocab_size_,
                                     vocab_size_padded_,
                                     d_input_lengths_,
                                     max_input_length_,
                                     step_,
                                     param.repetition_penalty_type,
                                     stream);

        float* h_repetition_penalties = new float[batch_size_];
        for (size_t i = 0; i < batch_size_; ++i) {
            h_repetition_penalties[i] = param.repetition_penalties[0];
        }
        d_repetition_penalties_ = reinterpret_cast<float*>(allocator->malloc(sizeof(T) * batch_size_));
        cudaAutoCpy(d_repetition_penalties_, h_repetition_penalties, batch_size_, stream);

        T* d_logits_batch = reinterpret_cast<T*>(allocator->malloc(sizeof(T) * batch_size_ * vocab_size_padded_));
        cudaAutoCpy(d_logits_batch, h_logits_, batch_size_ * vocab_size_padded_, stream);
        invokeBatchApplyRepetitionPenalty(d_logits_batch,
                                          d_repetition_penalties_,
                                          d_output_ids_,
                                          batch_size_,
                                          batch_size_,
                                          vocab_size_padded_,
                                          d_input_lengths_,
                                          max_input_length_,
                                          step_,
                                          param.repetition_penalty_type,
                                          stream);
        bool passed =
            checkResult(param.toString(), d_logits_, d_logits_batch, batch_size_ * vocab_size_padded_, true, true);
        EXPECT_TRUE(passed);

        // Tear down test
        delete[] h_repetition_penalties;
        subteardown();
    }
};

TYPED_TEST_SUITE(RepetitionPenaltyTest, FloatAndHalfTypes);

TYPED_TEST(RepetitionPenaltyTest, NoPenalty)
{
    float repetition_penalty = 1.0f;
    this->runTest({6, 4, 5, &repetition_penalty, 1, RepetitionPenaltyType::Multiplicative});
}

TYPED_TEST(RepetitionPenaltyTest, LessThanOne)
{
    float repetition_penalty = 0.53f;
    this->runTest({6, 4, 5, &repetition_penalty, 1, RepetitionPenaltyType::Multiplicative});
}

TYPED_TEST(RepetitionPenaltyTest, GreaterThaneOne)
{
    float repetition_penalty = 2.01f;
    this->runTest({6, 4, 5, &repetition_penalty, 1, RepetitionPenaltyType::Multiplicative});
}

TYPED_TEST(RepetitionPenaltyTest, LargeVocab)
{
    float repetition_penalty = 2.01f;
    this->runTest({6, 50001, 1003, &repetition_penalty, 1, RepetitionPenaltyType::Multiplicative});
}

TYPED_TEST(RepetitionPenaltyTest, BatchNoPenalty)
{
    size_t batch_size           = 6;
    float* repetition_penalties = new float[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        repetition_penalties[i] = 1.0f;
    }
    this->runTest({batch_size, 4, 5, repetition_penalties, batch_size, RepetitionPenaltyType::Multiplicative});
}

TYPED_TEST(RepetitionPenaltyTest, BatchLessThanOne)
{
    size_t batch_size           = 6;
    float* repetition_penalties = new float[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        repetition_penalties[i] = 0.53f;
    }
    this->runTest({batch_size, 4, 5, repetition_penalties, batch_size, RepetitionPenaltyType::Multiplicative});
}

TYPED_TEST(RepetitionPenaltyTest, BatchGreaterThaneOne)
{
    size_t batch_size   = 6;
    float* temperatures = new float[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        temperatures[i] = 2.01f;
    }
    this->runTest({batch_size, 4, 5, temperatures, batch_size, RepetitionPenaltyType::Multiplicative});
}

TYPED_TEST(RepetitionPenaltyTest, BatchMixed)
{
    size_t batch_size           = 6;
    float* repetition_penalties = new float[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        repetition_penalties[i] = i % 2 == 0 ? 2.01f : 0.53f;
    }
    this->runTest({batch_size, 4, 5, repetition_penalties, batch_size, RepetitionPenaltyType::Multiplicative});
}

TYPED_TEST(RepetitionPenaltyTest, Consistency)
{
    float repetition_penalty = 2.01f;
    this->runConsistencyTest({6, 4, 5, &repetition_penalty, 1, RepetitionPenaltyType::Multiplicative});
}

TYPED_TEST(RepetitionPenaltyTest, PenaltyTypeAdditive)
{
    size_t batch_size           = 6;
    float* repetition_penalties = new float[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        repetition_penalties[i] = i % 2 == 0 ? 2.01f : 0.53f;
    }
    this->runTest({batch_size, 4, 5, repetition_penalties, batch_size, RepetitionPenaltyType::Additive});
}

TYPED_TEST(RepetitionPenaltyTest, PenaltyTypeAdditiveHasDefaultValueZero)
{
    float repetition_penalty = 1.0f;
    this->runTest({6, 4, 5, &repetition_penalty, 1, RepetitionPenaltyType::Additive});
}

TYPED_TEST(RepetitionPenaltyTest, PenaltyTypeAdditiveHasDefaultValueZero2)
{
    size_t batch_size           = 6;
    float* repetition_penalties = new float[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        repetition_penalties[i] = i % 2 == 0 ? 1.0f : 0.0f;
    }
    this->runTest({batch_size, 4, 5, repetition_penalties, batch_size, RepetitionPenaltyType::Additive});
}

// Turn on the warning message.
#pragma nv_diag_suppress 177
