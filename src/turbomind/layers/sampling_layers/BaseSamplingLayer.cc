/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "src/turbomind/layers/sampling_layers/BaseSamplingLayer.h"
#include "src/turbomind/kernels/sampling_penalty_kernels.h"
#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"

#include <algorithm>

namespace turbomind {

template<typename T>
void BaseSamplingLayer<T>::allocateBuffer(size_t batch_size, Tensor top_k, Tensor top_p)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    curandstate_buf_ = reinterpret_cast<curandState_t*>(
        allocator_->reMalloc(curandstate_buf_, sizeof(curandState_t) * batch_size, false));
    random_seeds_buf_ = reinterpret_cast<unsigned long long*>(
        allocator_->reMalloc(random_seeds_buf_, sizeof(unsigned long long) * batch_size, false));
    temperature_buf_ =
        reinterpret_cast<float*>(allocator_->reMalloc(temperature_buf_, sizeof(float) * batch_size, false));
    repetition_penalty_buf_ =
        reinterpret_cast<float*>(allocator_->reMalloc(repetition_penalty_buf_, sizeof(float) * batch_size, false));
    min_lengths_buf_ = reinterpret_cast<int*>(allocator_->reMalloc(min_lengths_buf_, sizeof(int) * batch_size, false));
    runtime_logits_buf_ = reinterpret_cast<T*>(
        allocator_->reMalloc(runtime_logits_buf_, sizeof(T) * batch_size * vocab_size_padded_, false));
    skip_decode_buf_ =
        reinterpret_cast<bool*>(allocator_->reMalloc(skip_decode_buf_, sizeof(bool) * batch_size, false));

    // host buffers.
    temperature_        = (float*)std::realloc((void*)temperature_, batch_size * sizeof(float));
    repetition_penalty_ = (float*)std::realloc((void*)repetition_penalty_, batch_size * sizeof(float));
    min_lengths_        = (int*)std::realloc((void*)min_lengths_, batch_size * sizeof(int));
    skip_decode_        = (bool*)std::realloc((void*)skip_decode_, batch_size * sizeof(bool));

    is_allocate_buffer_ = true;
}

template<typename T>
void BaseSamplingLayer<T>::freeBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&curandstate_buf_));
        allocator_->free((void**)(&random_seeds_buf_));
        allocator_->free((void**)(&temperature_buf_));
        allocator_->free((void**)(&repetition_penalty_buf_));
        allocator_->free((void**)(&min_lengths_buf_));
        allocator_->free((void**)(&runtime_logits_buf_));
        allocator_->free((void**)(&skip_decode_buf_));
        std::free(temperature_);
        std::free(repetition_penalty_);
        std::free(min_lengths_);
        std::free(skip_decode_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
BaseSamplingLayer<T>::BaseSamplingLayer(size_t             max_batch_size,
                                        size_t             vocab_size,
                                        size_t             vocab_size_padded,
                                        int                end_id,
                                        size_t             top_k,
                                        float              top_p,
                                        unsigned long long random_seed,
                                        float              temperature,
                                        float              len_penalty,
                                        float              repetition_penalty,
                                        cudaStream_t       stream,
                                        cublasMMWrapper*   cublas_wrapper,
                                        IAllocator*        allocator,
                                        bool               is_free_buffer_after_forward,
                                        cudaDeviceProp*    cuda_device_prop):
    DynamicDecodeBaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    vocab_size_(vocab_size),
    vocab_size_padded_(vocab_size_padded)
{
}

template<typename T>
BaseSamplingLayer<T>::BaseSamplingLayer(BaseSamplingLayer const& sampling_layer):
    DynamicDecodeBaseLayer(sampling_layer),
    vocab_size_(sampling_layer.vocab_size_),
    vocab_size_padded_(sampling_layer.vocab_size_padded_),
    sampling_workspace_size_(sampling_layer.sampling_workspace_size_)
{
}

template<typename T>
BaseSamplingLayer<T>::~BaseSamplingLayer()
{
}

template<typename T>
void BaseSamplingLayer<T>::setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args)
{
    // Set up the sampling layer for given runtime arguments.
    //
    // runtime_args:
    //     runtime_top_k [1] or [batch_size] on cpu, optional.
    //     runtime_top_p [1] or [batch_size] on cpu, optional
    //     temperature [1] or [batch_size] on cpu, optional
    //     repetition_penalty [1] or [batch_size] on cpu, optional
    //     presence_penalty [1] or [batch_size] on cpu, optional,
    //         repetition_penalty and presence_penalty are mutually exclusive.
    //     min_length [1] or [batch_size] on cpu, optional

    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    Tensor runtime_top_k = runtime_args->isExist("runtime_top_k") ? runtime_args->at("runtime_top_k") : Tensor();
    Tensor runtime_top_p = runtime_args->isExist("runtime_top_p") ? runtime_args->at("runtime_top_p") : Tensor();
    allocateBuffer(batch_size, runtime_top_k, runtime_top_p);

    // If runtime argument has single random seed, using this random seed to initialize the random table of all
    // sentences. If the argument has [batch_size] random seeds, initializing the random table by different random seeds
    // respectively. If no random seed, initialize the random table of all sentences by 0 directly.
    if (runtime_args->isExist("random_seed")) {
        Tensor random_seeds = runtime_args->at("random_seed");
        FT_CHECK_WITH_INFO(random_seeds.shape.size() == 1
                               && (random_seeds.size() == 1 || random_seeds.size() == batch_size),
                           fmtstr("random_seeds must be of shape [1] or [batch_size(%ld)], got random_seeds.shape=%s",
                                  batch_size,
                                  vec2str(random_seeds.shape).c_str()));
        if (random_seeds.size() == 1) {
            invokeCurandInitialize(curandstate_buf_, batch_size, random_seeds.getVal<unsigned long long>(), stream_);
            sync_check_cuda_error();
        }
        else {
            unsigned long long* random_seed_ptr = random_seeds.getPtr<unsigned long long>();
            cudaAutoCpy(random_seeds_buf_, random_seed_ptr, batch_size, stream_);
            invokeCurandBatchInitialize(curandstate_buf_, batch_size, random_seeds_buf_, stream_);
            sync_check_cuda_error();
        }
    }
    else {
        // Initialize curand states using the default seed 0.
        invokeCurandInitialize(curandstate_buf_, batch_size, 0, stream_);
    }

    // Setup penalties.
    const float default_temperature = 1.0f;
    Tensor      temperature         = runtime_args->isExist("temperature") ?
                                          runtime_args->at("temperature") :
                                          Tensor(MEMORY_CPU, TYPE_FP32, {1}, &default_temperature);
    if (temperature.size() == 1) {
        float tp = temperature.getVal<float>();
        deviceFill(temperature_buf_, batch_size, tp, stream_);
        std::fill_n(temperature_, batch_size, tp);
    }
    else {
        cudaAutoCpy(temperature_buf_, temperature.getPtr<float>(), batch_size, stream_);
        std::copy_n(temperature.getPtr<float>(), batch_size, temperature_);
    }

    if (runtime_args->isExist("repetition_penalty") || runtime_args->isExist("presence_penalty")) {
        FT_CHECK_WITH_INFO(
            !(runtime_args->isExist("repetition_penalty") && runtime_args->isExist("presence_penalty")),
            "Found ambiguous parameters repetition_penalty and presence_penalty which are mutually exclusive. "
            "Please provide one of repetition_penalty or presence_penalty.");
        repetition_penalty_type_ = runtime_args->isExist("repetition_penalty") ? RepetitionPenaltyType::Multiplicative :
                                                                                 RepetitionPenaltyType::Additive;
        Tensor repetition_penalty = repetition_penalty_type_ == RepetitionPenaltyType::Multiplicative ?
                                        runtime_args->at("repetition_penalty") :
                                        runtime_args->at("presence_penalty");
        if (repetition_penalty.size() == 1) {
            float rp = repetition_penalty.getVal<float>();
            deviceFill(repetition_penalty_buf_, batch_size, rp, stream_);
            std::fill_n(repetition_penalty_, batch_size, rp);
        }
        else {
            cudaAutoCpy(repetition_penalty_buf_, repetition_penalty.getPtr<float>(), batch_size, stream_);
            std::copy_n(repetition_penalty.getPtr<float>(), batch_size, repetition_penalty_);
        }
    }
    else {
        repetition_penalty_type_ = RepetitionPenaltyType::None;
    }

    const int default_min_length = 0;
    Tensor    min_lengths = runtime_args->at("min_length", Tensor(MEMORY_CPU, TYPE_INT32, {1}, &default_min_length));
    if (min_lengths.size() == 1) {
        int minlen = min_lengths.getVal<int>();
        deviceFill(min_lengths_buf_, batch_size, minlen, stream_);
        std::fill_n(min_lengths_, batch_size, minlen);
    }
    else {
        cudaAutoCpy(min_lengths_buf_, min_lengths.getPtr<int>(), batch_size, stream_);
        std::copy_n(min_lengths.getPtr<int>(), batch_size, min_lengths_);
    }
}

template<typename T>
void BaseSamplingLayer<T>::forward(std::vector<Tensor>* output_tensors, const std::vector<Tensor>* input_tensors)
{
    // input_tensors:
    //      logits [local_batch_size, vocab_size_padded]
    //      embedding_bias [vocab_size_padded]
    //      step [1] on cpu
    //      max_input_length [1] on cpu
    //      input_lengths [local_batch_size]
    //      ite [1] on cpu
    //      random_seed [1] on cpu, optional

    // output_tensors:
    //      output_ids [max_seq_len, batch_size]
    //      finished [local_batch_size]
    //      sequence_length [local_batch_size]
    //      cum_log_probs [local_batch_size], must be float*

    FT_CHECK(false);  // TODO deprecated, need to remove
    std::unordered_map<std::string, Tensor> input_tensors_map{{"logits", input_tensors->at(0)},
                                                              {"embedding_bias", input_tensors->at(1)},
                                                              {"step", input_tensors->at(2)},
                                                              {"max_input_length", input_tensors->at(3)},
                                                              {"input_lengths", input_tensors->at(4)},
                                                              {"ite", input_tensors->at(5)}};
    if (input_tensors->size() == 7) {
        input_tensors_map.insert({"random_seed", input_tensors->at(6)});
    }

    std::unordered_map<std::string, Tensor> output_tensors_map{{"output_ids", output_tensors->at(0)},
                                                               {"finished", output_tensors->at(1)},
                                                               {"sequence_length", output_tensors->at(2)},
                                                               {"cum_log_probs", output_tensors->at(3)}};
    forward(&output_tensors_map, &input_tensors_map);
}

template<typename T>
void BaseSamplingLayer<T>::forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                                   const std::unordered_map<std::string, Tensor>* input_tensors)
{
    TM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    TensorMap input_map(*input_tensors);
    TensorMap output_map(*output_tensors);
    forward(&output_map, &input_map);
}

template<typename T>
void BaseSamplingLayer<T>::forward(TensorMap* output_tensors, TensorMap* input_tensors)
{
    // input_tensors:
    //      logits [local_batch_size, vocab_size_padded]
    //      embedding_bias [vocab_size_padded], optional
    //      step [1] on cpu
    //      max_input_length [1] on cpu
    //      input_lengths [local_batch_size], optional
    //      ite [1] on cpu
    //      end_id [local_batch_size], optional

    // output_tensors:
    //      output_ids [max_seq_len, batch_size]
    //      finished [local_batch_size], optional
    //      sequence_length [local_batch_size], optional
    //      cum_log_probs [batch_size], must be float*, optional
    //          The cumultative log probability of generated tokens.
    //      output_log_probs [local_batch_size], must be float*, optional
    //          The log probs at the current step.

    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() >= 4);
    FT_CHECK(output_tensors->size() >= 1);
    const int batch_size       = output_tensors->at("output_ids").shape[1];
    const int local_batch_size = input_tensors->at("logits").shape[0];
    const int step             = input_tensors->at("step").getVal<int>();
    const int ite              = input_tensors->at("ite").getVal<int>();
    const int max_input_length = input_tensors->at("max_input_length").getVal<int>();
    T*        logits           = input_tensors->at("logits").getPtr<T>();

#define ALL_OF(p_, sz_, dt_, v_) (std::all_of(p_, p_ + sz_, [&](dt_ b) { return b == v_; }))

    bool* skip_decode = skip_decode_ + ite * local_batch_size;
    if (ALL_OF(skip_decode, local_batch_size, bool, true)) {
        // No sample in the current batch to do TopX sampling.
        return;
    }
    skip_any_ = std::any_of(skip_decode, skip_decode + local_batch_size, [](bool b) { return b; });
    if (skip_any_) {
        // A TopX Sampling layer directly changes the logit values. In case of skip_any==true,
        // meaning topk and topp layers will run simultaneously for a batch in the same step.
        // We copy the logits to an internal buffer, not affecting the other sampling layers.
        FT_CHECK(input_tensors->at("logits").size() == local_batch_size * vocab_size_padded_);
        cudaD2Dcpy(runtime_logits_buf_, logits, input_tensors->at("logits").size());
        logits = runtime_logits_buf_;
    }

    const T* embedding_bias =
        input_tensors->isExist("embedding_bias") ? input_tensors->at("embedding_bias").getPtr<T>() : nullptr;
    if (embedding_bias != nullptr || !ALL_OF(temperature_ + ite * local_batch_size, local_batch_size, float, 1.0f)) {
        invokeBatchApplyTemperaturePenalty(logits,
                                           embedding_bias,
                                           temperature_buf_ + ite * local_batch_size,
                                           local_batch_size,
                                           vocab_size_,
                                           vocab_size_padded_,
                                           stream_);
    }
    sync_check_cuda_error();

    if (step > 1 && repetition_penalty_type_ != RepetitionPenaltyType::None) {
        float default_value = getDefaultPenaltyValue(repetition_penalty_type_);
        if (!ALL_OF(repetition_penalty_ + ite * local_batch_size, local_batch_size, float, default_value)) {
            invokeBatchApplyRepetitionPenalty(
                logits,
                repetition_penalty_buf_ + ite * local_batch_size,
                output_tensors->at("output_ids").getPtrWithOffset<int>(ite * local_batch_size),
                batch_size,
                local_batch_size,
                vocab_size_padded_,
                input_tensors->at("input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {}, nullptr}).getPtr<int>(),
                max_input_length,
                step,
                repetition_penalty_type_,
                stream_);
            sync_check_cuda_error();
        }
    }

    const int  num_generated_tokens      = step - max_input_length;
    const int* min_lengths               = min_lengths_ + ite * local_batch_size;
    const bool invoke_min_length_penalty = std::any_of(
        min_lengths, min_lengths + local_batch_size, [&](int min_length) { return min_length > num_generated_tokens; });
    if (invoke_min_length_penalty) {
        FT_CHECK_WITH_INFO(input_tensors->isExist("end_id"), "Need end_id to apply min length penlaty");
        invokeMinLengthPenalty(logits,
                               min_lengths_buf_ + ite * local_batch_size,
                               input_tensors->getPtr<const int>("end_id"),
                               output_tensors->getPtr<const int>("sequence_length"),
                               max_input_length,
                               local_batch_size,
                               vocab_size_padded_,
                               stream_);
        sync_check_cuda_error();
    }
#undef ALL_OF

    runSampling(output_tensors, input_tensors);

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
    sync_check_cuda_error();
    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template class BaseSamplingLayer<float>;
template class BaseSamplingLayer<half>;

}  // namespace turbomind
