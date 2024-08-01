/*
 * Copyright (c) 2022-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/turbomind/layers/DynamicDecodeLayer.h"
#include "src/turbomind/layers/sampling_layers/BanBadWordsLayer.h"
#include "src/turbomind/layers/sampling_layers/PenaltyLayer.h"
#include "src/turbomind/layers/sampling_layers/SamplingLayer.h"
#include "src/turbomind/layers/sampling_layers/StopCriteriaLayer.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<typename T>
void DynamicDecodeLayer<T>::allocateBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void DynamicDecodeLayer<T>::freeBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void DynamicDecodeLayer<T>::initialize()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    DynamicDecodeCommonArgs args{vocab_size_, vocab_size_padded_};
    layers_.emplace_back(new BanBadWordsLayer<T>(stream_, allocator_, args));
    layers_.emplace_back(new PenaltyLayer<T>(stream_, allocator_, args));
    layers_.emplace_back(new SamplingLayer<T>(stream_, allocator_, args));
    layers_.emplace_back(new StopCriteriaLayer<T>(stream_, allocator_, args));
}

template<typename T>
DynamicDecodeLayer<T>::DynamicDecodeLayer(size_t           vocab_size,
                                          size_t           vocab_size_padded,
                                          int              end_id,
                                          cudaStream_t     stream,
                                          cublasMMWrapper* cublas_wrapper,
                                          IAllocator*      allocator,
                                          bool             is_free_buffer_after_forward,
                                          cudaDeviceProp*  cuda_device_prop):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    vocab_size_(vocab_size),
    vocab_size_padded_(vocab_size_padded),
    cuda_device_prop_(cuda_device_prop)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    initialize();
}

template<typename T>
DynamicDecodeLayer<T>::~DynamicDecodeLayer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
DynamicDecodeLayer<T>::DynamicDecodeLayer(DynamicDecodeLayer const& dynamic_decode_layer):
    BaseLayer(dynamic_decode_layer),
    vocab_size_(dynamic_decode_layer.vocab_size_),
    vocab_size_padded_(dynamic_decode_layer.vocab_size_padded_),
    cuda_device_prop_(dynamic_decode_layer.cuda_device_prop_)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    initialize();
}

template<typename T>
void DynamicDecodeLayer<T>::setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args)
{
    /**
     * @brief Set up the dynamic decode layer for given input runtime arguments.
     *
     * runtime_args:
     *   \param  runtime_top_k [batch_size] on cpu, optional.
     *   \param  runtime_top_p [batch_size] on cpu, optional
     *   \param  temperature [batch_size] on cpu, optional
     *   \param  len_penalty [batch_size] on cpu, optional
     *   \param  repetition_penalty [batch_size] on cpu, optional
     *   \param  presence_penalty [batch_size] on cpu, optional, float
     *   \param  min_length [batch_size], optional
     */

    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    FT_CHECK_WITH_INFO(beam_width == 1, "only support beam_width=1");
    for (const auto& layer : layers_) {
        layer->setup(batch_size, beam_width, runtime_args);
    }
}

template<typename T>
void DynamicDecodeLayer<T>::forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                                    const std::unordered_map<std::string, Tensor>* input_tensors)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap input_map(*input_tensors);
    TensorMap output_map(*output_tensors);
    forward(&output_map, &input_map);
}

template<typename T>
void DynamicDecodeLayer<T>::forward(TensorMap* output_tensors, TensorMap* input_tensors)
{
    /**
     * @brief
     * input_tensors:
     *   \param  logits [batch_size, beam_width, vocab_size_padded]
     *   \param  embedding_bias [vocab_size_padded], optional
     *   \param  step [1] on cpu
     *   \param  max_input_length [1] on cpu
     *   \param  input_lengths [batch_size, beam_width], optional
     *   \param  min_length [batch_size], optional
     *   \param  sequence_limit_length [batch_size]
     *   \param  ite [1] on cpu
     *   \param  local_batch_size [1] on cpu
     *   \param  stop_words_list [batch_size, 2, stop_words_length], optional
     *   \param  runtime_top_k [1] or [batch_size] on cpu, optional, uint
     *   \param  runtime_top_p [1] or [batch_size] on cpu, optional, float
     *   \param  temperature [1] or [batch_size] on cpu, optional, float
     *   \param  len_penalty [1] or [batch_size] on cpu, optional, float
     *   \param  repetition_penalty [1] or [batch_size] on cpu, optional, float
     *   \param  presence_penalty [1] or [batch_size] on cpu, optional, float
     *                Only one of repetition and presence penalties is allowed.
     *   \param  random_seed [1] or [batch_size] on cpu, optional, unsigned long long int
     *   \param  bad_words_list [2, bad_words_length] or [batch_size, 2, bad_words_length], optional
     *   \param  src_cache_indirection
     *                [local_batch_size, beam_width, max_seq_len]
     *                the k/v cache index for beam search
     *   \param  is_initialize_random_table [1] on cpu, bool
     *   \param  top_p_decay [batch_size] on gpu, float, optional
     *   \param  top_p_min [batch_size] on gpu, float, optional
     *   \param  top_p_reset_ids [batch_size] on gpu, uint32, optional
     *
     * output_tensors:
     *   \param  output_ids [max_seq_len, batch_size]
     *   \param  curand_state [local_batch_size]
     *   \param  finished [batch_size * beam_width], optional
     *   \param  should_stop [1] on cpu
     *   \param  cum_log_probs [batch_size * beam_width], necessary in beam search
     *   \param  parent_ids [max_seq_len, batch_size * beam_width]
     *   \param  sequence_length [batch_size * beam_width], optional
     *   \param  output_log_probs [request_ouptut_length, batch_size * beam_width], must be float*, optional
     *   \param  tgt_cache_indirection
     *                [local_batch_size, beam_width, max_seq_len]
     *                the k/v cache index for beam search
     *   \param  beam_hyps: [1] on cpu, a special structure which maintains some pointers of beam search
     *
     */

    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const size_t beam_width = input_tensors->at("logits").shape[1];
    FT_CHECK_WITH_INFO(beam_width == 1, "Beam-search is not supported.");
    FT_CHECK(input_tensors->at("logits").shape.size() == 3);

    const int    ite              = (int)input_tensors->at("ite").getVal<uint>();
    const int    step             = input_tensors->at("step").getVal<int>();
    const size_t batch_size       = input_tensors->at("logits").shape[0];
    const size_t local_batch_size = (size_t)input_tensors->at("local_batch_size").getVal<int>();
    FT_CHECK(batch_size == local_batch_size);
    FT_CHECK(ite == 0);

    const size_t local_batch_offset = ite * local_batch_size * beam_width;

    Tensor logits = input_tensors->at("logits");
    Tensor end_id = input_tensors->at("end_id");

    TensorMap decode_input_tensors(
        {{"logits",
          logits.slice({local_batch_size, beam_width, logits.shape[2]}, local_batch_offset * logits.shape[2])},
         {"step", input_tensors->at("step")},
         {"max_input_length", input_tensors->at("max_input_length")},
         {"end_id", end_id.slice({local_batch_size}, ite * local_batch_size)},
         {"ite", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &ite}}});

    if (input_tensors->isExist("input_lengths")) {
        Tensor input_lengths = input_tensors->at("input_lengths");
        decode_input_tensors.insert(
            {"input_lengths", input_lengths.slice({local_batch_size, beam_width}, local_batch_offset)});
    }

    TensorMap decode_output_tensors(
        {{"output_ids", output_tensors->at("output_ids")}, {"curand_state", output_tensors->at("curand_state")}});
    if (output_tensors->isExist("sequence_length")) {
        Tensor sequence_length = output_tensors->at("sequence_length");
        decode_output_tensors.insert(
            {"sequence_length", sequence_length.slice({local_batch_size * beam_width}, local_batch_offset)});
    }
    if (output_tensors->isExist("finished")) {
        Tensor finished = output_tensors->at("finished");
        decode_output_tensors.insert({"finished", finished.slice({local_batch_size * beam_width}, local_batch_offset)});
    }

    if (output_tensors->isExist("sampled_logprobs")) {
        Tensor sampled_logprobs = output_tensors->at("sampled_logprobs");
        decode_output_tensors.insert({"sampled_logprobs",
                                      sampled_logprobs.slice({local_batch_size, beam_width, sampled_logprobs.shape[2]},
                                                             local_batch_offset * sampled_logprobs.shape[2])});
    }
    if (output_tensors->isExist("sampled_indexes")) {
        Tensor sampled_indexes = output_tensors->at("sampled_indexes");
        decode_output_tensors.insert({"sampled_indexes",
                                      sampled_indexes.slice({local_batch_size, beam_width, sampled_indexes.shape[2]},
                                                            local_batch_offset * sampled_indexes.shape[2])});
    }
    if (output_tensors->isExist("sampled_nums")) {
        Tensor sampled_nums = output_tensors->at("sampled_nums");
        decode_output_tensors.insert(
            {"sampled_nums", sampled_nums.slice({local_batch_size, beam_width}, local_batch_offset)});
    }

    for (const auto& layer : layers_) {
        layer->forward(output_tensors, input_tensors);
    }
}

template class DynamicDecodeLayer<float>;
// template class DynamicDecodeLayer<half>;

}  // namespace turbomind
