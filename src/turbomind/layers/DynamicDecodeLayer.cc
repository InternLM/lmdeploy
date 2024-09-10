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
#include "src/turbomind/layers/sampling_layers/LogitsProcessorLayer.h"
#include "src/turbomind/layers/sampling_layers/SamplingLayer.h"
#include "src/turbomind/layers/sampling_layers/StopCriteriaLayer.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<typename T>
void DynamicDecodeLayer<T>::allocateBuffer()
{
}

template<typename T>
void DynamicDecodeLayer<T>::freeBuffer()
{
}

template<typename T>
void DynamicDecodeLayer<T>::initialize()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    DynamicDecodeCommonArgs args{vocab_size_, vocab_size_padded_};
    layers_.emplace_back(new LogitsProcessorLayer<T>(stream_, allocator_, is_free_buffer_after_forward_, args));
    layers_.emplace_back(new SamplingLayer<T>(stream_, allocator_, is_free_buffer_after_forward_, args));
    layers_.emplace_back(new StopCriteriaLayer<T>(stream_, allocator_, is_free_buffer_after_forward_, args));
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
     *   \param  repetition_penalty [batch_size] on cpu, optional
     *   \param  min_length [batch_size], optional
     *   \param  context_length [batch_size], optional
     *   \param  prompt_length [batch_size], optional
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
     *   \param  step [1] on cpu
     *   \param  max_input_length [1] on cpu
     *   \param  input_lengths [batch_size, beam_width], optional
     *   \param  sequence_limit_length [batch_size]
     *   \param  ite [1] on cpu
     *   \param  local_batch_size [1] on cpu
     *   \param  stop_words_list [batch_size, 2, stop_words_length], optional
     *   \param  runtime_top_k [batch_size] on cpu, optional, uint
     *   \param  runtime_top_p [batch_size] on cpu, optional, float
     *   \param  temperature [batch_size] on cpu, optional, float
     *   \param  repetition_penalty [batch_size] on cpu, optional, float
     *   \param  bad_words_list [batch_size, 2, bad_words_length], optional
     *
     * output_tensors:
     *   \param  output_ids [max_seq_len, batch_size, 1]
     *   \param  curand_state [local_batch_size]
     *   \param  finished [batch_size * beam_width], optional
     *   \param  sequence_length [batch_size * beam_width], optional
     *   \param  sampled_indexes [batch_size, 1, kMaxLogProb], optional
     *   \param  sampled_logprobs [batch_size, 1, kMaxLogProb], optional
     *   \param  sampled_nums [batch_size, 1], optional
     */

    const int    ite              = (int)input_tensors->at("ite").getVal<uint>();
    const size_t batch_size       = input_tensors->at("logits").shape[0];
    const size_t local_batch_size = (size_t)input_tensors->at("local_batch_size").getVal<int>();

    FT_CHECK(ite == 0);
    FT_CHECK(local_batch_size == batch_size);
    FT_CHECK(input_tensors->at("logits").shape.size() == 3);

    for (const auto& layer : layers_) {
        layer->forward(output_tensors, input_tensors);
    }
}

template class DynamicDecodeLayer<float>;
// template class DynamicDecodeLayer<half>;

}  // namespace turbomind
