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
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/layers/BaseDynamicDecodeLayer.h"
#include "src/turbomind/layers/sampling_layers/GuidedDecodeMaskLayer.h"
#include "src/turbomind/layers/sampling_layers/GuidedDecodeUpdateLayer.h"
#include "src/turbomind/layers/sampling_layers/LogitsProcessorLayer.h"
#include "src/turbomind/layers/sampling_layers/SamplingLayer.h"
#include "src/turbomind/layers/sampling_layers/StopCriteriaLayer.h"
#include "src/turbomind/macro.h"

namespace turbomind {

DynamicDecodeLayer::DynamicDecodeLayer(DataType              dtype,
                                       int                   max_batch_size,
                                       int                   vocab_size,
                                       int                   vocab_size_padded,
                                       cudaStream_t          stream,
                                       const cudaDeviceProp* device_prop)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    TM_CHECK(dtype == kFloat32);
    BaseDynamicDecodeLayer::BaseParam param{max_batch_size, vocab_size, vocab_size_padded, stream, device_prop};
    layers_.emplace_back(new LogitsProcessorLayer<float>{param});
    layers_.emplace_back(new GuidedDecodeMaskLayer<float>{param});
    layers_.emplace_back(new SamplingLayer<float>{param});
    layers_.emplace_back(new GuidedDecodeUpdateLayer<float>{param});
    layers_.emplace_back(new StopCriteriaLayer<float>{param});
}

DynamicDecodeLayer::~DynamicDecodeLayer() {}

void DynamicDecodeLayer::Setup(const std::vector<const Request*>& rs, const TensorMap& args)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    for (const auto& layer : layers_) {
        layer->Setup(rs, args);
    }
}

void DynamicDecodeLayer::Forward(TensorMap& args)
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

    for (const auto& layer : layers_) {
        layer->Forward(args);
    }
}

}  // namespace turbomind
