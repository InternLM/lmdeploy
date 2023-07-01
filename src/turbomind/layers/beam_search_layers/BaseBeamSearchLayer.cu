/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/turbomind/kernels/beam_search_penalty_kernels.h"
#include "src/turbomind/layers/beam_search_layers/BaseBeamSearchLayer.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

__global__ void update_indir_cache_kernel(int*        tgt_indir_cache,
                                          const int*  src_indir_cache,
                                          const int*  beam_ids,
                                          const bool* finished,
                                          int         start_step,
                                          int         batch_dim,
                                          int         local_batch_size,
                                          int         beam_width,
                                          int         max_seq_len,
                                          int         step)
{
    int       time_step = threadIdx.x + blockIdx.x * blockDim.x;
    int       bb_id     = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch_id  = bb_id / beam_width;
    const int beam_id   = bb_id % beam_width;

    if (bb_id >= beam_width * local_batch_size || time_step >= min(step + 1, max_seq_len) || finished[bb_id]) {
        return;
    }
    time_step += start_step;
    const int time_step_circ = time_step % max_seq_len;

    const int src_beam = beam_ids[batch_id * beam_width + beam_id];

    const uint tgt_offset = batch_id * beam_width * max_seq_len + beam_id * max_seq_len + time_step_circ;
    const uint src_offset = batch_id * beam_width * max_seq_len + src_beam * max_seq_len + time_step_circ;

    tgt_indir_cache[tgt_offset] = (time_step == step) ? beam_id : src_indir_cache[src_offset];
}

void update_indir_cache_kernelLauncher(int*         tgt_indir_cache,
                                       const int*   src_indir_cache,
                                       const int*   beam_ids,
                                       const bool*  finished,
                                       int          batch_dim,
                                       int          local_batch_size,
                                       int          beam_width,
                                       int          max_seq_len,
                                       int          step,
                                       cudaStream_t stream)
{
    const dim3 block(32);
    const int  start_step = max(0, step + 1 - max_seq_len);
    const int  num_steps  = min(step + 1, max_seq_len);
    // Update indirections steps [start_step, step], included
    const dim3 grid((num_steps + block.x - 1) / block.x, local_batch_size * beam_width);
    update_indir_cache_kernel<<<grid, block, 0, stream>>>(tgt_indir_cache,
                                                          src_indir_cache,
                                                          beam_ids,
                                                          finished,
                                                          start_step,
                                                          batch_dim,
                                                          local_batch_size,
                                                          beam_width,
                                                          max_seq_len,
                                                          step);
}

template<typename T>
BaseBeamSearchLayer<T>::BaseBeamSearchLayer(size_t           max_batch_size,
                                            size_t           head_num,
                                            size_t           size_per_head,
                                            size_t           beam_width,
                                            size_t           vocab_size,
                                            size_t           vocab_size_padded,
                                            int              end_id,
                                            float            diversity_rate,
                                            float            temperature,
                                            float            len_penalty,
                                            float            repetition_penalty,
                                            cudaStream_t     stream,
                                            cublasMMWrapper* cublas_wrapper,
                                            IAllocator*      allocator,
                                            bool             is_free_buffer_after_forward):
    DynamicDecodeBaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr),
    vocab_size_(vocab_size),
    vocab_size_padded_(vocab_size_padded)
{
}

template<typename T>
BaseBeamSearchLayer<T>::BaseBeamSearchLayer(BaseBeamSearchLayer<T> const& beam_search_layer):
    DynamicDecodeBaseLayer(beam_search_layer),
    vocab_size_(beam_search_layer.vocab_size_),
    vocab_size_padded_(beam_search_layer.vocab_size_padded_),
    topk_softmax_workspace_size_(beam_search_layer.topk_softmax_workspace_size_)
{
}

template<typename T>
BaseBeamSearchLayer<T>::~BaseBeamSearchLayer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    freeBuffer();
}

template<typename T>
void BaseBeamSearchLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&topk_softmax_workspace_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void BaseBeamSearchLayer<T>::setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args)
{
    // do nothing.
}

template<typename T>
void BaseBeamSearchLayer<T>::forward(std::vector<Tensor>* output_tensors, const std::vector<Tensor>* input_tensors)
{
    // input_tensors:
    //      logits [local_batch_size, beam_width, vocab_size_padded]
    //      embedding_bias [vocab_size_padded]
    //      step [1] on cpu
    //      src_cache_indirection [local_batch_size, beam_width, max_seq_len]
    //      max_input_length [1] on cpu
    //      input_lengths [local_batch_size * beam_width]
    //      ite [1] on cpu

    // output_tensors:
    //      output_ids [max_seq_len, batch_size, beam_width]
    //      finished [local_batch_size * beam_width]
    //      cum_log_probs [local_batch_size * beam_width]
    //      parent_ids [max_seq_len, batch_size * beam_width]
    //      sequence_length [local_batch_size * beam_width]
    //      tgt_cache_indirection [local_batch_size, beam_width, max_seq_len]

    std::unordered_map<std::string, Tensor> input_tensors_map{{"logits", input_tensors->at(0)},
                                                              {"embedding_bias", input_tensors->at(1)},
                                                              {"step", input_tensors->at(2)},
                                                              {"src_cache_indirection", input_tensors->at(4)},
                                                              {"max_input_length", input_tensors->at(5)},
                                                              {"input_lengths", input_tensors->at(6)},
                                                              {"ite", input_tensors->at(7)}};

    std::unordered_map<std::string, Tensor> output_tensors_map{{"output_ids", output_tensors->at(0)},
                                                               {"finished", output_tensors->at(1)},
                                                               {"cum_log_probs", output_tensors->at(2)},
                                                               {"parent_ids", output_tensors->at(3)},
                                                               {"sequence_length", output_tensors->at(4)},
                                                               {"tgt_cache_indirection", output_tensors->at(5)}};
    forward(&output_tensors_map, &input_tensors_map);
}

template<typename T>
void BaseBeamSearchLayer<T>::forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                                     const std::unordered_map<std::string, Tensor>* input_tensors)
{
    TensorMap input_map(*input_tensors);
    TensorMap output_map(*output_tensors);
    forward(&output_map, &input_map);
}

template<typename T>
void BaseBeamSearchLayer<T>::forward(TensorMap* output_tensors, TensorMap* input_tensors)
{
    // input_tensors:
    //      logits [local_batch_size, beam_width, vocab_size_padded]
    //      embedding_bias [vocab_size_padded]
    //      step [1] on cpu
    //      src_cache_indirection [local_batch_size, beam_width, max_seq_len]
    //      end_id [local_batch_size]
    //      max_input_length [1] on cpu
    //      input_lengths [local_batch_size * beam_width], optional
    //      ite [1] on cpu
    //      beam_search_diversity_rate [1] on cpu, optional
    //      temperature [1] on cpu, optional
    //      len_penalty [1] on cpu, optional
    //      repetition_penalty [1] on cpu, optional
    //      presence_penalty [1] on cpu, optional
    //          Only one of repetition and presence penalties is allowed.
    //      min_length [1] on cpu, int, optional

    // output_tensors:
    //      output_ids [max_seq_len, batch_size, beam_width]
    //      finished [local_batch_size * beam_width], optional
    //      cum_log_probs [local_batch_size * beam_width]
    //      parent_ids [max_seq_len, batch_size * beam_width]
    //      sequence_length [local_batch_size * beam_width], optional
    //      tgt_cache_indirection [local_batch_size, beam_width, max_seq_len]
    //      output_log_probs [max_seq_len, batch_size, beam_width], optional
    //      beam_hyps, optional

    FT_CHECK(input_tensors->size() >= 7);
    FT_CHECK(output_tensors->size() >= 5);
    const int batch_size = output_tensors->at("output_ids").shape[1];
    const int beam_width = output_tensors->at("output_ids").shape[2];
    allocateBuffer(batch_size, beam_width);

    const int step             = input_tensors->at("step").getVal<int>();
    const int ite              = input_tensors->at("ite").getVal<int>();
    const int local_batch_size = input_tensors->at("logits").shape[0];

    const float temperature    = input_tensors->getVal<float>("temperature", 1.0f);
    const T*    embedding_bias = input_tensors->getPtr<const T>("embedding_bias", nullptr);

    RepetitionPenaltyType repetition_penalty_type = RepetitionPenaltyType::None;
    float                 repetition_penalty      = getDefaultPenaltyValue(repetition_penalty_type);
    if (input_tensors->isExist("repetition_penalty") || input_tensors->isExist("presence_penalty")) {
        FT_CHECK_WITH_INFO(
            !(input_tensors->isExist("repetition_penalty") && input_tensors->isExist("presence_penalty")),
            "Found ambiguous parameters repetition_penalty and presence_penalty which are mutually exclusive. "
            "Please provide one of repetition_penalty or presence_penalty.");
        repetition_penalty_type = input_tensors->isExist("repetition_penalty") ? RepetitionPenaltyType::Multiplicative :
                                                                                 RepetitionPenaltyType::Additive;
        repetition_penalty      = repetition_penalty_type == RepetitionPenaltyType::Multiplicative ?
                                      input_tensors->getVal<float>("repetition_penalty") :
                                      input_tensors->getVal<float>("presence_penalty");
    }

    invokeAddBiasApplyPenalties(
        step,
        input_tensors->at("logits").getPtr<T>(),
        output_tensors->at("output_ids")
            .getPtrWithOffset<const int>((step - 1) * batch_size * beam_width + ite * local_batch_size * beam_width),
        output_tensors->getPtr<const int>("output_ids"),
        output_tensors->getPtr<const int>("parent_ids"),
        input_tensors->getPtr<const int>("input_lengths", nullptr),
        output_tensors->getPtr<const int>("sequence_length", nullptr),
        embedding_bias,
        ite,
        input_tensors->getVal<int>("max_input_length"),
        local_batch_size,
        batch_size,
        beam_width,
        vocab_size_,
        vocab_size_padded_,
        input_tensors->getPtr<const int>("end_id", nullptr),
        temperature,
        repetition_penalty,
        repetition_penalty_type,
        input_tensors->getVal<const int>("min_length", 0),
        stream_);
    sync_check_cuda_error();

    invokeSoftMax(output_tensors, input_tensors);

    if (beam_width > 1) {
        const int max_seq_len = output_tensors->at("output_ids").shape[0];

        update_indir_cache_kernelLauncher(
            output_tensors->at("tgt_cache_indirection").getPtr<int>(),
            input_tensors->at("src_cache_indirection").getPtr<const int>(),
            output_tensors->at("parent_ids")
                .getPtrWithOffset<const int>(+step * beam_width * batch_size + ite * local_batch_size * beam_width),
            output_tensors->at("finished").getPtr<const bool>(),
            batch_size,
            local_batch_size,
            beam_width,
            max_seq_len,
            step,
            stream_);
        sync_check_cuda_error();
    }
    sync_check_cuda_error();
    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template class BaseBeamSearchLayer<float>;
template class BaseBeamSearchLayer<half>;

}  // namespace turbomind
