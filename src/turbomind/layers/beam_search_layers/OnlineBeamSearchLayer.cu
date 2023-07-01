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

#include "src/turbomind/layers/beam_search_layers/OnlineBeamSearchLayer.h"

namespace turbomind {

static const int SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS = 128;
static const int MAX_K                             = 4;

template<typename T>
__global__ void update_kernel(bool*          finished,
                              int*           parent_ids,
                              int*           sequence_length,
                              int*           word_ids,
                              int*           output_ids,
                              BeamHypotheses beam_hyps,
                              const int      vocab_size,
                              const int*     end_ids,
                              const int      local_batch_size,
                              const int      beam_width)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < local_batch_size * beam_width;
         index += blockDim.x * gridDim.x) {

        int batch_id           = index / beam_width;
        sequence_length[index] = finished[index] ? sequence_length[index] : sequence_length[index] + 1;

        int beam_id = (word_ids[index] / vocab_size) % beam_width;
        int word_id = word_ids[index] % vocab_size;

        sequence_length[index] = sequence_length[batch_id * beam_width + beam_id];
        finished[index]        = word_id == end_ids[index / beam_width] ? 1 : 0;
        parent_ids[index]      = beam_id;
        word_ids[index]        = word_id;
        output_ids[index]      = word_id;

        if (beam_hyps.num_beams != nullptr) {
            if (beam_hyps.num_beams[beam_hyps.ite * beam_hyps.local_batch_size + batch_id] == beam_width) {
                for (int i = 0; i < beam_width; i++) {
                    finished[batch_id * beam_width + i] = true;
                }
            }
        }
    }
}

void invokeUpdate(bool*           finished,
                  int*            parent_ids,
                  int*            sequence_length,
                  int*            word_ids,
                  int*            output_ids,
                  BeamHypotheses* beam_hyps,
                  const int       local_batch_size,
                  const int       beam_width,
                  const int       vocab_size_padded,
                  const int*      end_ids,
                  cudaStream_t    stream)
{
    dim3 grid((int)ceil(local_batch_size * beam_width * 1.0 / 256));
    dim3 block(256);

    update_kernel<float><<<grid, block, 0, stream>>>(finished,
                                                     parent_ids,
                                                     sequence_length,
                                                     word_ids,
                                                     output_ids,
                                                     *beam_hyps,
                                                     vocab_size_padded,
                                                     end_ids,
                                                     local_batch_size,
                                                     beam_width);
}

template<typename T>
void OnlineBeamSearchLayer<T>::invokeSoftMax(TensorMap* output_tensors, TensorMap* input_tensors)
{
    // input_tensors:
    //      logits [local_batch_size, beam_width, vocab_size_padded]
    //      embedding_bias [vocab_size_padded]
    //      step [1] on cpu
    //      src_cache_indirection [local_batch_size, beam_width, max_seq_len]
    //      max_input_length [1] on cpu
    //      input_lengths [local_batch_size * beam_width]
    //      ite [1] on cpu
    //      beam_search_diversity_rate [1] on cpu, optional
    //      temperature [1] on cpu, optional
    //      len_penalty [1] on cpu, optional
    //      repetition_penalty [1] on cpu, optional

    // output_tensors:
    //      output_ids [max_seq_len, batch_size, beam_width]
    //      finished [local_batch_size * beam_width]
    //      cum_log_probs [local_batch_size * beam_width]
    //      parent_ids [max_seq_len, batch_size * beam_width]
    //      sequence_length [local_batch_size * beam_width]
    //      tgt_cache_indirection [local_batch_size, beam_width, max_seq_len]
    //      output_log_probs [max_seq_len, batch_size, beam_width]

    FT_CHECK(input_tensors->size() >= 7);
    FT_CHECK(output_tensors->size() >= 6);

    const int   batch_size       = output_tensors->at("output_ids").shape[1];
    const int   beam_width       = output_tensors->at("output_ids").shape[2];
    const int   step             = input_tensors->at("step").getVal<int>();
    const int   ite              = input_tensors->at("ite").getVal<int>();
    const int   local_batch_size = input_tensors->at("logits").shape[0];
    const float diversity_rate   = input_tensors->isExist("beam_search_diversity_rate") ?
                                       input_tensors->at("beam_search_diversity_rate").getVal<float>() :
                                       0.0f;
    const float length_penalty =
        input_tensors->isExist("len_penalty") ? input_tensors->at("len_penalty").getVal<float>() : 0.0f;

    const int id_offset = step * batch_size * beam_width + local_batch_size * ite * beam_width;

    BeamHypotheses beam_hyps;
    if (output_tensors->isExist("beam_hyps")) {
        beam_hyps                      = *((BeamHypotheses*)(output_tensors->at("beam_hyps").getPtr<void>()));
        beam_hyps.step                 = step;
        beam_hyps.ite                  = ite;
        beam_hyps.local_batch_size     = local_batch_size;
        beam_hyps.batch_size           = output_tensors->at("output_ids").shape[1];
        beam_hyps.max_seq_len          = output_tensors->at("output_ids").shape[0];
        beam_hyps.output_ids_src       = output_tensors->at("output_ids").getPtr<int>();
        beam_hyps.parent_ids_src       = output_tensors->at("parent_ids").getPtr<int>();
        beam_hyps.sequence_lengths_src = output_tensors->at("sequence_length").getPtr<int>();
        beam_hyps.log_probs_src        = output_tensors->getPtr<float>("output_log_probs", nullptr);
        beam_hyps.length_penalty       = length_penalty;
        beam_hyps.end_ids              = input_tensors->at("end_id").getPtr<int>();
    }

    invokeTopkSoftMax(input_tensors->at("logits").getPtr<T>(),
                      (const T*)(nullptr),
                      output_tensors->at("finished").getPtr<bool>(),
                      output_tensors->at("sequence_length").getPtr<int>(),
                      output_tensors->at("cum_log_probs").getPtr<float>(),
                      output_tensors->getPtrWithOffset<float>("output_log_probs", id_offset, nullptr),
                      output_tensors->at("output_ids").getPtrWithOffset<int>(id_offset),
                      topk_softmax_workspace_,
                      topk_softmax_workspace_size_,
                      &beam_hyps,
                      local_batch_size,
                      beam_width,
                      vocab_size_padded_,
                      input_tensors->at("end_id").getPtr<int>(),
                      diversity_rate,
                      length_penalty,
                      stream_);
    sync_check_cuda_error();

    invokeUpdate(output_tensors->at("finished").getPtr<bool>(),
                 output_tensors->at("parent_ids").getPtrWithOffset<int>(id_offset),
                 output_tensors->at("sequence_length").getPtr<int>(),
                 output_tensors->at("output_ids").getPtrWithOffset<int>(id_offset),
                 output_tensors->at("output_ids").getPtrWithOffset<int>(id_offset),
                 &beam_hyps,
                 local_batch_size,
                 beam_width,
                 vocab_size_padded_,
                 input_tensors->at("end_id").getPtr<const int>(),
                 stream_);
    sync_check_cuda_error();
}

template<typename T>
void OnlineBeamSearchLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void OnlineBeamSearchLayer<T>::allocateBuffer(size_t batch_size, size_t beam_width)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // we need to check 2 * beam_width candidates each time
    // 64 is the max beam width we support now.
    topk_softmax_workspace_size_ =
        (size_t)(ceil(batch_size * 64 * (64 * 2) / 4.) * 4 * 2
                 + ceil(batch_size * (64 * 2) * SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * (MAX_K * 2) + 2) / 4.) * 4);

    topk_softmax_workspace_ = reinterpret_cast<float*>(
        allocator_->reMalloc(topk_softmax_workspace_, sizeof(float) * topk_softmax_workspace_size_, true));
    is_allocate_buffer_ = true;
}

template<typename T>
OnlineBeamSearchLayer<T>::OnlineBeamSearchLayer(size_t           max_batch_size,
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
    BaseBeamSearchLayer<T>(max_batch_size,
                           head_num,
                           size_per_head,
                           beam_width,
                           vocab_size,
                           vocab_size_padded,
                           end_id,
                           diversity_rate,
                           temperature,
                           len_penalty,
                           repetition_penalty,
                           stream,
                           cublas_wrapper,
                           allocator,
                           is_free_buffer_after_forward)
{
}

template<typename T>
OnlineBeamSearchLayer<T>::OnlineBeamSearchLayer(OnlineBeamSearchLayer<T> const& beam_search_layer):
    BaseBeamSearchLayer<T>(beam_search_layer)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
OnlineBeamSearchLayer<T>::~OnlineBeamSearchLayer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template class OnlineBeamSearchLayer<float>;
template class OnlineBeamSearchLayer<half>;

}  // namespace turbomind
