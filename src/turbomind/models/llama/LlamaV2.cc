/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 * Copyright (c) 2022, SK Telecom Authored by A. Dialog
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

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.cc

#include <algorithm>
#include <memory>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/typecvt.h"
#include "src/turbomind/macro.h"

#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/LlamaV2.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/unified_decoder.h"

#include "src/turbomind/kernels/gpt_kernels.h"

#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

/// TODO: Padded vocab size should also be divisible by 8
inline int pad_vocab_size(int vocab_size, int tp)
{
    return (vocab_size + tp - 1) / tp * tp;
}

LlamaV2::LlamaV2(DataType                     dtype,
                 const ModelParam&            model,
                 const EngineParam&           engine,
                 const AttentionParam&        attn,
                 const MoeParam&              moe,
                 const LoraParam&             lora,
                 const Context&               ctx,
                 int                          max_batch_size,
                 std::shared_ptr<LlamaWeight> weights):
    dtype_{dtype},
    param_(model),
    attn_param_(attn),
    lora_param_(lora),
    comm_(&ctx.comm),
    tp_size_(engine.attn_tp_size),
    tp_rank_(engine.attn_tp_rank),
    head_num_(model.head_num),
    size_per_head_(model.head_dim),
    hidden_units_(model.hidden_units),
    layer_num_(model.layer_num),
    vocab_size_(model.vocab_size),
    vocab_size_padded_(pad_vocab_size(model.vocab_size, tp_size_)),
    rmsnorm_eps_(model.norm_eps),
    local_head_num_(model.head_num / engine.attn_tp_size),
    local_kv_head_num_(model.kv_head_num / engine.attn_tp_size),
    weights_(std::move(weights)),
    stream_(ctx.stream),
    cublas_wrapper_(ctx.cublas_wrapper.get()),
    allocator_(ctx.allocator.get()),
    linear_(ctx.linear.get()),
    is_free_buffer_after_forward_(false),
    debug_(isDebug())
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (comm_->d_comm && comm_->d_comm->Query(comm::kHasAllGather2D)) {
        use_allgather_2d_ = true;
    }

    unified_decoder_ = std::make_unique<UnifiedDecoder>(model, engine, attn, moe, lora, ctx);

    dynamic_decode_layer_ = std::make_unique<DynamicDecodeLayer<float>>(vocab_size_,
                                                                        vocab_size_padded_,
                                                                        stream_,
                                                                        cublas_wrapper_,
                                                                        allocator_,
                                                                        is_free_buffer_after_forward_,
                                                                        (cudaDeviceProp*)&ctx.cuda_device_prop);
}

LlamaV2::~LlamaV2()
{
    dynamic_decode_layer_.reset();
    unified_decoder_.reset();
}

void LlamaV2::updateEmbedding(char*            decoder_input,
                              const int        bsz,
                              const int*       h_input_length,
                              const Sequence** sequences,
                              int              token_num,
                              int*             lora_mask,
                              bool*            have_embeddings)
{
    if (isTuning())
        return;

    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    *have_embeddings          = false;
    int*             mask_ptr = nullptr;
    std::vector<int> mask;
    if (lora_mask != nullptr) {
        mask     = std::vector<int>(token_num);
        mask_ptr = mask.data();
    }

    const size_t elem_size = core::get_byte_size(dtype_);

    for (int i = 0; i < bsz; i++) {
        const auto& seq        = *sequences[i];
        const auto& embeddings = seq.input_embeddings;
        const auto& ranges     = seq.input_embedding_ranges;
        for (int j = embeddings.size() - 1; j >= 0; j--) {
            int begin = ranges[j].first;
            int end   = ranges[j].second;
            if (seq.cache_len + h_input_length[i] - 1 < begin) {
                continue;
            }
            if (end <= seq.cache_len) {
                break;
            }
            int off_dst = std::max(0, begin - seq.cache_len);
            int off_src = std::max(0, seq.cache_len - begin);
            // calculate intersection of [begin, end) and [seq.cache_len, seq.cache_len + h_input_length[i])
            begin            = std::max(begin, seq.cache_len);
            end              = std::min(end, seq.cache_len + h_input_length[i]);
            size_t byte_size = elem_size * (end - begin) * hidden_units_;
            char*  dst_ptr   = decoder_input + elem_size * off_dst * hidden_units_;
            auto   src_ptr   = embeddings[j].data() + elem_size * off_src * hidden_units_;
            check_cuda_error(cudaMemcpyAsync(dst_ptr, src_ptr, byte_size, cudaMemcpyDefault, stream_));
            if (lora_mask != nullptr) {
                std::fill_n(mask_ptr + off_dst, (end - begin), 1);
                *have_embeddings = true;
            }
        }
        decoder_input += elem_size * h_input_length[i] * hidden_units_;
        mask_ptr += h_input_length[i];
    }

    if (lora_mask != nullptr && *have_embeddings) {
        cudaMemcpyAsync(lora_mask, mask.data(), sizeof(int) * token_num, cudaMemcpyDefault, stream_);
        cudaStreamSynchronize(stream_);
    }
    sync_check_cuda_error();
}

void LlamaV2::Forward(Buffer_<int>     input_ids,
                      core::Tensor     hidden_states_out,
                      core::Tensor     decode_out,
                      Buffer           kv_block_ptrs,
                      Buffer           cu_block_nums,
                      Buffer_<int>     h_input_length,
                      Buffer_<int>     h_context_length,
                      Buffer           rope_base,
                      Buffer           finished,
                      Buffer           local_token_nums,
                      Buffer           lora_mask,
                      int              decode_num,
                      int              prefil_num,
                      const Sequence** sequences)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    core::Tensor input_embeds;

    const int token_num = input_ids.size();
    const int bsz       = decode_out.shape(0);

    if (token_num) {
        const auto& embedding_table = weights_->pre_decoder_embedding.weight;
        TM_CHECK_EQ(embedding_table.shape(1) * tp_size_, hidden_units_);
        input_embeds = core::Tensor{{token_num, (int)hidden_units_}, dtype_, MEMORY_GPU};
        if (tp_size_ == 1) {
            invokeEmbeddingLookup(input_embeds, input_ids, embedding_table, stream_);
            sync_check_cuda_error();
        }
        else {

            const auto   local_hidden_units = embedding_table.shape(1);
            core::Tensor temp{hidden_states_out.buffer(), {tp_size_, token_num, local_hidden_units}};

            auto local = temp.slice(tp_rank_);

            invokeEmbeddingLookup(local, input_ids, embedding_table, stream_);
            sync_check_cuda_error();

            comm_->d_comm->AllGather(
                local.raw_data(), temp.raw_data(), local.size(), dtype_, comm_->d_tp_group, stream_);
            sync_check_cuda_error();

            invokeInPlaceTranspose102((uint16_t*)input_embeds.raw_data(),
                                      (uint16_t*)temp.raw_data(),
                                      tp_size_,
                                      token_num,
                                      local_hidden_units,
                                      false,
                                      stream_);

            sync_check_cuda_error();
        }

        TM_DEBUG_TENSOR(input_embeds, "embeddings", 1);
    }

    bool have_embeddings = false;
    if (token_num) {
        // Copy input embeddings from corresponding sequences
        updateEmbedding((char*)input_embeds.raw_data(),
                        h_input_length.size(),
                        h_input_length.data(),
                        sequences,
                        token_num,
                        lora_mask ? lora_mask.data<int>() : nullptr,
                        &have_embeddings);
        sync_check_cuda_error();
    }

    core::TensorMap args{{"decoder_input", input_embeds},
                         {"decoder_output", hidden_states_out.view({-1, (int)hidden_units_})},
                         {"last_token_hidden_units", decode_out},
                         {"output_norm_weight", weights_->output_norm_weight},
                         {"h_q_len", h_input_length},
                         {"h_k_len", h_context_length},
                         {"finished", finished},
                         {"decode_num", Buffer{&decode_num, 1, MEMORY_CPU}},
                         {"prefil_num", Buffer{&prefil_num, 1, MEMORY_CPU}},
                         {"rope_base", rope_base},
                         {"cu_block_nums", cu_block_nums},
                         {"kv_block_ptrs", kv_block_ptrs},
                         {"local_token_nums", local_token_nums}};

    unified_decoder_->Forward(args, weights_->decoder_layer_weights);
}

void LlamaV2::postDecodeEmbedding(float* logits, float* local_logits, const void* decoder_output, int batch_size)
{
    NvtxScope scope("postDecodeEmbedding");
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    cudaDataType_t data_type = to_cuda_dtype(dtype_);
    float          alpha     = 1.f;
    float          beta      = 0.f;
    FT_CHECK(vocab_size_padded_ % tp_size_ == 0);
    const size_t local_vocab_size = vocab_size_padded_ / tp_size_;

    const core::Tensor src{const_cast<void*>(decoder_output), {batch_size, (int)hidden_units_}, dtype_, MEMORY_GPU};

    if (tp_size_ == 1) {
        core::Tensor out{logits, {batch_size, (int)vocab_size_padded_}, MEMORY_GPU};
        linear_->forward(src, weights_->post_decoder_embedding, LlamaLinear::kGemm, out);
        sync_check_cuda_error();
    }
    else if (use_allgather_2d_ == false) {
        FT_CHECK(logits != local_logits);
        core::Tensor logits_{local_logits, {tp_size_, batch_size, (int)local_vocab_size}, MEMORY_GPU};
        core::Tensor local = logits_.slice({tp_rank_, 0, 0}, {1, -1, -1});
        linear_->forward(src, weights_->post_decoder_embedding, LlamaLinear::kGemm, local.squeeze(0));
        sync_check_cuda_error();
        comm_->d_comm->AllGather(
            local.raw_data(), local_logits, local.size(), local.dtype(), comm_->d_tp_group, stream_);
        sync_check_cuda_error();
        invokeTransposeAxis01(logits, local_logits, tp_size_, batch_size, local_vocab_size, stream_);
        sync_check_cuda_error();
    }
    else {
        FT_CHECK(logits == local_logits);
        core::Tensor logits_{logits, {batch_size, tp_size_, (int)local_vocab_size}, MEMORY_GPU};
        core::Tensor local = logits_.slice({0, tp_rank_, 0}, {-1, 1, -1});
        linear_->forward(src, weights_->post_decoder_embedding, LlamaLinear::kGemm, local.squeeze(1));
        sync_check_cuda_error();
        comm_->d_comm->AllGather2D(local.raw_data(),
                                   logits_.raw_data(),
                                   vocab_size_padded_,
                                   local_vocab_size,
                                   local_vocab_size,
                                   batch_size,
                                   logits_.dtype(),
                                   {true, true},
                                   comm_->d_tp_group,
                                   stream_);
        sync_check_cuda_error();
    }
}

void LlamaV2::dynamicDecode(int*            token_ids,
                            bool*           finished,
                            int*            sequence_length,
                            bool*           should_stop,
                            curandState_t*  curand_state,
                            TensorMap*      inputs,
                            TensorMap*      outputs,
                            const float*    logits,
                            const uint32_t* seq_limit_len,
                            const int*      context_length,
                            int             step,
                            int             ite,
                            size_t          max_context_len,
                            size_t          token_ids_len,
                            size_t          batch_size)
{
    NvtxScope scope("dynamicDecode");
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    int local_batch_size = (int)batch_size;

    std::unordered_map<std::string, Tensor> dynamic_decode_input_tensors{
        {"logits", {MEMORY_GPU, TYPE_FP32, {batch_size, (size_t)1, vocab_size_padded_}, logits}},
        {"step", {MEMORY_CPU, TYPE_INT32, {1}, &step}},
        {"max_input_length", {MEMORY_CPU, TYPE_INT32, {1}, &max_context_len}},
        {"sequence_limit_length", {MEMORY_GPU, TYPE_UINT32, {batch_size}, seq_limit_len}},
        {"input_lengths", {MEMORY_GPU, TYPE_INT32, {batch_size, 1}, context_length}},
        {"ite", {MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
        {"local_batch_size", {MEMORY_CPU, TYPE_INT32, {1}, &local_batch_size}},
    };

    const std::vector<std::string> optional_inputs{"end_ids",
                                                   "stop_words_list",
                                                   "bad_words_list",
                                                   "runtime_top_k",
                                                   "runtime_top_p",
                                                   "temperature",
                                                   "repetition_penalty"};
    for (const auto& key : optional_inputs) {
        if (inputs->isExist(key)) {
            dynamic_decode_input_tensors.insert({key, inputs->at(key)});
        }
    }

    std::unordered_map<std::string, Tensor> dynamic_decode_output_tensors{
        {"output_ids", {MEMORY_GPU, TYPE_INT32, {token_ids_len, batch_size, 1U}, token_ids}},
        {"finished", {MEMORY_GPU, TYPE_BOOL, {batch_size}, finished}},
        {"sequence_length", {MEMORY_GPU, TYPE_INT32, {batch_size}, sequence_length}},
        {"should_stop", {MEMORY_CPU, TYPE_BOOL, {1}, should_stop}},
        {"curand_state", {MEMORY_GPU, TYPE_VOID, {batch_size}, curand_state}}};

    const std::vector<std::string> optional_outputs{
        "cum_log_probs", "output_log_probs", "sampled_indexes", "sampled_logprobs", "sampled_nums"};
    for (const auto& key : optional_outputs) {
        if (outputs->isExist(key)) {
            dynamic_decode_output_tensors.insert({key, outputs->at(key)});
        }
    }

    dynamic_decode_layer_->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);
}

}  // namespace turbomind
