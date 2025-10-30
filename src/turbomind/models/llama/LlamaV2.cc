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
#include "src/turbomind/core/core.h"
#include "src/turbomind/macro.h"

#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/LlamaV2.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/unified_decoder.h"

#include "src/turbomind/kernels/gpt_kernels.h"

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
    linear_(*ctx.linear),
    debug_(isDebug())
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (comm_->d_comm && comm_->d_comm->Query(comm::kHasAllGather2D)) {
        use_allgather_2d_ = true;
    }

    unified_decoder_ = std::make_unique<UnifiedDecoder>(model, engine, attn, moe, lora, ctx);

    // using float to avoid data overflow
    dynamic_decode_ = std::make_unique<DynamicDecodeLayer>(
        kFloat32, max_batch_size, vocab_size_, vocab_size_padded_, stream_, &ctx.device_prop);
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

    const size_t elem_size = byte_size(dtype_, 1);

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
                      Tensor           hidden_states_out,
                      Tensor           decoder_out,
                      Buffer           kv_block_ptrs,
                      Buffer           cu_block_nums,
                      Buffer_<int>     h_input_length,
                      Buffer_<int>     h_context_length,
                      Buffer           rope_base,
                      MropeRope*       mrope,
                      Buffer           finished,
                      Buffer           local_token_nums,
                      Buffer           lora_mask,
                      int              decode_num,
                      int              prefil_num,
                      const Sequence** sequences)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    Tensor input_embeds;

    const int token_num = input_ids.size();

    if (token_num) {
        const auto& embedding_table = weights_->pre_decoder_embedding.weight;
        TM_CHECK_EQ(embedding_table.shape(1) * tp_size_, hidden_units_);

        input_embeds = Tensor{{token_num, (int)hidden_units_}, dtype_, kDEVICE};

        if (tp_size_ == 1) {
            invokeEmbeddingLookup(input_embeds, input_ids, embedding_table, stream_);
            sync_check_cuda_error();
        }
        else if (use_allgather_2d_) {
            const auto local_hidden_units = embedding_table.shape(1);
            Tensor     temp{hidden_states_out.buffer(), {token_num, tp_size_, local_hidden_units}};

            auto local = temp.slice({0, tp_rank_, 0}, {-1, 1, -1}).squeeze(1);

            invokeEmbeddingLookup(local, input_ids, embedding_table, stream_);
            sync_check_cuda_error();

            comm_->d_comm->AllGather2D(local.raw_data(),
                                       temp.raw_data(),
                                       hidden_units_,
                                       local_hidden_units,
                                       local_hidden_units,
                                       token_num,
                                       local.dtype(),
                                       {true, true},
                                       comm_->d_tp_group,
                                       stream_);
            sync_check_cuda_error();

            Copy(temp.buffer(), input_embeds.buffer());
        }
        else {
            const auto local_hidden_units = embedding_table.shape(1);
            Tensor     temp{hidden_states_out.buffer(), {tp_size_, token_num, local_hidden_units}};

            auto local = temp.slice(tp_rank_).squeeze(0);

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

    TM_DEBUG_TENSOR(input_embeds, "embeddings", 1);

    TensorMap args{{"decoder_input", input_embeds},
                   {"decoder_output", hidden_states_out.view({-1, (int)hidden_units_}).borrow()},
                   {"last_token_hidden_units", decoder_out},
                   {"output_norm_weight", weights_->output_norm_weight},
                   {"h_q_len", h_input_length},
                   {"h_k_len", h_context_length},
                   {"finished", finished},
                   {"decode_num", Buffer{&decode_num, 1, kCPU}},
                   {"prefil_num", Buffer{&prefil_num, 1, kCPU}},
                   {"rope_base", rope_base},
                   {"cu_block_nums", cu_block_nums},
                   {"kv_block_ptrs", kv_block_ptrs},
                   {"local_token_nums", local_token_nums}};

    if (mrope != nullptr && mrope->position_ids) {
        args.insert({"mrope_position_ids", mrope->position_ids});
        args.insert({"mrope_position_delta", mrope->position_delta});
        args.insert({"mrope_position_length", mrope->length});
    }

    unified_decoder_->Forward(args, weights_->decoder_layer_weights);
}

Tensor LlamaV2::postDecodeEmbedding(const Tensor& features, Buffer local_logits)
{
    NvtxScope scope("postDecodeEmbedding");
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    TM_CHECK(vocab_size_padded_ % tp_size_ == 0) << vocab_size_padded_ << " " << tp_size_;

    const int bsz              = features.shape(0);
    const int local_vocab_size = vocab_size_padded_ / tp_size_;

    if (tp_size_ == 1) {
        Tensor logits{local_logits, {bsz, (int)vocab_size_padded_}};
        linear_.Forward(features, weights_->post_decoder_embedding, logits);
        sync_check_cuda_error();

        TM_DEBUG_TENSOR(logits, "logits", 1);
        return logits;
    }
    else if (use_allgather_2d_) {
        Tensor logits{local_logits, {bsz, tp_size_, local_vocab_size}};
        Tensor local = logits.slice({0, tp_rank_, 0}, {-1, 1, -1});
        linear_.Forward(features, weights_->post_decoder_embedding, local.squeeze(1));
        sync_check_cuda_error();
        comm_->d_comm->AllGather2D(local.raw_data(),
                                   logits.raw_data(),
                                   vocab_size_padded_,
                                   local_vocab_size,
                                   local_vocab_size,
                                   bsz,
                                   logits.dtype(),
                                   {true, true},
                                   comm_->d_tp_group,
                                   stream_);
        sync_check_cuda_error();
        return logits.view({bsz, -1});
    }
    else {
        Tensor logits{local_logits, {tp_size_, bsz, local_vocab_size}};
        Tensor local = logits.slice({tp_rank_, 0, 0}, {1, -1, -1});
        linear_.Forward(features, weights_->post_decoder_embedding, local.squeeze(0));
        sync_check_cuda_error();
        comm_->d_comm->AllGather(
            local.raw_data(), logits.raw_data(), local.size(), local.dtype(), comm_->d_tp_group, stream_);
        sync_check_cuda_error();
        Tensor out{{bsz, (int)vocab_size_padded_}, features.dtype(), features.device()};
        invokeTransposeAxis01(
            (uint16_t*)out.raw_data(), (uint16_t*)logits.raw_data(), tp_size_, bsz, local_vocab_size, stream_);
        sync_check_cuda_error();
        return out;
    }
}

void LlamaV2::dynamicDecode(Buffer token_ids,
                            Buffer finished,
                            Buffer sequence_length,
                            Tensor curand_state,
                            Tensor logits,
                            Buffer seq_limit_len,
                            Buffer init_context_length,
                            Buffer context_length,
                            Buffer prompt_length,
                            Buffer sampled_logprobs,
                            Buffer sampled_indexes,
                            Buffer sampled_nums,
                            int    step,
                            int    max_context_len)
{
    NvtxScope scope("dynamicDecode");
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap args{
        {"logits", logits},
        {"step", Buffer{&step, 1, kCPU}},
        {"max_input_length", Buffer{&max_context_len, 1, kCPU}},
        {"sequence_limit_length", seq_limit_len},
        {"init_context_length", init_context_length},
        {"context_length", context_length},
        {"prompt_length", prompt_length},
        {"output_ids", token_ids},             // inout
        {"finished", finished},                // inout
        {"sequence_length", sequence_length},  // inout
        {"curand_state", curand_state},        // inout
    };

    if (sampled_logprobs) {
        args.emplace("sampled_logprobs", sampled_logprobs);
        args.emplace("sampled_indexes", sampled_indexes);
        args.emplace("sampled_nums", sampled_nums);
    }

    dynamic_decode_->Forward(args);
}

}  // namespace turbomind
