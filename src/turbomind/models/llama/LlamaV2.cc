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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/models/multi_gpu_gpt/ParallelGpt.cc

#include "src/turbomind/models/llama/LlamaV2.h"
#include "src/turbomind/kernels/decoding_kernels.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/LlamaBatch.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/Request.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace turbomind {

template<typename T>
LlamaV2<T>::LlamaV2(size_t                       head_num,
                    size_t                       kv_head_num,
                    size_t                       size_per_head,
                    size_t                       inter_size,
                    size_t                       num_layer,
                    size_t                       vocab_size,
                    const LlamaAttentionParams&  attn_params,
                    float                        norm_eps,
                    int                          max_batch_size,
                    int                          max_context_token_num,
                    int                          session_len,
                    int                          step_length,
                    int                          start_id,
                    int                          end_id,
                    int                          cache_max_entry_count,
                    int                          cache_chunk_size,
                    int                          quant_policy,
                    bool                         use_context_fmha,
                    std::shared_ptr<SharedState> shared_state,
                    LlamaWeight<T>*              weights,
                    NcclParam                    tensor_para,
                    cudaStream_t                 stream,
                    cublasMMWrapper*             cublas_wrapper,
                    IAllocator*                  allocator,
                    bool                         is_free_buffer_after_forward,
                    cudaDeviceProp*              cuda_device_prop):
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    vocab_size_padded_(vocab_size),
    rmsnorm_eps_(norm_eps),
    start_id_(start_id),
    end_id_(end_id),
    hidden_units_(head_num * size_per_head),
    local_head_num_(head_num / tensor_para.world_size_),
    weights_(weights),
    tensor_para_(tensor_para),
    stream_(stream),
    cublas_wrapper_(cublas_wrapper),
    allocator_(allocator),
    is_free_buffer_after_forward_(is_free_buffer_after_forward),
    cuda_device_prop_(cuda_device_prop),
    debug_(isDebug()),
    step_length_(step_length),
    batch_(max_batch_size, max_context_token_num, session_len, this),
    shared_state_(shared_state)

{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    TM_LOG_INFO("NCCL group_id = %d", tensor_para_.group_id_);

    vocab_size_padded_ =
        (vocab_size_padded_ + tensor_para_.world_size_ - 1) / tensor_para_.world_size_ * tensor_para_.world_size_;

    size_t elem_bits = 0;
    if (quant_policy & QuantPolicy::kCacheKVInt8) {
        elem_bits = sizeof(int8_t) * 8;
        if (use_context_fmha) {
            TM_LOG_ERROR("use_context_fmha not support int8");
            assert(0);
        }
    }
    else {
        elem_bits = sizeof(T) * 8;
    }

    const size_t local_kv_head_num = kv_head_num / tensor_para.world_size_;

    kv_cache_mgr_ = std::make_unique<LlamaCacheManager>(num_layer_,
                                                        local_kv_head_num,
                                                        size_per_head_,
                                                        session_len,
                                                        elem_bits,
                                                        cache_max_entry_count,
                                                        cache_chunk_size,
                                                        tensor_para.rank_,
                                                        allocator);
    initialize(attn_params, kv_head_num, use_context_fmha, quant_policy);
    start();
}

template<typename T>
LlamaV2<T>::~LlamaV2()
{
    shared_state_->request_queue.close();
    internal_thread_.join();

    delete decoder_;
    delete dynamic_decode_layer_;
    delete context_decoder_;
}

template<typename T>
void LlamaV2<T>::initialize(const LlamaAttentionParams& attn_params,
                            size_t                      kv_head_num,
                            bool                        use_context_fmha,
                            int                         quant_policy)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    context_decoder_ = new LlamaContextDecoder<T>(head_num_,
                                                  kv_head_num,
                                                  size_per_head_,
                                                  inter_size_,
                                                  num_layer_,
                                                  attn_params,
                                                  rmsnorm_eps_,
                                                  tensor_para_,
                                                  stream_,
                                                  cublas_wrapper_,
                                                  allocator_,
                                                  is_free_buffer_after_forward_,
                                                  use_context_fmha,
                                                  quant_policy);

    decoder_ = new LlamaDecoder<T>(head_num_,
                                   kv_head_num,
                                   size_per_head_,
                                   inter_size_,
                                   num_layer_,
                                   attn_params,
                                   rmsnorm_eps_,
                                   tensor_para_,
                                   stream_,
                                   cublas_wrapper_,
                                   allocator_,
                                   is_free_buffer_after_forward_,
                                   quant_policy);

    dynamic_decode_layer_ = new DynamicDecodeLayer<float>(vocab_size_,
                                                          vocab_size_padded_,
                                                          0,  // end_id, deprecated
                                                          stream_,
                                                          cublas_wrapper_,
                                                          allocator_,
                                                          is_free_buffer_after_forward_,
                                                          cuda_device_prop_);
}

template<typename T>
void LlamaV2<T>::embeddingLookup(T* embeddings, const int* token_ids_buf, int batch_size, int step)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // ! This kernel can't be used in context decoding
    invokeEmbeddingLookupPosEncodingPadCount(embeddings,
                                             weights_->pre_decoder_embedding_table,
                                             static_cast<T*>(nullptr),  // position encoding
                                             token_ids_buf,
                                             static_cast<int*>(nullptr),  // padding count, not used w/o pos-code
                                             batch_size,
                                             hidden_units_,
                                             static_cast<T>(1.),  // scale
                                             step,                // step, used int index into output_ids_buf_
                                             batch_size,          // token_num
                                             0,                   // ite
                                             stream_);
    sync_check_cuda_error();
}

template<typename T>
void LlamaV2<T>::contextDecode(T*         deocder_output,
                               uintptr_t* k_cache_ptr,
                               uintptr_t* v_cache_ptr,
                               T*         context_decoder_input_buf,
                               T*         context_decoder_output_buf,
                               const int* input_ids,
                               const int* input_length,
                               const int* history_length,
                               const int* context_length,
                               size_t     token_num,
                               size_t     max_input_len,
                               size_t     max_context_len,
                               size_t     session_len,
                               size_t     batch_size)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (tensor_para_.rank_ == 0) {
        TM_LOG_INFO("context decoding start");
    }

    invokeInputIdsEmbeddingLookupPosEncoding(context_decoder_input_buf,
                                             nullptr,  // processed somewhere else
                                             weights_->pre_decoder_embedding_table,
                                             static_cast<T*>(nullptr),
                                             pPromptTuningParam<T>{},
                                             input_ids,
                                             0,  // only used for position encoding
                                             token_num,
                                             token_num,
                                             1,
                                             hidden_units_,
                                             stream_);
    sync_check_cuda_error();

    const auto dtype = getTensorType<T>();
    const auto bsz   = batch_size;

    const int max_q_len   = max_input_len;
    const int max_kv_len  = max_context_len;
    const int max_seq_len = session_len;

    std::unordered_map<std::string, Tensor> decoder_input_tensors{
        {"decoder_input", {MEMORY_GPU, dtype, {token_num, hidden_units_}, context_decoder_input_buf}},
        {"output_norm_weight", {MEMORY_GPU, dtype, {hidden_units_}, weights_->output_norm_weight}},
        {"input_lengths", {MEMORY_GPU, TYPE_INT32, {bsz}, input_length}},
        {"history_lengths", {MEMORY_GPU, TYPE_INT32, {bsz}, history_length}},
        {"context_lengths", {MEMORY_GPU, TYPE_INT32, {bsz}, context_length}},
        {"max_q_len", {MEMORY_CPU, TYPE_INT32, {1}, &max_q_len}},
        {"max_kv_len", {MEMORY_CPU, TYPE_INT32, {1}, &max_kv_len}},
        {"max_seq_len", {MEMORY_CPU, TYPE_INT32, {1}, &max_seq_len}},
    };

    std::unordered_map<std::string, Tensor> decoder_output_tensors{
        {"decoder_output", {MEMORY_GPU, dtype, {token_num, hidden_units_}, context_decoder_output_buf}},
        {"key_cache", {MEMORY_GPU, TYPE_UINT64, {bsz}, k_cache_ptr}},
        {"value_cache", {MEMORY_GPU, TYPE_UINT64, {bsz}, v_cache_ptr}},
        {"last_token_hidden_units", {MEMORY_GPU, dtype, {bsz, hidden_units_}, deocder_output}}};

    context_decoder_->forward(&decoder_output_tensors, &decoder_input_tensors, &weights_->decoder_layer_weights);

    if (tensor_para_.rank_ == 0) {
        TM_LOG_INFO("context decoding end");
    }
}

template<typename T>
void LlamaV2<T>::decoderForward(T*         decoder_output,
                                uintptr_t* k_cache_ptr,
                                uintptr_t* v_cache_ptr,
                                T*         decoder_input,
                                const int* sequence_length,
                                const int* total_padding_count,
                                bool*      finished,
                                int        step,
                                int        ite,
                                size_t     session_len,
                                size_t     batch_size)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const int  max_seq_len = session_len;
    const auto dtype       = getTensorType<T>();

    // max_input_length is not used w/o linear_bias_slopes
    // sequence_lengths_ will be incremented in dynamic decode
    std::unordered_map<std::string, Tensor> decoder_input_tensors{
        {"decoder_input", {MEMORY_GPU, dtype, {batch_size, hidden_units_}, decoder_input}},
        {"sequence_lengths", {MEMORY_GPU, TYPE_INT32, {batch_size}, sequence_length}},
        {"total_padding_tokens", {MEMORY_GPU, TYPE_INT32, {batch_size}, total_padding_count}},
        {"max_seq_len", {MEMORY_CPU, TYPE_INT32, {1}, &max_seq_len}},
        {"finished", {MEMORY_GPU, TYPE_BOOL, {batch_size}, finished}},
        {"output_norm_weight", {MEMORY_GPU, dtype, {hidden_units_}, weights_->output_norm_weight}},
        {"step", {MEMORY_CPU, TYPE_INT32, {1}, &step}},
        {"ite", {MEMORY_CPU, TYPE_INT32, {1}, &ite}},
    };

    // LOG(ERROR) << key_cache_ << " " << value_cache_;
    std::unordered_map<std::string, Tensor> decoder_output_tensors{
        {"decoder_output", {MEMORY_GPU, dtype, {batch_size, hidden_units_}, decoder_output}},
        {"key_cache", {MEMORY_GPU, TYPE_UINT64, {batch_size}, k_cache_ptr}},
        {"value_cache", {MEMORY_GPU, TYPE_UINT64, {batch_size}, v_cache_ptr}},
    };

    decoder_->forward(&decoder_output_tensors, &decoder_input_tensors, &weights_->decoder_layer_weights);
}

template<typename T>
void LlamaV2<T>::postDecodeEmbedding(float* logits, float* local_logits, const T* decoder_output, int batch_size)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    cudaDataType_t data_type = getCudaDataType<T>();
    float          alpha     = 1.f;
    float          beta      = 0.f;
    if (tensor_para_.world_size_ == 1) {
        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              vocab_size_,  // n
                              batch_size,
                              hidden_units_,  // k
                              &alpha,
                              weights_->post_decoder_embedding_kernel,
                              data_type,
                              hidden_units_,  // k
                              decoder_output,
                              data_type,
                              hidden_units_,  // k
                              &beta,
                              logits,
                              CUDA_R_32F,
                              vocab_size_,  // n
                              CUDA_R_32F,
                              cublasGemmAlgo_t(-1));
    }
    else {
        FT_CHECK(vocab_size_padded_ % tensor_para_.world_size_ == 0);
        const size_t local_vocab_size = vocab_size_padded_ / tensor_para_.world_size_;
        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              local_vocab_size,  // n
                              batch_size,
                              hidden_units_,  // k
                              &alpha,
                              weights_->post_decoder_embedding_kernel
                                  + tensor_para_.rank_ * local_vocab_size * hidden_units_,
                              data_type,
                              hidden_units_,  // k
                              decoder_output,
                              data_type,
                              hidden_units_,  // k
                              &beta,
                              local_logits + tensor_para_.rank_ * batch_size * local_vocab_size,
                              CUDA_R_32F,
                              local_vocab_size,  // n
                              CUDA_R_32F,
                              cublasGemmAlgo_t(-1));
        {
            NcclGuard nccl_guard(tensor_para_, stream_);
            ftNcclAllGather(local_logits,                   // send_buf
                            local_logits,                   // recv_buf
                            batch_size * local_vocab_size,  // data_size
                            tensor_para_.rank_,
                            tensor_para_,
                            stream_);
        }
        invokeTransposeAxis01(logits, local_logits, tensor_para_.world_size_, batch_size, local_vocab_size, stream_);
        sync_check_cuda_error();
    }
}

template<typename T>
void LlamaV2<T>::dynamicDecode(int*            token_ids,
                               bool*           finished,
                               int*            sequence_length,
                               bool*           should_stop,
                               TensorMap*      inputs,
                               TensorMap*      outputs,
                               const float*    logits,
                               const uint32_t* seq_limit_len,
                               const int*      context_length,
                               const int*      end_ids,
                               int             step,
                               int             ite,
                               size_t          max_context_len,
                               size_t          token_ids_len,
                               size_t          batch_size)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    int local_batch_size = (int)batch_size;

    std::unordered_map<std::string, Tensor> dynamic_decode_input_tensors{
        {"logits", {MEMORY_GPU, TYPE_FP32, {batch_size, (size_t)1, vocab_size_padded_}, logits}},
        {"step", {MEMORY_CPU, TYPE_INT32, {1}, &step}},
        {"max_input_length", {MEMORY_CPU, TYPE_INT32, {1}, &max_context_len}},
        {"sequence_limit_length", {MEMORY_GPU, TYPE_UINT32, {batch_size}, seq_limit_len}},
        {"input_lengths", {MEMORY_GPU, TYPE_INT32, {batch_size, 1}, context_length}},
        {"ite", {MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
        {"end_id", {MEMORY_GPU, TYPE_INT32, {batch_size}, end_ids}},
        {"local_batch_size", {MEMORY_CPU, TYPE_INT32, {1}, &local_batch_size}},
    };

    const std::vector<std::string> optional_inputs{"stop_words_list",
                                                   "bad_words_list",
                                                   "runtime_top_k",
                                                   "runtime_top_p",
                                                   "temperature",
                                                   "repetition_penalty",
                                                   "random_seed"};
    for (const auto& key : optional_inputs) {
        if (inputs->isExist(key)) {
            dynamic_decode_input_tensors.insert({key, inputs->at(key)});
        }
    }

    std::unordered_map<std::string, Tensor> dynamic_decode_output_tensors{
        {"output_ids", {MEMORY_GPU, TYPE_INT32, {token_ids_len, batch_size, 1U}, token_ids}},
        {"finished", {MEMORY_GPU, TYPE_BOOL, {batch_size}, finished}},
        {"sequence_length", {MEMORY_GPU, TYPE_INT32, {batch_size}, sequence_length}},
        {"should_stop", {MEMORY_CPU, TYPE_BOOL, {1}, should_stop}}};

    const std::vector<std::string> optional_outputs{"cum_log_probs", "output_log_probs"};
    for (const auto& key : optional_outputs) {
        if (outputs->isExist(key)) {
            dynamic_decode_output_tensors.insert({key, outputs->at(key)});
        }
    }

    dynamic_decode_layer_->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);
}

template<typename T>
void LlamaV2<T>::internalThreadEntry(int device_id)
{
    TM_LOG_INFO("[internalThreadEntry] %d", (int)tensor_para_.rank_);
    check_cuda_error(cudaSetDevice(device_id));

    auto& request_queue  = shared_state_->request_queue;
    auto& infer_requests = shared_state_->infer_requests;
    auto& stop_requests  = shared_state_->stop_requests;

    while (1) {
        if (tensor_para_.rank_ == 0) {
            const int  free_slot_count = batch_.maxSize() - batch_.size() + batch_.finishedCount();
            const bool is_empty        = free_slot_count == batch_.maxSize();

            request_queue.dequeue(stop_requests, infer_requests, free_slot_count, is_empty);

            // request queue was closed
            // and there are no unprocessed requests in the queue
            if (is_empty && infer_requests.empty() && stop_requests.empty()) {
                // rank 0 sets flag
                shared_state_->should_stop = true;
            }

            batch_.verifyRequests(stop_requests, infer_requests);
        }

        // wait while rank-0 is dequeueing
        shared_state_->barrier->wait();

        // exit if job is done
        if (shared_state_->should_stop) {
            return;
        }

        bool modified = false;

        if (!(batch_.finishedCount() == 0 && stop_requests.empty() && infer_requests.empty())) {
            batch_.handleStopRequests(stop_requests);
            batch_.synchronize();
            modified = true;
        }

        const int infer_request_count = infer_requests.size();

        if (!infer_requests.empty()) {
            batch_.initialize(infer_requests);  // reinitialize when new requests come, possible buffer allocation
            batch_.contextDecode();
            modified = true;
        }

        // wait while shared stop/infer_requests is being used
        shared_state_->barrier->wait();

        if (batch_.size()) {
            if (modified) {
                batch_.initializeGeneration();
                batch_.initializeSampling(infer_request_count);
            }
            for (int i = 0; i < step_length_; ++i) {
                if (!batch_.generate()) {
                    break;
                }
            }
            batch_.finish();
        }
    }
}

template<typename T>
void LlamaV2<T>::start()
{
    int device_id = -1;
    check_cuda_error(cudaGetDevice(&device_id));
    internal_thread_ = std::thread(&LlamaV2<T>::internalThreadEntry, this, device_id);
}

static inline Tensor slice(const Tensor& tensor, int index)
{
    auto shape = tensor.shape;
    if (shape.at(0) == 1) {
        return tensor;
    }
    shape[0]          = 1;
    const auto offset = std::accumulate(shape.begin(), shape.end(), (size_t)index, std::multiplies<>{});
    return tensor.slice(shape, offset);
}

// ! implicit conversion from `unordered_map` to `TensorMap` drops 0-sized tensors
static inline TensorMap slice(const std::unordered_map<std::string, Tensor>& src, int index)
{
    TensorMap dst;
    for (const auto& kv : src) {
        dst.insert({kv.first, slice(kv.second, index)});
    }
    return dst;
}

template<typename T>
void LlamaV2<T>::forward(std::unordered_map<std::string, Tensor>*       outputs,
                         const std::unordered_map<std::string, Tensor>* inputs,
                         Control                                        control)
{
    if (debug_) {
        if (tensor_para_.rank_ == 0) {
            for (const auto& kv : *inputs) {
                TM_LOG_INFO("[forward][rank=%d] INPUT: %s", (int)tensor_para_.rank_, format(kv).c_str());
            }
            for (const auto& kv : *outputs) {
                TM_LOG_INFO("[forward][rank=%d] OUTPUT: %s", (int)tensor_para_.rank_, format(kv).c_str());
            }
        }
    }

    const int batch_size = outputs->at("output_ids").shape[0];

    const auto rank = tensor_para_.rank_;

    std::vector<std::shared_ptr<Request>> requests(batch_size);

    // rank-0 allocates all requests for the batch
    if (rank == 0) {
        for (int i = 0; i < batch_size; ++i) {
            requests[i] = std::make_shared<Request>();
            requests[i]->inputs.resize(tensor_para_.world_size_);
            requests[i]->outputs.resize(tensor_para_.world_size_);
        }
        control.comm->setSharedObject(&requests);
    }

    control.comm->barrier();

    if (rank != 0) {
        requests = *(std::vector<std::shared_ptr<Request>>*)control.comm->getSharedObject();
    }

    for (int i = 0; i < batch_size; ++i) {
        auto& r = requests[i];

        r->inputs[rank]  = slice(*inputs, i);
        r->outputs[rank] = slice(*outputs, i);

        if (rank == 0) {
            r->id         = r->inputs[rank].getVal<uint64_t>("CORRID", i);
            r->start_flag = r->inputs[rank].getVal<int>("START", 1);
            r->end_flag   = r->inputs[rank].getVal<int>("END", 1);
            r->stop_flag  = r->inputs[rank].getVal<int>("STOP", 0);
            r->stream_cb  = control.callback;
        }
    }

    control.comm->barrier();

    // rank-0 now takes the ownership of `requests`
    // rank-0 submits the tasks and wait for finish
    std::vector<int> error_codes;
    bool             has_error = 0;
    if (rank == 0) {
        TM_LOG_INFO("[forward] Enqueue requests");
        auto futures = shared_state_->request_queue.enqueue(std::move(requests));

        TM_LOG_INFO("[forward] Wait for requests to complete ...");
        for (auto& f : futures) {
            auto ec = f.get();
            error_codes.push_back(ec);
            if (ec) {
                has_error = true;
            }
        }
    }

    // prevents request tensors being freed before the batch completes
    control.comm->barrier();

    if (rank == 0 && has_error) {
        std::stringstream ss;
        for (int i = 0; i < error_codes.size(); ++i) {
            ss << (i ? "" : " ") << error_codes[i];
        }
        throw std::runtime_error(ss.str());
    }
}

template class LlamaV2<half>;
template class LlamaV2<float>;

}  // namespace turbomind
