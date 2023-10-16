// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/LlamaBatch.h"
#include "src/turbomind/kernels/decoding_kernels.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/LlamaV2.h"
#include "src/turbomind/models/llama/Request.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/logger.h"
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace turbomind {

template<typename T>
void LlamaBatch<T>::verifyRequests(std::vector<std::shared_ptr<Request>>& stop_reqs,
                                   std::vector<std::shared_ptr<Request>>& infer_reqs)
{
    std::unordered_map<uint64_t, int> occurrence;

    auto count_occurrence = [&occurrence](const std::vector<std::shared_ptr<Request>>& rs) {
        for (const auto& r : rs) {
            ++occurrence[r->id];
        }
    };

    auto invalidate = [](const char* type, std::shared_ptr<Request>& req, int ec) {
        TM_LOG_WARNING("[verifyRequests] Skipping invalid %s request for id %ld, code = %d", type, (long)req->id, ec);
        // We don't need a barrier there because
        // this lambda is called only for new requests
        // which are visible only for rank = 0 thread.
        req->signal.set_value(ec);
        req.reset();
    };

    auto handle_conflict_or_invalid = [this, &occurrence, &invalidate](std::vector<std::shared_ptr<Request>>& rs,
                                                                       const char*                            type) {
        for (auto& r : rs) {
            if (r) {
                int ec = 0;

                if (occurrence[r->id] != 1) {
                    ec = Request::kConflict;
                }
                else if (r->start_flag && r->stop_flag) {
                    ec = Request::kInvalid;
                }
                else if (!r->start_flag && !llama_->kv_cache_mgr_->contains(r->id)) {
                    ec = Request::kInvalid;
                }

                if (ec) {
                    invalidate(type, r, ec);
                }
            }
        }
    };

    auto drop_invalid = [](std::vector<std::shared_ptr<Request>>& rs) {
        int count = 0;
        for (int i = 0; i < rs.size(); ++i) {
            if (rs[i]) {
                rs[count++] = std::move(rs[i]);
            }
        }
        rs.resize(count);
    };

    count_occurrence(stop_reqs);
    count_occurrence(infer_reqs);

    if (!stop_reqs.empty()) {
        handle_conflict_or_invalid(stop_reqs, "stop");

        // invalidate stop-only requests for inactive sequences
        for (auto& r : stop_reqs) {
            if (r && r->end_flag == false) {
                int ec = Request::kInactive;
                for (int i = 0; i < batch_size_; ++i) {
                    if (requests_[i] && requests_[i]->id == r->id) {
                        ec = 0;
                        break;
                    }
                }
                if (ec) {
                    invalidate("stop", r, ec);
                }
            }
        }

        drop_invalid(stop_reqs);
    }

    if (!infer_reqs.empty()) {
        handle_conflict_or_invalid(infer_reqs, "infer");

        // invalidate requests for busy sequences
        for (auto& r : infer_reqs) {
            if (r) {
                for (int i = 0; i < batch_size_; ++i) {
                    if (requests_[i] && requests_[i]->id == r->id) {
                        invalidate("infer", r, Request::kBusy);
                        break;
                    }
                }
            }
        }

        drop_invalid(infer_reqs);
    }
}

template<typename T>
void LlamaBatch<T>::handleStopRequests(const std::vector<std::shared_ptr<Request>>& requests)
{
    for (const auto& r : requests) {
        int ec = Request::kFail;
        // find matching active sequence
        for (int i = 0; i < batch_size_; ++i) {
            // stop & optionally erase active sequence
            if (requests_[i] && requests_[i]->id == r->id) {
                ec = 0;
                finishRequest(i, r->end_flag);
                break;
            }
        }
        // mismatch, try erase inactive sequence
        if (ec && r->end_flag) {
            ec = 0;
            llama_->kv_cache_mgr_->erase(r->id);
        }
        // clear output buffers (prevent leaking conversations) if request is successful
        if (ec == 0) {
            auto& output_ids      = r->outputs[rank_].at("output_ids");
            auto& sequence_length = r->outputs[rank_].at("sequence_length");
            check_cuda_error(
                cudaMemsetAsync(output_ids.getPtr<int>(), 0, sizeof(int) * output_ids.shape.at(2), stream_));
            check_cuda_error(cudaMemsetAsync(sequence_length.getPtr<int>(), 0, sizeof(int), stream_));
            check_cuda_error(cudaStreamSynchronize(stream_));
        }

        // When the signal is set threads from LlamaV2::forward can exit
        // and free inputs/outputs tensors.
        // Therefore we need to make sure that no threads from LlamaV2::internalThreadEntry
        // are accessing the tensors.
        llama_->shared_state_->barrier->wait();
        if (rank_ == 0) {
            r->signal.set_value(ec);
        }
    }
}

template<typename T>
void LlamaBatch<T>::allocateBuffer(size_t batch_size, size_t session_len)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    const size_t batchxbeam = batch_size;

    const size_t hidden_units = llama_->hidden_units_;
    const size_t vocab_size   = llama_->vocab_size_padded_;

    context_decoder_input_buf_ =
        (T*)allocator_->reMalloc(context_decoder_input_buf_, sizeof(T) * max_context_token_num_ * hidden_units, false);
    context_decoder_output_buf_ =
        (T*)allocator_->reMalloc(context_decoder_output_buf_, sizeof(T) * max_context_token_num_ * hidden_units, false);
    context_decoder_ids_buf_ =
        (int*)allocator_->reMalloc(context_decoder_ids_buf_, sizeof(int) * max_context_token_num_, false);

    decoder_input_buf_  = (T*)allocator_->reMalloc(decoder_input_buf_, sizeof(T) * batchxbeam * hidden_units, false);
    decoder_output_buf_ = (T*)allocator_->reMalloc(decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units, false);

    input_ids_buf_      = (int*)allocator_->reMalloc(input_ids_buf_, sizeof(int) * batchxbeam * session_len, true);
    input_length_buf_   = (int*)allocator_->reMalloc(input_length_buf_, sizeof(int) * batchxbeam);
    history_length_buf_ = (int*)allocator_->reMalloc(history_length_buf_, sizeof(int) * batchxbeam);
    context_length_buf_ = (int*)allocator_->reMalloc(context_length_buf_, sizeof(int) * batchxbeam);

    total_padding_count_ = (int*)allocator_->reMalloc(total_padding_count_, sizeof(int) * batchxbeam, false);
    sequence_lengths_    = (int*)allocator_->reMalloc(sequence_lengths_, sizeof(int) * batchxbeam, false);

    k_cache_ptr_buf_ = (uint64_t*)allocator_->reMalloc(k_cache_ptr_buf_, sizeof(uint64_t) * batchxbeam);
    v_cache_ptr_buf_ = (uint64_t*)allocator_->reMalloc(v_cache_ptr_buf_, sizeof(uint64_t) * batchxbeam);

    logits_buf_       = (float*)allocator_->reMalloc(logits_buf_, sizeof(float) * batchxbeam * vocab_size, false);
    local_logits_buf_ = (float*)allocator_->reMalloc(local_logits_buf_, sizeof(float) * batchxbeam * vocab_size, false);

    token_ids_buf_ = (int*)allocator_->reMalloc(token_ids_buf_, sizeof(int) * batchxbeam * session_len * 2, true);

    end_ids_buf_   = (int*)allocator_->reMalloc(end_ids_buf_, sizeof(int) * batch_size, false);
    finished_buf_  = (bool*)allocator_->reMalloc(finished_buf_, sizeof(bool) * batchxbeam, false);
    seq_limit_len_ = (uint32_t*)allocator_->reMalloc(seq_limit_len_, sizeof(uint32_t) * batch_size, false);

    is_allocate_buffer_ = true;
}

template<typename T>
void LlamaBatch<T>::allocatePersistantBuffer(size_t max_batch_size)
{
    output_ids_buf_ = (int*)allocator_->reMalloc(output_ids_buf_, sizeof(int) * max_batch_size * session_len_, true);

    stop_words_buf_ =
        (int*)allocator_->reMalloc(stop_words_buf_, sizeof(int) * max_batch_size * kMaxStopBadWordsLen, true);
    bad_words_buf_ =
        (int*)allocator_->reMalloc(bad_words_buf_, sizeof(int) * max_batch_size * kMaxStopBadWordsLen, true);

    h_runtime_top_k_ = (int*)allocator_->reMalloc(h_runtime_top_k_, sizeof(int) * max_batch_size, true, true);
    h_runtime_top_p_ = (float*)allocator_->reMalloc(h_runtime_top_p_, sizeof(float) * max_batch_size, true, true);
    h_temperature_   = (float*)allocator_->reMalloc(h_temperature_, sizeof(float) * max_batch_size, true, true);
    h_repetition_penalty_ =
        (float*)allocator_->reMalloc(h_repetition_penalty_, sizeof(float) * max_batch_size, true, true);
    h_random_seed_ = (uint64_t*)allocator_->reMalloc(h_random_seed_, sizeof(uint64_t) * max_batch_size, true, true);

    sampling_params_ = {{"stop_words_list", stop_words_buf_},
                        {"bad_words_list", bad_words_buf_},
                        {"runtime_top_k", h_runtime_top_k_},
                        {"runtime_top_p", h_runtime_top_p_},
                        {"temperature", h_temperature_},
                        {"repetition_penalty", h_repetition_penalty_},
                        {"random_seed", h_random_seed_}};

    topk_curandstate_buf_ = allocator_->reMalloc(topk_curandstate_buf_, sizeof(curandState_t) * max_batch_size, true);
    topp_curandstate_buf_ = allocator_->reMalloc(topp_curandstate_buf_, sizeof(curandState_t) * max_batch_size, true);

    {
        NcclGuard barrier(llama_->tensor_para_, stream_, true);
        h_input_ids_buf_ =
            (int*)allocator_->reMalloc(h_input_ids_buf_, sizeof(int) * max_batch_size * session_len_, false, true);
        h_input_length_buf_ =
            (int*)allocator_->reMalloc(h_input_length_buf_, sizeof(int) * max_batch_size, false, true);
        h_history_length_buf_ =
            (int*)allocator_->reMalloc(h_history_length_buf_, sizeof(int) * max_batch_size, false, true);
        h_context_length_buf_ =
            (int*)allocator_->reMalloc(h_context_length_buf_, sizeof(int) * max_batch_size, false, true);
        h_sequence_lengths_ =
            (int*)allocator_->reMalloc(h_sequence_lengths_, sizeof(int) * max_batch_size, false, true);
        h_k_cache_ptr_buf_ =
            (uintptr_t*)allocator_->reMalloc(h_k_cache_ptr_buf_, sizeof(uintptr_t) * max_batch_size, true, true);
        h_v_cache_ptr_buf_ =
            (uintptr_t*)allocator_->reMalloc(h_v_cache_ptr_buf_, sizeof(uintptr_t) * max_batch_size, true, true);
        h_finished_buf_ = (bool*)allocator_->reMalloc(h_finished_buf_, sizeof(bool) * max_batch_size, false, true);
        h_seq_limit_len_ =
            (uint32_t*)allocator_->reMalloc(h_seq_limit_len_, sizeof(uint32_t) * max_batch_size, false, true);
    }

    is_allocate_persistant_buffer_ = true;
}

template<typename T>
void LlamaBatch<T>::freeBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)&context_decoder_input_buf_);
        allocator_->free((void**)&context_decoder_output_buf_);
        allocator_->free((void**)&context_decoder_ids_buf_);

        allocator_->free((void**)&decoder_input_buf_);
        allocator_->free((void**)&decoder_output_buf_);

        allocator_->free((void**)&input_ids_buf_);
        allocator_->free((void**)&input_length_buf_);
        allocator_->free((void**)&history_length_buf_);
        allocator_->free((void**)&context_length_buf_);

        allocator_->free((void**)&total_padding_count_);
        allocator_->free((void**)&sequence_lengths_);

        allocator_->free((void**)&k_cache_ptr_buf_);
        allocator_->free((void**)&v_cache_ptr_buf_);

        allocator_->free((void**)&logits_buf_);
        allocator_->free((void**)&local_logits_buf_);

        if (local_context_logits_buf_) {
            allocator_->free((void**)&local_context_logits_buf_);
        }
        if (context_logits_buf_) {
            allocator_->free((void**)&context_logits_buf_);
        }

        allocator_->free((void**)&token_ids_buf_);

        allocator_->free((void**)&end_ids_buf_);
        allocator_->free((void**)&finished_buf_);
        allocator_->free((void**)&seq_limit_len_);

        is_allocate_buffer_ = false;
    }

    if (is_allocate_persistant_buffer_) {
        allocator_->free((void**)&h_input_ids_buf_, true);
        allocator_->free((void**)&h_input_length_buf_, true);
        allocator_->free((void**)&h_history_length_buf_, true);
        allocator_->free((void**)&h_context_length_buf_, true);
        allocator_->free((void**)&h_sequence_lengths_, true);
        allocator_->free((void**)&h_k_cache_ptr_buf_, true);
        allocator_->free((void**)&h_v_cache_ptr_buf_, true);
        allocator_->free((void**)&h_seq_limit_len_, true);
        allocator_->free((void**)&h_finished_buf_, true);

        allocator_->free((void**)&output_ids_buf_);

        is_allocate_persistant_buffer_ = false;
    }
}

template<typename T>
LlamaBatch<T>::LlamaBatch(int max_batch_size, int max_context_token_num, int session_len, LlamaV2<T>* llama):
    max_batch_size_(max_batch_size),
    max_context_token_num_(max_context_token_num),
    session_len_(session_len),
    rank_(llama->tensor_para_.rank_),
    debug_(llama->debug_),
    llama_(llama),
    data_type_(getTensorType<T>())
{
    stream_         = llama_->stream_;
    allocator_      = llama_->allocator_;
    cublas_wrapper_ = llama_->cublas_wrapper_;

    requests_.resize(max_batch_size);
    request_seq_len_limit_.resize(max_batch_size);
    cached_seq_.resize(max_batch_size);

    allocatePersistantBuffer(max_batch_size);
}

template<typename T>
void LlamaBatch<T>::initializeSampling(int infer_request_count)
{
    TensorMap inputs;
    for (const auto& param : sampling_params_) {
        const Tensor* ptr{};
        for (int i = 0; i < batch_size_; ++i) {
            if (requests_[i]->inputs[rank_].isExist(param.first)) {
                ptr = &requests_[i]->inputs[rank_].at(param.first);
                break;
            }
        }
        if (ptr) {
            const auto& ref   = *ptr;
            auto        shape = ref.shape;
            FT_CHECK(shape[0] == 1);
            shape[0]                = batch_size_;
            const int size_in_bytes = ref.sizeBytes();
            check_cuda_error(cudaMemsetAsync(param.second, 0, size_in_bytes * batch_size_, stream_));
            for (int i = 0; i < batch_size_; ++i) {
                if (requests_[i]->inputs[rank_].isExist(param.first)) {
                    auto& src = requests_[i]->inputs[rank_].at(param.first);
                    FT_CHECK(ref.shape == src.shape);
                    check_cuda_error(cudaMemcpyAsync((uint8_t*)param.second + size_in_bytes * i,
                                                     src.getPtr<void>(),
                                                     size_in_bytes,
                                                     cudaMemcpyDefault,
                                                     stream_));
                }
            }
            inputs.insert({param.first, {ref.where, ref.type, shape, param.second}});
            if (debug_ && rank_ == 0) {
                TM_LOG_INFO("[initializeSampling] %s", format({param.first, inputs.at(param.first)}).c_str());
            }
        }
    }

    inputs_ = std::move(inputs);

    llama_->dynamic_decode_layer_->setup(batch_size_, 1, &inputs_);

    for (int i = 0; i < batch_size_; ++i) {
        // recover random states if not a new request or new request w/o "random_seed"
        if (i < batch_size_ - infer_request_count || !requests_[i]->inputs[rank_].isExist("random_seed")) {
            check_cuda_error(cudaMemcpyAsync(llama_->dynamic_decode_layer_->topk_curandstate_buf() + i,
                                             (curandState_t*)topk_curandstate_buf_ + i,
                                             sizeof(curandState_t),
                                             cudaMemcpyDefault,
                                             stream_));
            check_cuda_error(cudaMemcpyAsync(llama_->dynamic_decode_layer_->topp_curandstate_buf() + i,
                                             (curandState_t*)topp_curandstate_buf_ + i,
                                             sizeof(curandState_t),
                                             cudaMemcpyDefault,
                                             stream_));
        }
    }

    handleOptArg(&inputs_, "end_id", end_ids_buf_, llama_->end_id_, batch_size_);
    cudaStreamSynchronize(0);
}

template<typename T>
void LlamaBatch<T>::initializeGeneration()
{
    max_context_len_ = *std::max_element(h_context_length_buf_, h_context_length_buf_ + batch_size_);

    check_cuda_error(cudaMemsetAsync(token_ids_buf_, 0, sizeof(int) * batch_size_ * session_len_ * 2, stream_));
    invokeTransposeAxis01(token_ids_buf_, output_ids_buf_, batch_size_, session_len_, 1, stream_);
    sync_check_cuda_error();

    // token_ids_buf_[s, b]
    // ABCDe            ABCDe     e
    // ABCDEFGHIJk      ABCDEFGHIJk
    // ABCDEFGHi    ->  ABCDEFGHi i
    // ABCDEFGh         ABCDEFGh  h
    // ABCd             ABCd      d
    for (int i = 0; i < batch_size_; ++i) {
        auto token_ids = token_ids_buf_ + i;
        auto p_src     = h_context_length_buf_[i] - 1;
        auto p_dst     = max_context_len_ - 1;
        if (p_src != p_dst) {  // dst and src of `cudaMemcpyAsync` must not overlap
            check_cuda_error(cudaMemcpyAsync(token_ids + p_dst * batch_size_,
                                             token_ids + p_src * batch_size_,
                                             sizeof(int),
                                             cudaMemcpyDefault,
                                             stream_));
        }
    }

    check_cuda_error(cudaMemcpyAsync(
        context_length_buf_, h_context_length_buf_, sizeof(int) * batch_size_, cudaMemcpyDefault, stream_));
    check_cuda_error(cudaMemcpyAsync(
        k_cache_ptr_buf_, h_k_cache_ptr_buf_, sizeof(uintptr_t) * batch_size_, cudaMemcpyDefault, stream_));
    check_cuda_error(cudaMemcpyAsync(
        v_cache_ptr_buf_, h_v_cache_ptr_buf_, sizeof(uintptr_t) * batch_size_, cudaMemcpyDefault, stream_));

    check_cuda_error(
        cudaMemcpyAsync(sequence_lengths_, context_length_buf_, sizeof(int) * batch_size_, cudaMemcpyDefault, stream_));
    // `sequence_lengths_` will be increased by dynamic decode
    // note that in decoder and in output "sequence length" has different semantic
    // - in decoder it means length of sequence that has kv cache already computed
    // - in output it means length of all tokens (the last generated token does not have k/v cache computed yet)
    invokePlusScalar(sequence_lengths_, -1, batch_size_, stream_);
    sync_check_cuda_error();

    // total_padding_count_
    // decoding starts at max_context_len
    check_cuda_error(cudaMemsetAsync(total_padding_count_, 0, sizeof(int) * batch_size_, stream_));
    invokeUpdatePaddingCount(total_padding_count_,  //
                             context_length_buf_,
                             max_context_len_,
                             batch_size_,
                             1,
                             stream_);
    sync_check_cuda_error();

    // seq_limit_len_, will be compared to `step` instead of `sequence_length`, so padding len should be accounted for
    for (int i = 0; i < batch_size_; ++i) {
        h_seq_limit_len_[i] = request_seq_len_limit_[i] + (max_context_len_ - h_context_length_buf_[i]);
        // mask finished sequences
        h_finished_buf_[i] = max_context_len_ >= h_seq_limit_len_[i];
    }
    check_cuda_error(
        cudaMemcpyAsync(seq_limit_len_, h_seq_limit_len_, sizeof(uint32_t) * batch_size_, cudaMemcpyDefault, stream_));
    check_cuda_error(
        cudaMemcpyAsync(finished_buf_, h_finished_buf_, sizeof(bool) * batch_size_, cudaMemcpyDefault, stream_));

    // ! range of step_ [1, 2 * session_len]
    // consider a sequence with context_len == session_len and another sequence with context_len == 1 and
    // request_output_len == session_len - 1 => step_ will loop in [session_len, 2 * session_len)
    step_ = max_context_len_;

    if (rank_ == 0) {
        TM_LOG_INFO("[initGen] batch_size = %d", (int)batch_size_);
        TM_LOG_INFO("[initGen] max_context_len = %d", (int)max_context_len_);

        TM_LOG_INFO("[initGen] slot  sequence_id  context_len  seq_limit_len  finished");
        for (int i = 0; i < batch_size_; ++i) {
            TM_LOG_INFO("[initGen] %4d  %11ld  %11d  %13d  %8d",
                        i,
                        (long)cached_seq_[i].id,
                        h_context_length_buf_[i],
                        (int)h_seq_limit_len_[i],
                        (int)h_finished_buf_[i]);
        }
    }
}

template<typename T>
bool LlamaBatch<T>::generate()
{
    constexpr int kLogInterval = 10;
    if (rank_ == 0 && (step_ - 1) % kLogInterval == 0) {
        TM_LOG_INFO("------------------------- step = %d -------------------------", step_ - 1);
    }

    const bool is_first_step = step_ == max_context_len_;

    std::vector<int> prev;
    if (debug_ && rank_ == 0 && is_first_step) {
        prev.resize(batch_size_);
        cudaMemcpyAsync(prev.data(),
                        token_ids_buf_ + (step_ - 1) * batch_size_,
                        sizeof(int) * batch_size_,
                        cudaMemcpyDefault,
                        stream_);
    }

    // embeddingLookup(step_ - 1);
    llama_->embeddingLookup(decoder_input_buf_,  //
                            token_ids_buf_,
                            batch_size_,
                            step_ - 1);

    llama_->decoderForward(decoder_output_buf_,
                           k_cache_ptr_buf_,
                           v_cache_ptr_buf_,
                           decoder_input_buf_,
                           sequence_lengths_,
                           total_padding_count_,
                           finished_buf_,
                           step_,
                           0,
                           session_len_,
                           batch_size_);

    llama_->postDecodeEmbedding(logits_buf_,  //
                                local_logits_buf_,
                                decoder_output_buf_,
                                batch_size_);

    // stop-words & bad-words require the matched tokens to be contiguous, so item size > 1 is
    // not supported yet.
    bool should_stop{};
    llama_->dynamicDecode(token_ids_buf_,
                          finished_buf_,
                          sequence_lengths_,
                          &should_stop,
                          &inputs_,
                          &outputs_,
                          logits_buf_,
                          seq_limit_len_,
                          context_length_buf_,
                          end_ids_buf_,
                          step_,
                          0,
                          max_context_len_,
                          session_len_ * 2,
                          batch_size_);

    if (debug_ && rank_ == 0) {
        std::vector<int> curr(batch_size_);

        cudaMemcpyAsync(
            curr.data(), token_ids_buf_ + step_ * batch_size_, sizeof(int) * batch_size_, cudaMemcpyDefault, stream_);
        cudaStreamSynchronize(stream_);

        if (is_first_step) {
            std::stringstream sprev;
            for (int k = 0; k < prev.size(); ++k) {
                sprev << std::setw(6) << prev[k];
            }
            TM_LOG_INFO("[ lookup ] step = %d, [%s]", step_ - 1, sprev.str().c_str());
        }

        std::stringstream scurr;
        for (int k = 0; k < curr.size(); ++k) {
            scurr << std::setw(6) << curr[k];
        }
        TM_LOG_INFO("[generate] step = %d, [%s]", step_ - 1, scurr.str().c_str());
    }

    ////////////////////////////////////////////////
    /// ! increase the step counter
    ++step_;

    return !should_stop;
}

template<typename T>
void LlamaBatch<T>::initialize(const std::vector<std::shared_ptr<Request>>& infer_requests)
{
    FT_CHECK(batch_size_ + infer_requests.size() <= max_batch_size_);

    const int infer_request_count = infer_requests.size();

    allocateBuffer(batch_size_ + infer_request_count, session_len_);

    // handle infer requests
    std::vector<int>       tmp_input_length(infer_request_count);
    std::vector<CachedSeq> tmp_cached_seq;
    tmp_cached_seq.reserve(infer_request_count);

    int tmp_max_input_length = 0;
    for (int i = 0; i < infer_request_count; ++i) {
        auto& r = *infer_requests[i];

        LlamaCacheManager::Sequence seq{};
        if (r.start_flag) {
            seq = llama_->kv_cache_mgr_->create(r.id, stream_);
        }
        else {
            seq = llama_->kv_cache_mgr_->fetch(r.id, stream_);
        }

        const int step = r.inputs[rank_].getVal<int>("step", -1);
        if (step >= 0) {
            if (step <= seq.token_ids.size()) {
                seq.token_ids.resize(step);
                seq.cache_len = std::min(seq.cache_len, (size_t)step);
            }
            else if (rank_ == 0) {
                TM_LOG_WARNING("[initialize] Skipping invalid step (%d) setting for ID %ld", step, (long)seq.id);
            }
        }

        // input length with missing cache accounted for
        int actual_input_len = r.inputs[rank_].getVal<int>("input_lengths") + (seq.token_ids.size() - seq.cache_len);

        // insert `start_id` for empty sequences
        if (seq.token_ids.empty() && actual_input_len == 0) {
            seq.token_ids.push_back(llama_->start_id_);
            seq.cache_len    = 0;
            actual_input_len = seq.token_ids.size() - seq.cache_len;
        }

        tmp_input_length[i] = actual_input_len;

        tmp_max_input_length = std::max((int)tmp_max_input_length, actual_input_len);
        tmp_cached_seq.push_back(std::move(seq));
    }

    FT_CHECK(tmp_max_input_length > 0);
    const int max_input_length = tmp_max_input_length;

    // arrange requests in ascending order w.r.t actual input lengths, so that requests need context decoding will
    // be together
    {
        std::vector<int> idxs(tmp_input_length.size());
        std::iota(idxs.begin(), idxs.end(), 0);
        std::sort(idxs.begin(), idxs.end(), [&](int i, int j) { return tmp_input_length[i] < tmp_input_length[j]; });
        for (int i = 0; i < idxs.size(); ++i) {
            requests_[batch_size_ + i]   = infer_requests[idxs[i]];
            cached_seq_[batch_size_ + i] = tmp_cached_seq[idxs[i]];
        }
    }

    const int count = batch_size_ + infer_requests.size();

    std::vector<int> tmp_input_len(count);

    for (int i = batch_size_; i < count; ++i) {
        const auto& seq = cached_seq_[i];

        h_input_length_buf_[i] = requests_[i]->inputs[rank_].getVal<int>("input_lengths");
        tmp_input_len[i]       = h_input_length_buf_[i];
        // prepare output ids
        // <--------> max_context_len
        // aaaAAAA
        // bbbbBBBBBB
        // ccCCC
        auto output_ids_ptr = output_ids_buf_ + i * session_len_;

        // clear the persistent buffer to prevent leaking previous conversation
        check_cuda_error(cudaMemsetAsync(output_ids_ptr, 0, sizeof(int) * session_len_, stream_));

        if (!seq.token_ids.empty()) {
            check_cuda_error(cudaMemcpyAsync(output_ids_ptr,  //
                                             seq.token_ids.data(),
                                             sizeof(int) * seq.token_ids.size(),
                                             cudaMemcpyDefault,
                                             stream_));
            output_ids_ptr += seq.token_ids.size();
        }

        if (h_input_length_buf_[i]) {
            auto input_ids_ptr = requests_[i]->inputs[rank_].getPtr<int>("input_ids");
            check_cuda_error(cudaMemcpyAsync(output_ids_ptr,  //
                                             input_ids_ptr,
                                             sizeof(int) * h_input_length_buf_[i],
                                             cudaMemcpyDefault,
                                             stream_));
        }

        if (!requests_[i]->start_flag && !seq.random_state_.empty()) {
            check_cuda_error(cudaMemcpyAsync((curandState_t*)topk_curandstate_buf_ + i,
                                             seq.random_state_.data(),
                                             sizeof(curandState_t),
                                             cudaMemcpyDefault,
                                             stream_));
            check_cuda_error(cudaMemcpyAsync((curandState_t*)topp_curandstate_buf_ + i,
                                             seq.random_state_.data() + sizeof(curandState_t),
                                             sizeof(curandState_t),
                                             cudaMemcpyDefault,
                                             stream_));
        }
    }

    for (int i = batch_size_; i < count; ++i) {
        const auto& seq           = cached_seq_[i];
        const int   missed        = (int)seq.token_ids.size() - seq.cache_len;
        auto        input_ids_buf = input_ids_buf_ + i * session_len_;
        FT_CHECK(missed >= 0);
        if (missed > 0) {
            check_cuda_error(cudaMemcpyAsync(input_ids_buf,  //
                                             seq.token_ids.data() + seq.cache_len,
                                             sizeof(int) * missed,
                                             cudaMemcpyDefault,
                                             stream_));
            input_ids_buf += missed;
        }
        auto& input_ids = requests_[i]->inputs[rank_].at("input_ids");
        check_cuda_error(cudaMemcpyAsync(input_ids_buf,  //
                                         input_ids.getPtr<int>(),
                                         sizeof(int) * h_input_length_buf_[i],
                                         cudaMemcpyDefault,
                                         stream_));
        h_input_length_buf_[i] += missed;
        h_history_length_buf_[i] = seq.cache_len;
        h_context_length_buf_[i] = h_input_length_buf_[i] + h_history_length_buf_[i];

        const int request_output_len = requests_[i]->inputs[rank_].getVal<int>("request_output_len");
        request_seq_len_limit_[i]    = h_context_length_buf_[i] + request_output_len;
        // `length_criterion` sets finish flag when step >= seq_limit_len, however when step == seq_limit_len
        // the actual sequence length is seq_limit_len + 1, hence seq_limit_len must truncated to session_len - 1
        if (request_seq_len_limit_[i] >= session_len_) {
            request_seq_len_limit_[i] = session_len_ - 1;
            if (rank_ == 0) {
                const int trunc_output_len = request_seq_len_limit_[i] - h_context_length_buf_[i];
                TM_LOG_WARNING(
                    "[initialize] [%ld] total sequence length (%d + %d) exceeds session_len (%d), request_output_len is truncated to %d",
                    (long)seq.id,
                    h_context_length_buf_[i],
                    request_output_len,
                    (int)session_len_,
                    trunc_output_len);
            }
        }

        h_k_cache_ptr_buf_[i] = (uint64_t)seq.k_cache;
        h_v_cache_ptr_buf_[i] = (uint64_t)seq.v_cache;
    }

    const int max_context_len = *std::max_element(h_context_length_buf_ + batch_size_, h_context_length_buf_ + count);

    batch_size_      = count;
    max_context_len_ = max_context_len;
    step_            = max_context_len;

    check_cuda_error(
        cudaMemcpyAsync(input_length_buf_, h_input_length_buf_, sizeof(int) * batch_size_, cudaMemcpyDefault, stream_));
    check_cuda_error(cudaMemcpyAsync(
        history_length_buf_, h_history_length_buf_, sizeof(int) * batch_size_, cudaMemcpyDefault, stream_));
    check_cuda_error(cudaMemcpyAsync(
        context_length_buf_, h_context_length_buf_, sizeof(int) * batch_size_, cudaMemcpyDefault, stream_));
    check_cuda_error(cudaMemcpyAsync(
        k_cache_ptr_buf_, h_k_cache_ptr_buf_, sizeof(uintptr_t) * batch_size_, cudaMemcpyDefault, stream_));
    check_cuda_error(cudaMemcpyAsync(
        v_cache_ptr_buf_, h_v_cache_ptr_buf_, sizeof(uintptr_t) * batch_size_, cudaMemcpyDefault, stream_));

    if (llama_->tensor_para_.rank_ == 0) {
        TM_LOG_INFO("[init] infer_request_count = %d", (int)infer_request_count);
        TM_LOG_INFO("[init] batch_size = %d", (int)batch_size_);
        TM_LOG_INFO("[init] session_len = %d", (int)session_len_);
        TM_LOG_INFO("[init] max_input_length = %d", (int)max_input_length);
        TM_LOG_INFO("[init] max_context_len = %d", (int)max_context_len);
        TM_LOG_INFO(
            "[init] slot  sequence_id  history_len  input_len  context_len  tmp_input_len  token_ids.size  cache_len");
        for (int i = batch_size_ - infer_request_count; i < batch_size_; ++i) {
            TM_LOG_INFO("[init] %4d  %11ld  %11d  %9d  %11d  %13d  %14d  %9d",
                        i,
                        (int)cached_seq_[i].id,
                        h_history_length_buf_[i],
                        h_input_length_buf_[i],
                        h_context_length_buf_[i],
                        tmp_input_len[i],
                        (int)cached_seq_[i].token_ids.size(),
                        (int)cached_seq_[i].cache_len);
        }
    }
}

template<typename T>
void LlamaBatch<T>::contextDecode()
{
    int base = -1;
    for (int i = 0; i < batch_size_; ++i) {
        if (h_input_length_buf_[i] > 1) {
            base = i;
            break;
        }
    }
    if (base >= 0) {
        check_cuda_error(cudaStreamSynchronize(stream_));
        const auto tick = std::chrono::high_resolution_clock::now();

        const int context_decode_count = batch_size_ - base;
        if (rank_ == 0) {
            TM_LOG_INFO("[decodeContext] base = %d, count = %d", base, context_decode_count);
        }
        invokePlusScalar(input_length_buf_ + base, -1, context_decode_count, stream_);
        invokePlusScalar(context_length_buf_ + base, -1, context_decode_count, stream_);

        auto get_input_len   = [this](int index) { return h_input_length_buf_[index] - 1; };
        auto get_context_len = [this](int index) { return h_context_length_buf_[index] - 1; };

        std::vector<int> decode_indices{base};
        std::vector<int> decode_lengths{get_input_len(base)};

        auto token_num       = get_input_len(base);
        auto max_input_len   = get_input_len(base);
        auto max_context_len = get_context_len(base);
        auto offset          = base;
        for (int i = offset + 1; i <= batch_size_; ++i) {
            if (i == batch_size_ || token_num + h_context_length_buf_[i] > max_context_token_num_) {
                const int context_decode_batch_size = i - offset;
                if (rank_ == 0) {
                    TM_LOG_INFO(
                        "[decodeContext] offset = %d, batch_size = %d, token_num = %d, max_input_len = %d, max_context_len = %d",
                        base,
                        context_decode_batch_size,
                        token_num,
                        max_input_len,
                        max_context_len);
                }
                // construct context_decoder_ids w/o padding
                // aaaa____
                // bb______ -> aaaabbcccccccc
                // cccccccc
                auto context_decoder_ids = context_decoder_ids_buf_;
                for (int j = offset; j < i; ++j) {
                    check_cuda_error(cudaMemcpyAsync(context_decoder_ids,
                                                     input_ids_buf_ + j * session_len_,
                                                     sizeof(int) * get_input_len(j),
                                                     cudaMemcpyDefault,
                                                     stream_));
                    context_decoder_ids += get_input_len(j);
                }
                llama_->contextDecode(nullptr,
                                      k_cache_ptr_buf_ + offset,
                                      v_cache_ptr_buf_ + offset,
                                      context_decoder_input_buf_,
                                      context_decoder_output_buf_,
                                      context_decoder_ids_buf_,
                                      input_length_buf_ + offset,
                                      history_length_buf_ + offset,
                                      context_length_buf_ + offset,
                                      token_num,
                                      max_input_len,
                                      max_context_len,
                                      session_len_,
                                      context_decode_batch_size);

                // compute logits of inputs if requested
                outputContextLogits(context_decoder_output_buf_, decode_indices, decode_lengths);

                if (i < batch_size_) {
                    // initialize next sub-batch
                    token_num       = get_input_len(i);
                    max_input_len   = get_input_len(i);
                    max_context_len = get_context_len(i);
                    offset          = i;

                    decode_indices = {i};
                    decode_lengths = {get_input_len(i)};
                }
            }
            else {
                // add to current sub-batch
                token_num += get_input_len(i);
                max_input_len   = std::max(max_input_len, get_input_len(i));
                max_context_len = std::max(max_context_len, get_context_len(i));

                decode_indices.push_back(i);
                decode_lengths.push_back(get_input_len(i));
            }
        }

        invokePlusScalar(context_length_buf_ + base, 1, context_decode_count, stream_);
        invokePlusScalar(input_length_buf_ + base, 1, context_decode_count, stream_);

        for (int i = offset; i < batch_size_; ++i) {
            h_input_length_buf_[i] = 0;
        }

        check_cuda_error(cudaStreamSynchronize(stream_));
        const auto tock = std::chrono::high_resolution_clock::now();
        if (rank_ == 0) {
            TM_LOG_INFO("[decodeContext] %.2f ms", std::chrono::duration<float, std::milli>(tock - tick).count());
        }
    }
    else if (rank_ == 0) {
        TM_LOG_INFO("[decodeContext] Context decoding is not needed.");
    }
}

template<typename T>
void LlamaBatch<T>::outputContextLogits(T*                      context_decoder_output,
                                        const std::vector<int>& indices,
                                        const std::vector<int>& lengths)
{
    std::vector<float*> output_logits;
    int                 num_token = 0;
    {
        bool is_return_logits = false;
        for (int k = 0; k < indices.size(); ++k) {
            auto& request = requests_[indices[k]];
            output_logits.push_back(request->outputs[rank_].getPtr<float>("logits", nullptr));
            num_token += lengths[k];
            if (output_logits.back()) {
                is_return_logits = true;
            }
        }
        if (!is_return_logits) {
            return;
        }
    }

    if (context_logits_buf_ == nullptr) {
        NcclGuard guard(llama_->tensor_para_, stream_, true);
        context_logits_buf_ =
            (float*)allocator_->malloc(sizeof(float) * llama_->vocab_size_padded_ * max_context_token_num_);
        const auto tp = llama_->tensor_para_.world_size_;
        if (tp > 1) {
            FT_CHECK(llama_->vocab_size_padded_ % tp == 0);
            const auto local_vocab_size = llama_->vocab_size_padded_ / tp;
            local_context_logits_buf_ =
                (float*)allocator_->malloc(sizeof(float) * local_vocab_size * max_context_token_num_);
        }
    }

    llama_->postDecodeEmbedding(context_logits_buf_, local_context_logits_buf_, context_decoder_output, num_token);

    auto logits = context_logits_buf_;

    for (int k = 0; k < indices.size(); ++k) {
        if (output_logits[k]) {
            check_cuda_error(cudaMemcpyAsync(output_logits[k],
                                             logits,
                                             sizeof(float) * llama_->vocab_size_ * lengths[k],
                                             cudaMemcpyDefault,
                                             stream_));
        }
        logits += llama_->vocab_size_padded_ * lengths[k];
    }
}

template<typename T>
void LlamaBatch<T>::finish()
{
    // secure info needed by `synchronize()`
    check_cuda_error(
        cudaMemcpyAsync(h_finished_buf_, finished_buf_, sizeof(bool) * batch_size_, cudaMemcpyDefault, stream_));
    check_cuda_error(
        cudaMemcpyAsync(h_sequence_lengths_, sequence_lengths_, sizeof(int) * batch_size_, cudaMemcpyDefault, stream_));

    setOutputTensors(step_);

    check_cuda_error(cudaStreamSynchronize(stream_));

    if (rank_ == 0 && llama_->ffi_lock_) {
        llama_->ffi_lock_(1);
    }
    for (int i = 0; i < batch_size_; ++i) {
        FT_CHECK(requests_[i] != nullptr);
        if (requests_[i]->stream_cb && rank_ == 0) {
            requests_[i]->stream_cb(&requests_[i]->outputs[rank_].get());
        }
    }
    if (rank_ == 0 && llama_->ffi_lock_) {
        llama_->ffi_lock_(0);
    }

    if (debug_ && rank_ == 0) {
        std::stringstream ss;
        for (int i = 0; i < batch_size_; ++i) {
            ss << (i ? ", " : "") << "(" << h_sequence_lengths_[i] << "," << h_finished_buf_[i] << ")";
        }
        TM_LOG_INFO("[finish] [%s]", ss.str().c_str());
    }

    for (int i = 0; i < batch_size_; ++i) {
        if (h_finished_buf_[i]) {
            finishRequest(i, false);
            ++finished_count_;
        }
    }
}

template<typename T>
void LlamaBatch<T>::synchronize()
{
    // compact
    int idx = 0;
    for (int i = 0; i < batch_size_; ++i) {
        if (requests_[i]) {
            h_input_length_buf_[idx]   = 0;
            h_history_length_buf_[idx] = 0;

            h_context_length_buf_[idx] = h_sequence_lengths_[i] + 1;
            h_sequence_lengths_[idx]   = h_context_length_buf_[idx];

            check_cuda_error(cudaMemcpyAsync((curandState_t*)topk_curandstate_buf_ + idx,
                                             llama_->dynamic_decode_layer_->topk_curandstate_buf() + i,
                                             sizeof(curandState_t),
                                             cudaMemcpyDefault,
                                             stream_));
            check_cuda_error(cudaMemcpyAsync((curandState_t*)topp_curandstate_buf_ + idx,
                                             llama_->dynamic_decode_layer_->topp_curandstate_buf() + i,
                                             sizeof(curandState_t),
                                             cudaMemcpyDefault,
                                             stream_));

            if (i != idx) {
                h_finished_buf_[idx]        = h_finished_buf_[i];
                request_seq_len_limit_[idx] = request_seq_len_limit_[i];

                h_k_cache_ptr_buf_[idx] = h_k_cache_ptr_buf_[i];
                h_v_cache_ptr_buf_[idx] = h_v_cache_ptr_buf_[i];

                requests_[idx]   = std::move(requests_[i]);
                cached_seq_[idx] = std::move(cached_seq_[i]);
                check_cuda_error(cudaMemcpyAsync(output_ids_buf_ + idx * session_len_,
                                                 output_ids_buf_ + i * session_len_,
                                                 sizeof(int) * h_context_length_buf_[idx],
                                                 cudaMemcpyDefault,
                                                 stream_));
            }
            ++idx;
        }
    }
    batch_size_ = idx;

    if (rank_ == 0) {
        TM_LOG_INFO("[synchronize] batch_size = %d", (int)batch_size_);
    }

    finished_count_ = 0;
}

template<typename T>
void LlamaBatch<T>::setOutputTensors(int max_gen_step)
{
    // [s,b] -> [b,s] and skip padding in [context_len, max_context_len)
    invokeGatherOutput(output_ids_buf_,
                       token_ids_buf_,
                       context_length_buf_,
                       max_context_len_,
                       max_gen_step,
                       session_len_,
                       batch_size_,
                       stream_);
    sync_check_cuda_error();

    /// TODO: fuse the loop into a single kernel
    for (int i = 0; i < batch_size_; ++i) {
        if (requests_[i]) {
            auto& output_ids      = requests_[i]->outputs[rank_].at("output_ids");
            auto& sequence_length = requests_[i]->outputs[rank_].at("sequence_length");
            check_cuda_error(cudaMemcpyAsync(output_ids.getPtr<int>(),
                                             output_ids_buf_ + i * session_len_,
                                             sizeof(int) * output_ids.shape.at(2),
                                             cudaMemcpyDefault,
                                             stream_));
            check_cuda_error(cudaMemcpyAsync(
                sequence_length.getPtr<int>(), sequence_lengths_ + i, sizeof(int), cudaMemcpyDefault, stream_));
            if (max_gen_step > max_context_len_) {  // +1 for newly generated token
                invokePlusScalar(sequence_length.getPtr<int>(), 1, 1, stream_);
            }
        }
    }
}

template<typename T>
void LlamaBatch<T>::finishRequest(int index, bool force_end)
{
    if (rank_ == 0) {
        TM_LOG_INFO("[finishRequest] slot = %d, id = %lu", index, (long)requests_[index]->id);
    }

    if (debug_ && rank_ == 0) {
        std::vector<int> tokens(h_sequence_lengths_[index] + 1);
        cudaMemcpyAsync(tokens.data(),
                        output_ids_buf_ + index * session_len_,
                        sizeof(int) * tokens.size(),
                        cudaMemcpyDefault,
                        stream_);
        cudaStreamSynchronize(stream_);
        std::stringstream ss;
        for (const auto& t : tokens) {
            ss << " " << t;
        }
        TM_LOG_INFO("[finishRequest] slot %d, tokens [%s]", index, ss.str().c_str());
    }

    auto&      output_ids_tensor = requests_[index]->outputs[rank_].at("output_ids");
    const auto output_ids_data   = output_ids_tensor.getPtr<int>();
    if (requests_[index]->end_flag || force_end) {
        llama_->kv_cache_mgr_->erase(requests_[index]->id);
    }
    else {
        // the last generated token is not processed by decoder thus dont have k/v cache
        const int n_steps    = step_ - max_context_len_;
        const int cache_len  = h_sequence_lengths_[index];
        const int output_len = n_steps > 0 ? cache_len + 1 : cache_len;

        auto& seq = cached_seq_[index];

        seq.cache_len = cache_len;

        // update token IDs
        seq.token_ids.resize(output_len);
        check_cuda_error(cudaMemcpyAsync(
            seq.token_ids.data(), output_ids_data, sizeof(int) * output_len, cudaMemcpyDefault, stream_));

        // update random states
        seq.random_state_.resize(sizeof(curandState_t) * 2);
        check_cuda_error(cudaMemcpyAsync(seq.random_state_.data(),
                                         llama_->dynamic_decode_layer_->topk_curandstate_buf() + index,
                                         sizeof(curandState_t),
                                         cudaMemcpyDefault,
                                         stream_));
        check_cuda_error(cudaMemcpyAsync(seq.random_state_.data() + sizeof(curandState_t),
                                         llama_->dynamic_decode_layer_->topp_curandstate_buf() + index,
                                         sizeof(curandState_t),
                                         cudaMemcpyDefault,
                                         stream_));

        check_cuda_error(cudaStreamSynchronize(stream_));

        llama_->kv_cache_mgr_->update(cached_seq_[index], stream_);
    }

    // When the signal is set threads from LlamaV2::forward can exit
    // and free inputs/outputs tensors.
    // Therefore we need to make sure that no threads from LlamaV2::internalThreadEntry
    // are accessing the tensors.
    llama_->shared_state_->barrier->wait();
    if (rank_ == 0) {
        requests_[index]->signal.set_value(0);
    }

    requests_[index] = nullptr;
}

template class LlamaBatch<half>;
template class LlamaBatch<float>;

}  // namespace turbomind
