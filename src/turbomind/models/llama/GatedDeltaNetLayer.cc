#include "src/turbomind/models/llama/GatedDeltaNetLayer.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/gated_delta_net_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

GatedDeltaNetLayer::GatedDeltaNetLayer(const ModelParam&     model,
                                       const AttentionParam& attn,
                                       const EngineParam&    engine,
                                       int                   tp_size,
                                       const Context&        ctx,
                                       int                   phases):
    hidden_units_(model.hidden_units),
    num_k_heads_(model.linear_num_key_heads / tp_size),
    num_v_heads_(model.linear_num_value_heads / tp_size),
    key_head_dim_(model.linear_key_head_dim > 0 ? model.linear_key_head_dim : model.head_dim),
    value_head_dim_(model.linear_value_head_dim > 0 ? model.linear_value_head_dim : model.head_dim),
    d_conv_(model.linear_conv_kernel_dim > 0 ? model.linear_conv_kernel_dim : 4),
    key_dim_(num_k_heads_ * key_head_dim_),
    value_dim_(num_v_heads_ * value_head_dim_),
    conv_dim_(key_dim_ * 2 + value_dim_),
    enable_linear_prefix_caching_(engine.enable_prefix_caching),
    linear_prefix_cache_interval_tokens_(attn.cache_block_seq_len * engine.linear_prefix_cache_interval_blocks),
    norm_eps_(model.norm_eps),
    dtype_(model.data_type),
    state_dtype_(model.linear_state_dtype),
    is_warm_up_(*ctx.is_warm_up),
    linear_(*ctx.linear)
{
    layer_types_       = model.layer_types;
    num_linear_layers_ = 0;
    for (auto t : layer_types_) {
        if (t == 1)
            ++num_linear_layers_;
    }

    if (num_linear_layers_ > 0) {
        prefix_capture_state_bytes_ = static_cast<size_t>(num_linear_layers_) * d_conv_ * conv_dim_ * byte_size(dtype_)
                                      + static_cast<size_t>(num_linear_layers_) * num_v_heads_ * key_head_dim_
                                            * value_head_dim_ * byte_size(state_dtype_);
        conv_state_ptrs_buf_      = {engine.max_batch_size, kCPUpinned};
        recurrent_state_ptrs_buf_ = {engine.max_batch_size, kCPUpinned};
        if (enable_linear_prefix_caching_) {
            TM_CHECK_GT(linear_prefix_cache_interval_tokens_, 0);
            conv_capture_ptrs_buf_      = {engine.max_batch_size, kCPUpinned};
            recurrent_capture_ptrs_buf_ = {engine.max_batch_size, kCPUpinned};
        }
    }

    for (int i = 0; i < phases; ++i) {
        data_.emplace_back();
        if (num_linear_layers_ > 0) {
            data_.at(i).conv_state_ptrs      = empty_like(conv_state_ptrs_buf_, kDEVICE);
            data_.at(i).recurrent_state_ptrs = empty_like(recurrent_state_ptrs_buf_, kDEVICE);
            if (enable_linear_prefix_caching_) {
                data_.at(i).conv_capture_ptrs      = empty_like(conv_capture_ptrs_buf_, kDEVICE);
                data_.at(i).recurrent_capture_ptrs = empty_like(recurrent_capture_ptrs_buf_, kDEVICE);
            }
        }
    }

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device);
    work_counter_ = {1, kDEVICE};

    check_cuda_error(cudaStreamCreateWithPriority(&aux_stream_, cudaStreamNonBlocking, -1));
    check_cuda_error(cudaEventCreateWithFlags(&ev_before_, cudaEventDisableTiming));
    check_cuda_error(cudaEventCreateWithFlags(&ev_after_, cudaEventDisableTiming));
}

GatedDeltaNetLayer::~GatedDeltaNetLayer()
{
    cudaStreamDestroy(aux_stream_);
    cudaEventDestroy(ev_before_);
    cudaEventDestroy(ev_after_);
}

void GatedDeltaNetLayer::Run(BatchOp op, int phase, TensorMap& env)
{
    if (op == BatchOp::kAdd) {
        Buffer_<RequestCache*> rc    = env.at("requests").buffer();
        const auto             dtype = dtype_;
        for (int i = 0; i < rc.size(); ++i) {}
    }
    else if (op == BatchOp::kSetup) {
        Setup(phase, env);
    }
    else if (op == BatchOp::kPrepare) {
        auto& d     = data_.at(phase);
        d.q_offsets = env.at("q_offsets").buffer().borrow();
        d.k_offsets = env.at("k_offsets").buffer().borrow();
    }
    else if (op == BatchOp::kUpdate) {
        auto& d = data_.at(phase);
        for (int i = 0; i < d.batch_size; ++i) {
            auto& s = *d.seqs[i];
            if (d.capture_counts.empty() || d.capture_counts[i] == 0) {
                s.pending_linear_prefix_conv_states      = {};
                s.pending_linear_prefix_recurrent_states = {};
                s.pending_linear_prefix_capture_count    = 0;
                s.pending_linear_prefix_capture_base_len = 0;
                continue;
            }
            s.pending_linear_prefix_conv_states =
                d.conv_prefix_checkpoints.slice(d.capture_offsets[i], d.capture_counts[i]);
            s.pending_linear_prefix_recurrent_states =
                d.recurrent_prefix_checkpoints.slice(d.capture_offsets[i], d.capture_counts[i]);
            s.pending_linear_prefix_capture_count    = d.capture_counts[i];
            s.pending_linear_prefix_capture_base_len = d.history_lens[i];
        }
    }
}

void GatedDeltaNetLayer::Setup(int phase, TensorMap& env)
{
    auto&       d = data_.at(phase);
    const auto& b = *env.at("batch").data<BatchData*>()[0];

    d.batch_size = b.rc.size();
    d.seqs.resize(d.batch_size);
    d.input_lens.resize(d.batch_size);
    d.history_lens.resize(d.batch_size);
    d.capture_counts.assign(d.batch_size, 0);
    d.capture_offsets.assign(d.batch_size + 1, 0);
    d.total_capture_count = 0;

    d.conv_states.resize(d.batch_size);
    d.recurrent_states.resize(d.batch_size);

    for (int i = 0; i < d.batch_size; ++i) {
        d.seqs[i]         = b.rc[i]->seq;
        d.input_lens[i]   = b.rc[i]->input_len;
        d.history_lens[i] = b.rc[i]->history_len;

        auto& s = *d.seqs[i];
        TM_CHECK(s.conv_states && s.recurrent_states)
            << "Linear-attention state slot is not bound for sequence " << s.id;
        if (s.linear_states_need_reset) {
            // Reset newly assigned pooled slot state on first use. Keep GPU-side
            // state initialization out of SequenceManager.
            Clear(s.conv_states);
            Clear(s.recurrent_states);
            s.linear_states_need_reset = false;
        }

        // Linear-attention requests are restricted to stateless execution, so
        // the sequence-owned states can be passed directly here.
        d.conv_states[i]      = s.conv_states;
        d.recurrent_states[i] = s.recurrent_states;

        conv_state_ptrs_buf_[i]      = d.conv_states[i].raw_data();
        recurrent_state_ptrs_buf_[i] = d.recurrent_states[i].raw_data();

        if (enable_linear_prefix_caching_) {
            int capture_count = 0;
            // Warm-up requests never publish prefix cache entries, so avoid
            // provisioning large checkpoint buffers for synthetic prefills.
            if (!is_warm_up_ && !s.prompt.empty() && d.input_lens[i] > 0) {
                const int prompt_limit = std::min<int>(s.prompt.size(), d.history_lens[i] + d.input_lens[i]);
                if (prompt_limit > d.history_lens[i]) {
                    capture_count = prompt_limit / linear_prefix_cache_interval_tokens_
                                    - d.history_lens[i] / linear_prefix_cache_interval_tokens_;
                }
            }
            d.capture_counts[i]  = capture_count;
            d.capture_offsets[i] = d.total_capture_count;
            d.total_capture_count += capture_count;
        }
    }
    d.capture_offsets[d.batch_size] = d.total_capture_count;

    if (enable_linear_prefix_caching_) {
        if (!EnsurePrefixCaptureCapacity(d, d.total_capture_count)) {
            std::fill(d.capture_counts.begin(), d.capture_counts.end(), 0);
            std::fill(d.capture_offsets.begin(), d.capture_offsets.end(), 0);
            d.total_capture_count = 0;
        }
        for (int i = 0; i < d.batch_size; ++i) {
            const int capture_count = d.capture_counts[i];
            conv_capture_ptrs_buf_[i] =
                capture_count ? d.conv_prefix_checkpoints.slice(d.capture_offsets[i], capture_count).raw_data() :
                                nullptr;
            recurrent_capture_ptrs_buf_[i] =
                capture_count ? d.recurrent_prefix_checkpoints.slice(d.capture_offsets[i], capture_count).raw_data() :
                                nullptr;
        }
    }

    Copy(conv_state_ptrs_buf_, d.batch_size, d.conv_state_ptrs);
    Copy(recurrent_state_ptrs_buf_, d.batch_size, d.recurrent_state_ptrs);
    if (enable_linear_prefix_caching_) {
        Copy(conv_capture_ptrs_buf_, d.batch_size, d.conv_capture_ptrs);
        Copy(recurrent_capture_ptrs_buf_, d.batch_size, d.recurrent_capture_ptrs);
    }
}

bool GatedDeltaNetLayer::EnsurePrefixCaptureCapacity(Data& d, int capture_count)
{
    if (capture_count <= 0) {
        return true;
    }
    if (d.conv_prefix_checkpoints && d.conv_prefix_checkpoints.shape(0) >= capture_count
        && d.recurrent_prefix_checkpoints && d.recurrent_prefix_checkpoints.shape(0) >= capture_count) {
        return true;
    }
    if (!CanAllocatePrefixCapture(capture_count)) {
        return false;
    }
    try {
        d.conv_prefix_checkpoints      = {{capture_count, num_linear_layers_, d_conv_, conv_dim_}, dtype_, kDEVICE};
        d.recurrent_prefix_checkpoints = {
            {capture_count, num_linear_layers_, num_v_heads_, key_head_dim_, value_head_dim_}, state_dtype_, kDEVICE};
        return true;
    }
    catch (const std::exception& e) {
        if (!warned_prefix_capture_oom_) {
            TM_LOG_WARNING("[GDN] failed to allocate hybrid prefix capture staging for %d slots: %s. "
                           "This batch will run without storing new GDN prefix checkpoints.",
                           capture_count,
                           e.what());
            warned_prefix_capture_oom_ = true;
        }
        return false;
    }
}

bool GatedDeltaNetLayer::CanAllocatePrefixCapture(int capture_count)
{
    if (capture_count <= 0 || prefix_capture_state_bytes_ == 0) {
        return true;
    }

    // Prefix checkpoints are opportunistic acceleration data. Keep a generous
    // safety margin so long-context requests continue running even if we must
    // skip storing new GDN checkpoints for that batch.
    constexpr size_t kPrefixCaptureMaxBytes    = size_t{1} << 30;    // 1 GiB
    constexpr size_t kPrefixCaptureSafetyBytes = size_t{256} << 20;  // 256 MiB

    size_t free_bytes{};
    size_t total_bytes{};
    check_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
    (void)total_bytes;

    const size_t budget_bytes    = free_bytes > kPrefixCaptureSafetyBytes ?
                                       std::min(free_bytes - kPrefixCaptureSafetyBytes, kPrefixCaptureMaxBytes) :
                                       size_t{0};
    const size_t requested_bytes = static_cast<size_t>(capture_count) * prefix_capture_state_bytes_;

    if (requested_bytes > budget_bytes) {
        if (!warned_prefix_capture_budget_) {
            TM_LOG_WARNING("[GDN] skipping hybrid prefix checkpoint capture for this batch: requested %.2f MB, "
                           "budget %.2f MB. Prefix caching remains enabled, but this batch will not store new "
                           "linear-attention checkpoints.",
                           requested_bytes / (1024.0 * 1024.0),
                           budget_bytes / (1024.0 * 1024.0));
            warned_prefix_capture_budget_ = true;
        }
        return false;
    }
    return true;
}

static int linear_layer_index(int layer_id, const std::vector<int>& layer_types)
{
    int idx = 0;
    for (int i = 0; i < layer_id && i < (int)layer_types.size(); ++i) {
        if (layer_types[i] == 1)
            ++idx;
    }
    return idx;
}

void GatedDeltaNetLayer::Forward(ForwardParam p)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const int token_num = p.input.shape(0);
    if (token_num == 0)
        return;

    const auto  dtype   = p.input.dtype();
    const auto  device  = p.input.device();
    const auto  stream  = core::Context::stream().handle();
    const auto& weights = *p.weights;

    auto& pd = data_.at(p.phase);

    auto dispatch = [&](auto t) {
        using T = decltype(t);

        // =================================================================
        // 1. Single fused input projection: reads p.input once from HBM.
        //    Output columns are ordered: [qkv | z | b | a]
        //    where the split dims are: conv_dim_, value_dim_, v_heads_tp_, v_heads_tp_
        // =================================================================
        const int v_heads_tp = num_v_heads_;  // already TP-sharded
        Tensor    all_proj   = linear_.Forward(p.input, weights.in_proj_all);
        sync_check_cuda_error();

        // Column offsets per token (all_proj is token-major, row-major):
        //   [0, conv_dim_)           -> mixed_qkv
        //   [conv_dim_, +value_dim_) -> z
        //   [conv_dim_+value_dim_, +v_heads_tp) -> b (beta logit)
        //   [conv_dim_+value_dim_+v_heads_tp, +v_heads_tp) -> a (alpha/dt)
        const int all_col = conv_dim_ + value_dim_ + v_heads_tp * 2;
        // const T* sub-pointers are derived per-request below; stride = all_col.

        // =================================================================
        // 2. Compute beta and g for all tokens
        //    b_raw and a_raw are sliced from the fused projection output.
        //    Stride between tokens is all_col elements.
        // =================================================================
        const int bg_total = token_num * num_v_heads_;

        const int b_offset = conv_dim_ + value_dim_;  // column offset to b logits
        const int a_offset = b_offset + v_heads_tp;   // column offset to a logits

        Tensor beta{{token_num, num_v_heads_}, dtype, device};
        Tensor g{{token_num, num_v_heads_}, dtype, device};

        auto b = all_proj.slice({0, b_offset}, {-1, v_heads_tp});
        auto a = all_proj.slice({0, a_offset}, {-1, v_heads_tp});

        ComputeBetaG_v2(beta, g, b, a, weights.A_log, weights.dt_bias, stream);

        // =================================================================
        // 3. Process all requests at once via batched kernel launches
        // =================================================================
        Tensor attn_out{{token_num, value_dim_}, dtype, device};
        Tensor conv_out{{token_num, conv_dim_}, dtype, device};

        const int  state_layer_idx              = linear_layer_index(p.layer_id, layer_types_);
        const int  conv_state_layer_offset      = state_layer_idx * (conv_dim_ * d_conv_);
        const int  recurrent_state_layer_offset = state_layer_idx * (num_v_heads_ * key_head_dim_ * value_head_dim_);
        const int  conv_capture_stride          = num_linear_layers_ * d_conv_ * conv_dim_;
        const int  recurrent_capture_stride     = num_linear_layers_ * num_v_heads_ * key_head_dim_ * value_head_dim_;
        const bool has_prefix_captures          = enable_linear_prefix_caching_ && pd.total_capture_count > 0;

        // ----- 3a. Fused Causal Conv1d + SiLU (all requests) -----
        // all_proj carries the non-contiguous qkv slice (stride = all_col);
        // in_stride is derived from all_proj.stride(0) inside the launcher.
        if (has_prefix_captures) {
            invokeFusedConv1dSiLU(conv_out,
                                  all_proj,
                                  weights.conv1d,
                                  Tensor{},
                                  pd.conv_state_ptrs,
                                  pd.q_offsets,
                                  pd.k_offsets,
                                  pd.batch_size,
                                  conv_state_layer_offset,
                                  sm_count_,
                                  work_counter_.data(),
                                  stream,
                                  pd.conv_capture_ptrs,
                                  conv_capture_stride,
                                  linear_prefix_cache_interval_tokens_);
        }
        else {
            invokeFusedConv1dSiLU(conv_out,
                                  all_proj,
                                  weights.conv1d,
                                  Tensor{},
                                  pd.conv_state_ptrs,
                                  pd.q_offsets,
                                  pd.k_offsets,
                                  pd.batch_size,
                                  conv_state_layer_offset,
                                  sm_count_,
                                  work_counter_.data(),
                                  stream);
        }
        sync_check_cuda_error();

        // ----- 3b. Gated Delta Rule -----
        // Requests are sorted by input_len: decode (seq_len==1) first, prefill last.
        // Find the split point and dispatch each half to its optimal kernel.
        // When both are present, run them concurrently on separate streams.
        {
            int decode_count = 0;
            for (int i = 0; i < pd.batch_size; ++i) {
                if (pd.input_lens[i] <= 1 && (!has_prefix_captures || pd.capture_counts[i] == 0))
                    ++decode_count;
                else
                    break;
            }
            const int prefill_count = pd.batch_size - decode_count;

            if (decode_count > 0 && prefill_count > 0) {
                // Fork: aux_stream (high priority) waits for prior work on main stream
                check_cuda_error(cudaEventRecord(ev_before_, stream));
                check_cuda_error(cudaStreamWaitEvent(aux_stream_, ev_before_));

                // Decode on main stream
                auto dc_state = pd.recurrent_state_ptrs.slice(0, decode_count);
                auto dc_q     = pd.q_offsets.slice(0, decode_count + 1);
                invokeGatedDeltaRuleBatched_v3(attn_out,
                                               conv_out,
                                               beta,
                                               g,
                                               dc_state,
                                               dc_q,
                                               decode_count,
                                               num_k_heads_,
                                               recurrent_state_layer_offset,
                                               state_dtype_,
                                               sm_count_,
                                               work_counter_.data(),
                                               stream);

                // Prefill on aux stream (higher priority)
                auto pf_state = pd.recurrent_state_ptrs.slice(decode_count, prefill_count);
                auto pf_q     = pd.q_offsets.slice(decode_count, prefill_count + 1);
                auto pf_k     = pd.k_offsets.slice(decode_count, prefill_count + 1);
                if (has_prefix_captures) {
                    auto pf_capture = pd.recurrent_capture_ptrs.slice(decode_count, prefill_count);
                    invokeChunkedGatedDeltaRuleBatched(attn_out,
                                                       conv_out,
                                                       beta,
                                                       g,
                                                       pf_state,
                                                       pf_q,
                                                       pf_k,
                                                       prefill_count,
                                                       num_k_heads_,
                                                       recurrent_state_layer_offset,
                                                       state_dtype_,
                                                       sm_count_,
                                                       work_counter_.data(),
                                                       aux_stream_,
                                                       pf_capture,
                                                       recurrent_capture_stride,
                                                       linear_prefix_cache_interval_tokens_);
                }
                else {
                    invokeChunkedGatedDeltaRuleBatched(attn_out,
                                                       conv_out,
                                                       beta,
                                                       g,
                                                       pf_state,
                                                       pf_q,
                                                       prefill_count,
                                                       num_k_heads_,
                                                       recurrent_state_layer_offset,
                                                       state_dtype_,
                                                       sm_count_,
                                                       work_counter_.data(),
                                                       aux_stream_);
                }

                // Join: main stream waits for prefill to finish
                check_cuda_error(cudaEventRecord(ev_after_, aux_stream_));
                check_cuda_error(cudaStreamWaitEvent(stream, ev_after_));
            }
            else if (decode_count > 0) {
                auto state_slice = pd.recurrent_state_ptrs.slice(0, decode_count);
                auto q_slice     = pd.q_offsets.slice(0, decode_count + 1);
                invokeGatedDeltaRuleBatched_v3(attn_out,
                                               conv_out,
                                               beta,
                                               g,
                                               state_slice,
                                               q_slice,
                                               decode_count,
                                               num_k_heads_,
                                               recurrent_state_layer_offset,
                                               state_dtype_,
                                               sm_count_,
                                               work_counter_.data(),
                                               stream);
            }
            else if (prefill_count > 0) {
                auto state_slice = pd.recurrent_state_ptrs.slice(decode_count, prefill_count);
                auto q_slice     = pd.q_offsets.slice(decode_count, prefill_count + 1);
                auto k_slice     = pd.k_offsets.slice(decode_count, prefill_count + 1);
                if (has_prefix_captures) {
                    auto capture_slice = pd.recurrent_capture_ptrs.slice(decode_count, prefill_count);
                    invokeChunkedGatedDeltaRuleBatched(attn_out,
                                                       conv_out,
                                                       beta,
                                                       g,
                                                       state_slice,
                                                       q_slice,
                                                       k_slice,
                                                       prefill_count,
                                                       num_k_heads_,
                                                       recurrent_state_layer_offset,
                                                       state_dtype_,
                                                       sm_count_,
                                                       work_counter_.data(),
                                                       stream,
                                                       capture_slice,
                                                       recurrent_capture_stride,
                                                       linear_prefix_cache_interval_tokens_);
                }
                else {
                    invokeChunkedGatedDeltaRuleBatched(attn_out,
                                                       conv_out,
                                                       beta,
                                                       g,
                                                       state_slice,
                                                       q_slice,
                                                       prefill_count,
                                                       num_k_heads_,
                                                       recurrent_state_layer_offset,
                                                       state_dtype_,
                                                       sm_count_,
                                                       work_counter_.data(),
                                                       stream);
                }
            }
        }
        sync_check_cuda_error();

        // ----- 3c. RMSNormGated (all tokens at once) -----
        // Gate (z) lives at column conv_dim_ of all_proj with row-stride all_col.
        Tensor gate        = all_proj.slice({0, conv_dim_}, {-1, value_dim_});
        Tensor hidden_view = attn_out.view({token_num * num_v_heads_, value_head_dim_});
        invokeRMSNormGated(hidden_view, gate, weights.norm, norm_eps_, stream);
        sync_check_cuda_error();

        // =================================================================
        // 4. Output projection (all tokens at once)
        // =================================================================
        (void)linear_.Forward(attn_out, weights.out_proj, p.output);
        sync_check_cuda_error();
    };

    if (dtype == kHalf) {
        dispatch(half{});
    }
    else if (dtype == kBfloat16) {
        dispatch(nv_bfloat16{});
    }
    else {
        TM_CHECK(0) << "Unsupported dtype for GatedDeltaNetLayer";
    }
}

}  // namespace turbomind
