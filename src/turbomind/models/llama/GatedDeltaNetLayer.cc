#include "src/turbomind/models/llama/GatedDeltaNetLayer.h"
#include "src/turbomind/core/allocator.h"
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
    norm_eps_(model.norm_eps),
    dtype_(model.data_type),
    state_dtype_(model.linear_state_dtype),
    linear_(*ctx.linear)
{
    layer_types_       = model.layer_types;
    num_linear_layers_ = 0;
    for (auto t : layer_types_) {
        if (t == 1)
            ++num_linear_layers_;
    }

    TM_LOG_INFO("GatedDeltaNetLayer: num_k=%d num_v=%d k_dim=%d v_dim=%d "
                "conv_dim=%d d_conv=%d num_linear_layers=%d",
                num_k_heads_,
                num_v_heads_,
                key_dim_,
                value_dim_,
                conv_dim_,
                d_conv_,
                num_linear_layers_);

    if (num_linear_layers_ > 0) {
        conv_state_ptrs_buf_      = {engine.max_batch_size, kCPUpinned};
        recurrent_state_ptrs_buf_ = {engine.max_batch_size, kCPUpinned};
    }

    for (int i = 0; i < phases; ++i) {
        data_.emplace_back();
        if (num_linear_layers_ > 0) {
            data_.at(i).conv_state_ptrs      = empty_like(conv_state_ptrs_buf_, kDEVICE);
            data_.at(i).recurrent_state_ptrs = empty_like(recurrent_state_ptrs_buf_, kDEVICE);
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
}

void GatedDeltaNetLayer::Setup(int phase, TensorMap& env)
{
    auto&       d = data_.at(phase);
    const auto& b = *env.at("batch").data<BatchData*>()[0];

    d.batch_size = b.rc.size();
    d.rc.resize(d.batch_size);
    d.input_lens.resize(d.batch_size);

    d.conv_states.resize(d.batch_size);
    d.recurrent_states.resize(d.batch_size);

    for (int i = 0; i < d.batch_size; ++i) {
        d.rc[i]         = b.rc[i].get();
        d.input_lens[i] = b.rc[i]->input_len;

        auto& s = *b.rc[i]->seq;

        if (!s.recurrent_states) {
            // Create new conv/recurrent states
            s.conv_states = {{num_linear_layers_, d_conv_, conv_dim_}, dtype_, kDEVICE};
            Clear(s.conv_states);
            s.recurrent_states = {
                {num_linear_layers_, num_v_heads_, key_head_dim_, value_head_dim_}, state_dtype_, kDEVICE};
            Clear(s.recurrent_states);
        }

        // if (!d.rc[i]->autoregres) {
        if (b.rc[i]->req->session.end_flag) {  // non-stateful request
            // Session is ending, moving past the last step is OK
            d.conv_states[i]      = b.rc[i]->seq->conv_states;
            d.recurrent_states[i] = b.rc[i]->seq->recurrent_states;
        }
        else if (data_.size() > 1) {  // stateful request && async mode
            d.conv_states[i]      = empty_like(s.conv_states);
            d.recurrent_states[i] = empty_like(s.recurrent_states);
            Copy(s.conv_states, d.conv_states[i]);
            Copy(s.recurrent_states, d.recurrent_states[i]);
        }
        // }

        conv_state_ptrs_buf_[i]      = d.conv_states[i].raw_data();
        recurrent_state_ptrs_buf_[i] = d.recurrent_states[i].raw_data();
    }

    Copy(conv_state_ptrs_buf_, d.batch_size, d.conv_state_ptrs);
    Copy(recurrent_state_ptrs_buf_, d.batch_size, d.recurrent_state_ptrs);
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

        const int state_layer_idx              = linear_layer_index(p.layer_id, layer_types_);
        const int conv_state_layer_offset      = state_layer_idx * (conv_dim_ * d_conv_);
        const int recurrent_state_layer_offset = state_layer_idx * (num_v_heads_ * key_head_dim_ * value_head_dim_);

        // ----- 3a. Fused Causal Conv1d + SiLU (all requests) -----
        // all_proj carries the non-contiguous qkv slice (stride = all_col);
        // in_stride is derived from all_proj.stride(0) inside the launcher.
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
        sync_check_cuda_error();

        // ----- 3b. Gated Delta Rule -----
        // Requests are sorted by input_len: decode (seq_len==1) first, prefill last.
        // Find the split point and dispatch each half to its optimal kernel.
        // When both are present, run them concurrently on separate streams.
        {
            int decode_count = 0;
            for (int i = 0; i < pd.batch_size; ++i) {
                if (pd.input_lens[i] <= 1)
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
                // invokeChunkedGatedDeltaRuleBatched
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
