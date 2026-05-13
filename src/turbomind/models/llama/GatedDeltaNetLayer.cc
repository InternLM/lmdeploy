#include "src/turbomind/models/llama/GatedDeltaNetLayer.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/gated_delta_net_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

GatedDeltaNetLayer::GatedDeltaNetLayer(DataType                state_dtype,
                                       const std::vector<int>& layer_types,
                                       const EngineParam&      engine,
                                       const Context&          ctx,
                                       int                     phases):
    tp_size_(engine.attn_tp_size), num_linear_layers_(0), state_dtype_(state_dtype), linear_(*ctx.linear)
{
    layer_types_ = layer_types;
    for (auto t : layer_types_) {
        if (t == 1)
            ++num_linear_layers_;
    }

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
        Buffer_<RequestCache*> rc = env.at("requests").buffer();
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

        const auto& w              = *p.weights;
        const int   num_k_heads    = w.num_k_heads / tp_size_;
        const int   num_v_heads    = w.num_v_heads / tp_size_;
        const int   key_head_dim   = w.key_head_dim;
        const int   value_head_dim = w.value_head_dim;
        const int   d_conv         = w.d_conv;
        const int   key_dim        = num_k_heads * key_head_dim;
        const int   value_dim      = num_v_heads * value_head_dim;
        const int   conv_dim       = key_dim * 2 + value_dim;

        // =================================================================
        // 1. Single fused input projection: reads p.input once from HBM.
        //    Output columns are ordered: [qkv | z | b | a]
        //    where the split dims are: conv_dim, value_dim, v_heads_tp, v_heads_tp
        // =================================================================
        const int v_heads_tp = num_v_heads;  // already TP-sharded
        Tensor    all_proj   = linear_.Forward(p.input, *weights.in_proj_all);
        sync_check_cuda_error();

        // Column offsets per token (all_proj is token-major, row-major):
        //   [0, conv_dim)           -> mixed_qkv
        //   [conv_dim, +value_dim) -> z
        //   [conv_dim+value_dim, +v_heads_tp) -> b (beta logit)
        //   [conv_dim+value_dim+v_heads_tp, +v_heads_tp) -> a (alpha/dt)
        const int all_col = conv_dim + value_dim + v_heads_tp * 2;
        // const T* sub-pointers are derived per-request below; stride = all_col.

        // =================================================================
        // 2. Compute beta and g for all tokens
        //    b_raw and a_raw are sliced from the fused projection output.
        //    Stride between tokens is all_col elements.
        // =================================================================
        const int bg_total = token_num * num_v_heads;

        const int b_offset = conv_dim + value_dim;   // column offset to b logits
        const int a_offset = b_offset + v_heads_tp;  // column offset to a logits

        Tensor beta{{token_num, num_v_heads}, dtype, device};
        Tensor g{{token_num, num_v_heads}, dtype, device};

        auto b = all_proj.slice({0, b_offset}, {-1, v_heads_tp});
        auto a = all_proj.slice({0, a_offset}, {-1, v_heads_tp});

        ComputeBetaG_v2(beta, g, b, a, weights.A_log, weights.dt_bias, stream);

        // =================================================================
        // 3. Process all requests at once via batched kernel launches
        // =================================================================
        Tensor attn_out{{token_num, value_dim}, dtype, device};
        Tensor conv_out{{token_num, conv_dim}, dtype, device};

        const int state_layer_idx              = linear_layer_index(p.layer_id, layer_types_);
        const int conv_state_layer_offset      = state_layer_idx * (conv_dim * d_conv);
        const int recurrent_state_layer_offset = state_layer_idx * (num_v_heads * key_head_dim * value_head_dim);

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
                                               num_k_heads,
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
                                                   num_k_heads,
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
                                               num_k_heads,
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
                                                   num_k_heads,
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
        // Gate (z) lives at column conv_dim of all_proj with row-stride all_col.
        Tensor gate        = all_proj.slice({0, conv_dim}, {-1, value_dim});
        Tensor hidden_view = attn_out.view({token_num * num_v_heads, value_head_dim});
        invokeRMSNormGated(hidden_view, gate, weights.norm->weight, weights.norm->norm_eps_, stream);
        sync_check_cuda_error();

        // =================================================================
        // 4. Output projection (all tokens at once)
        // =================================================================
        (void)linear_.Forward(attn_out, *weights.out_proj, p.output);
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
