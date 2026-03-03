#include "src/turbomind/models/llama/GatedDeltaNetLayer.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
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

    for (int i = 0; i < phases; ++i) {
        phase_data_.push_back(std::make_shared<PhaseData>());
    }
}

GatedDeltaNetLayer::~GatedDeltaNetLayer() = default;

void GatedDeltaNetLayer::Run(BatchOp op, int phase, TensorMap& env)
{
    if (op == BatchOp::kAdd) {
        Buffer_<RequestCache*> rc    = env.at("requests").buffer();
        const auto             dtype = dtype_;
        for (int i = 0; i < rc.size(); ++i) {
            auto& c = *rc[i];
            if (num_linear_layers_ > 0) {
                c.conv_states = Tensor{{num_linear_layers_, conv_dim_, d_conv_}, dtype, kDEVICE};
                Clear(c.conv_states);
                c.recurrent_states =
                    Tensor{{num_linear_layers_, num_v_heads_, key_head_dim_, value_head_dim_}, dtype, kDEVICE};
                Clear(c.recurrent_states);
            }
        }
    }
    else if (op == BatchOp::kSetup) {
        Setup(phase, env);
    }
    else if (op == BatchOp::kPrepare) {
        auto& d     = *phase_data_.at(phase);
        d.q_offsets = env.at("q_offsets").buffer().borrow();
    }
}

void GatedDeltaNetLayer::Setup(int phase, TensorMap& env)
{
    auto&       d     = *phase_data_.at(phase);
    const auto& batch = *env.at("batch").data<BatchData*>()[0];

    d.batch_size = batch.rc.size();
    d.rc.resize(d.batch_size);
    d.input_lens.resize(d.batch_size);
    for (int i = 0; i < d.batch_size; ++i) {
        d.rc[i]         = batch.rc[i].get();
        d.input_lens[i] = batch.rc[i]->input_len;
    }
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

    auto& pd = *phase_data_.at(p.phase);

    auto dispatch = [&](auto t) {
        using T = decltype(t);

        // =================================================================
        // 1. Single fused input projection: reads p.input once from HBM.
        //    Output columns are ordered: [qkv | z | b | a]
        //    where the split dims are: conv_dim_, value_dim_, v_heads_tp_, v_heads_tp_
        // =================================================================
        const int v_heads_tp = num_v_heads_;     // already TP-sharded
        Tensor all_proj = linear_.Forward(p.input, weights.in_proj_all);
        sync_check_cuda_error();

        // Pointer-arithmetic slices — no copies, no allocations.
        auto all_data = all_proj.data<T>();
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
        const int bg_total   = token_num * num_v_heads_;
        const int b_offset   = conv_dim_ + value_dim_;       // column offset to b logits
        const int a_offset   = b_offset + v_heads_tp;        // column offset to a logits
        Tensor    beta{{token_num, num_v_heads_}, dtype, device};
        Tensor    g_tensor{{token_num, num_v_heads_}, dtype, device};

        // Gather b and a columns into contiguous buffers for the kernel.
        // Each has shape (token_num, v_heads_tp) but is strided inside all_proj.
        // Use the existing invokeComputeBetaG which needs contiguous b/a input.
        // We copy the b/a columns compactly first.
        Tensor b_contig{{token_num, v_heads_tp}, dtype, device};
        Tensor a_contig{{token_num, v_heads_tp}, dtype, device};
        check_cuda_error(cudaMemcpy2DAsync(b_contig.data<T>(),
                                           v_heads_tp * sizeof(T),
                                           all_data + b_offset,
                                           all_col * sizeof(T),
                                           v_heads_tp * sizeof(T),
                                           token_num,
                                           cudaMemcpyDeviceToDevice,
                                           stream));
        check_cuda_error(cudaMemcpy2DAsync(a_contig.data<T>(),
                                           v_heads_tp * sizeof(T),
                                           all_data + a_offset,
                                           all_col * sizeof(T),
                                           v_heads_tp * sizeof(T),
                                           token_num,
                                           cudaMemcpyDeviceToDevice,
                                           stream));

        invokeComputeBetaG(beta.data<T>(),
                           g_tensor.data<T>(),
                           b_contig.data<T>(),
                           a_contig.data<T>(),
                           weights.A_log.data<T>(),
                           weights.dt_bias.data<T>(),
                           bg_total,
                           num_v_heads_,
                           stream);
        sync_check_cuda_error();

        // =================================================================
        // 3. Process each request independently
        // =================================================================
        Tensor attn_out{{token_num, value_dim_}, dtype, device};

        int token_offset = 0;
        for (int req = 0; req < pd.batch_size; ++req) {
            auto&     rc      = *pd.rc[req];
            const int seq_len = pd.input_lens[req];
            if (seq_len == 0)
                continue;

            // Slice per-request portions from the fused projection output.
            // qkv lives in columns [0, conv_dim_) of all_proj with row-stride all_col.
            // z   lives in columns [conv_dim_, conv_dim_+value_dim_).
            // b/a are already extracted into contiguous beta/g_tensor above.
            T* qkv_row_ptr = all_data + token_offset * all_col;  // first token, col 0
            T* z_row_ptr   = all_data + token_offset * all_col + conv_dim_;
            T* beta_ptr    = beta.data<T>() + token_offset * num_v_heads_;
            T* g_ptr       = g_tensor.data<T>() + token_offset * num_v_heads_;
            T* out_ptr     = attn_out.data<T>() + token_offset * value_dim_;

            const int state_layer_idx = linear_layer_index(p.layer_id, layer_types_);

            T* conv_state_ptr      = nullptr;
            T* recurrent_state_ptr = nullptr;
            if (rc.conv_states) {
                conv_state_ptr =
                    rc.conv_states.data<T>() + state_layer_idx * (conv_dim_ * d_conv_);
            }
            if (rc.recurrent_states) {
                recurrent_state_ptr =
                    rc.recurrent_states.data<T>()
                    + state_layer_idx * (num_v_heads_ * key_head_dim_ * value_head_dim_);
            }

            // ----- 3a. Fused Causal Conv1d + SiLU -----
            // conv_out shape: (seq_len, conv_dim_) — packed [Q|K|V] per token.
            // No transpose: the new delta-rule kernels read this row-major layout directly.
            Tensor conv_out{{seq_len, conv_dim_}, dtype, device};
            invokeFusedConv1dSiLU(conv_out.data<T>(),
                                  qkv_row_ptr,
                                  weights.conv1d.data<T>(),
                                  (const T*)nullptr,
                                  conv_state_ptr,
                                  1,
                                  conv_dim_,
                                  seq_len,
                                  d_conv_,
                                  all_col,
                                  stream);
            sync_check_cuda_error();

            // ----- 3b. Gated Delta Rule -----
            // The kernels handle:
            //   • Strided Q/K/V access directly from the packed conv_out buffer
            //     (eliminates the three cudaMemcpy2DAsync strided-copy passes).
            //   • In-kernel L2 normalization of Q and K
            //     (eliminates two invokeL2Norm kernel launches).
            //   • GQA: kh = h / (num_v_heads / num_k_heads) computed per-thread
            //     (eliminates the invokeRepeatInterleave allocation).
            const int k_dim_total = key_dim_;  // num_k_heads * key_head_dim (per TP)

            if (seq_len == 1) {
                // Decode: single-step recurrent update using persistent state.
                invokeRecurrentGatedDeltaRule(out_ptr,
                                              conv_out.data<T>(),
                                              beta_ptr,
                                              g_ptr,
                                              recurrent_state_ptr,
                                              1,           // batch_size = 1 per request
                                              num_v_heads_,
                                              num_k_heads_,
                                              key_head_dim_,
                                              value_head_dim_,
                                              k_dim_total,
                                              stream);
            }
            else {
                // Prefill: process all timesteps in a SINGLE kernel launch.
                // (Previously: O(seq_len) kernel launches via host-side for-loop.)
                invokeGatedDeltaRulePrefill(out_ptr,
                                            conv_out.data<T>(),
                                            beta_ptr,
                                            g_ptr,
                                            recurrent_state_ptr,
                                            seq_len,
                                            num_v_heads_,
                                            num_k_heads_,
                                            key_head_dim_,
                                            value_head_dim_,
                                            k_dim_total,
                                            stream);
            }
            sync_check_cuda_error();

            // ----- 3c. RMSNormGated -----
            const int N = seq_len * num_v_heads_;
            invokeRMSNormGated(out_ptr, z_row_ptr, weights.norm.data<T>(), norm_eps_, N, value_head_dim_, all_col, num_v_heads_, stream);
            sync_check_cuda_error();

            token_offset += seq_len;
        }

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
