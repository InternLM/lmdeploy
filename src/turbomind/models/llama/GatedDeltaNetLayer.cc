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
    // Store layer types for index mapping and count linear layers
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

    // Initialize per-phase data
    for (int i = 0; i < phases; ++i) {
        phase_data_.push_back(std::make_shared<PhaseData>());
    }
}

GatedDeltaNetLayer::~GatedDeltaNetLayer() = default;

void GatedDeltaNetLayer::Run(BatchOp op, int phase, TensorMap& env)
{
    if (op == BatchOp::kAdd) {
        // Allocate persistent states for newly added requests
        Buffer_<RequestCache*> rc    = env.at("requests").buffer();
        const auto             dtype = dtype_;
        for (int i = 0; i < rc.size(); ++i) {
            auto& c = *rc[i];
            // Only allocate if this model has linear attention layers
            if (num_linear_layers_ > 0) {
                // conv_states: (num_linear_layers, conv_dim, d_conv) — zero-initialized
                c.conv_states = Tensor{{num_linear_layers_, conv_dim_, d_conv_}, dtype, kDEVICE};
                Clear(c.conv_states);
                // recurrent_states: (num_linear_layers, num_v_heads, key_head_dim, value_head_dim)
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
        d.input_lens[i] = batch.rc[i]->input_len;  // Snapshot before engine re-enters Schedule()
    }
}

// Helper: compute the linear attention layer index for a given absolute layer_id.
// Returns the ordinal index (0, 1, 2, ...) among linear attention layers.
// This must match the indexing used when allocating states.
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
        // 1. Linear projections on all tokens
        // =================================================================
        Tensor mixed_qkv = linear_.Forward(p.input, weights.in_proj_qkv);
        sync_check_cuda_error();

        Tensor z = linear_.Forward(p.input, weights.in_proj_z);
        sync_check_cuda_error();

        Tensor b_raw = linear_.Forward(p.input, weights.in_proj_b);
        sync_check_cuda_error();

        Tensor a_raw = linear_.Forward(p.input, weights.in_proj_a);
        sync_check_cuda_error();

        // =================================================================
        // 2. Compute beta and g for all tokens
        // =================================================================
        const int bg_total = token_num * num_v_heads_;
        Tensor    beta{{token_num, num_v_heads_}, dtype, device};
        Tensor    g_tensor{{token_num, num_v_heads_}, dtype, device};

        invokeComputeBetaG(beta.data<T>(),
                           g_tensor.data<T>(),
                           b_raw.data<T>(),
                           a_raw.data<T>(),
                           weights.A_log.data<T>(),
                           weights.dt_bias.data<T>(),
                           bg_total,
                           num_v_heads_,
                           stream);
        sync_check_cuda_error();

        // =================================================================
        // 3. Process each request independently (multi-batch support)
        // =================================================================
        // Each request has its own conv_states and recurrent_states,
        // and may have different input lengths (decode=1, prefill>1).
        // We process requests sequentially — within each request, the
        // kernels are parallel on the GPU.

        Tensor attn_out{{token_num, value_dim_}, dtype, device};

        int token_offset = 0;
        for (int req = 0; req < pd.batch_size; ++req) {
            auto&     rc = *pd.rc[req];
            const int seq_len =
                pd.input_lens[req];  // Use snapshot taken at Setup time (avoids race with engine Schedule)
            if (seq_len == 0)
                continue;

            // Slice this request's portion of the projected tensors
            T* qkv_ptr  = mixed_qkv.data<T>() + token_offset * conv_dim_;
            T* z_ptr    = z.data<T>() + token_offset * value_dim_;
            T* beta_ptr = beta.data<T>() + token_offset * num_v_heads_;
            T* g_ptr    = g_tensor.data<T>() + token_offset * num_v_heads_;
            T* out_ptr  = attn_out.data<T>() + token_offset * value_dim_;

            // Map absolute layer_id to linear-layer index (0-based among linear layers)
            const int state_layer_idx = linear_layer_index(p.layer_id, layer_types_);

            // Per-request, per-layer state pointers
            T* conv_state_ptr      = nullptr;
            T* recurrent_state_ptr = nullptr;
            if (rc.conv_states) {
                const int conv_state_size = conv_dim_ * d_conv_;
                conv_state_ptr            = rc.conv_states.data<T>() + state_layer_idx * conv_state_size;
            }
            if (rc.recurrent_states) {
                const int rec_state_size = num_v_heads_ * key_head_dim_ * value_head_dim_;
                recurrent_state_ptr      = rc.recurrent_states.data<T>() + state_layer_idx * rec_state_size;
            }

            // ----- 3a. Fused Causal Conv1d + SiLU (row-major) -----
            Tensor conv_out{{seq_len, conv_dim_}, dtype, device};
            invokeFusedConv1dSiLU(conv_out.data<T>(),
                                  qkv_ptr,
                                  weights.conv1d.data<T>(),
                                  (const T*)nullptr,  // no bias in HF Qwen3.5 conv1d
                                  conv_state_ptr,
                                  1,  // batch_size=1 per request
                                  conv_dim_,
                                  seq_len,
                                  d_conv_,
                                  stream);
            sync_check_cuda_error();

            // ----- 3b. Split Q/K/V from conv output -----
            T* conv_data = conv_out.data<T>();

            // ----- 3c. Gather Q, K, V into contiguous buffers from strided conv_out -----
            // conv_out is (seq_len, conv_dim_) where conv_dim_ = key_dim_*2 + value_dim_
            // Q = [:, 0:key_dim_], K = [:, key_dim_:2*key_dim_], V = [:, 2*key_dim_:]
            Tensor q_contig{{seq_len, key_dim_}, dtype, device};
            Tensor k_contig{{seq_len, key_dim_}, dtype, device};
            Tensor v_contig{{seq_len, value_dim_}, dtype, device};

            check_cuda_error(cudaMemcpy2DAsync(q_contig.data<T>(),
                                               key_dim_ * sizeof(T),
                                               conv_data,
                                               conv_dim_ * sizeof(T),
                                               key_dim_ * sizeof(T),
                                               seq_len,
                                               cudaMemcpyDeviceToDevice,
                                               stream));
            check_cuda_error(cudaMemcpy2DAsync(k_contig.data<T>(),
                                               key_dim_ * sizeof(T),
                                               conv_data + key_dim_,
                                               conv_dim_ * sizeof(T),
                                               key_dim_ * sizeof(T),
                                               seq_len,
                                               cudaMemcpyDeviceToDevice,
                                               stream));
            check_cuda_error(cudaMemcpy2DAsync(v_contig.data<T>(),
                                               value_dim_ * sizeof(T),
                                               conv_data + 2 * key_dim_,
                                               conv_dim_ * sizeof(T),
                                               value_dim_ * sizeof(T),
                                               seq_len,
                                               cudaMemcpyDeviceToDevice,
                                               stream));

            // ----- 3d. L2-normalize Q and K (now contiguous) -----
            invokeL2Norm(q_contig.data<T>(), seq_len * num_k_heads_, key_head_dim_, stream);
            invokeL2Norm(k_contig.data<T>(), seq_len * num_k_heads_, key_head_dim_, stream);
            sync_check_cuda_error();

            T* q_ptr_final = q_contig.data<T>();
            T* k_ptr_final = k_contig.data<T>();
            T* v_ptr_final = v_contig.data<T>();

            // ----- 3e. Repeat-interleave Q and K if needed -----
            Tensor q_expanded, k_expanded;
            if (num_k_heads_ != num_v_heads_) {
                q_expanded         = Tensor{{seq_len, num_v_heads_ * key_head_dim_}, dtype, device};
                k_expanded         = Tensor{{seq_len, num_v_heads_ * key_head_dim_}, dtype, device};
                const int total_qk = seq_len * num_v_heads_ * key_head_dim_;

                invokeRepeatInterleave(q_expanded.data<T>(),
                                       q_contig.data<T>(),
                                       total_qk,
                                       num_k_heads_,
                                       num_v_heads_,
                                       key_head_dim_,
                                       stream);
                invokeRepeatInterleave(k_expanded.data<T>(),
                                       k_contig.data<T>(),
                                       total_qk,
                                       num_k_heads_,
                                       num_v_heads_,
                                       key_head_dim_,
                                       stream);
                sync_check_cuda_error();
                q_ptr_final = q_expanded.data<T>();
                k_ptr_final = k_expanded.data<T>();
            }

            // ----- 3f. Gated Delta Rule -----
            if (seq_len == 1) {
                // Decode: single-step recurrent update using persistent state
                invokeRecurrentGatedDeltaRule(out_ptr,
                                              q_ptr_final,
                                              k_ptr_final,
                                              v_ptr_final,
                                              beta_ptr,
                                              g_ptr,
                                              recurrent_state_ptr,
                                              1,
                                              num_v_heads_,
                                              key_head_dim_,
                                              value_head_dim_,
                                              stream);
            }
            else {
                // Prefill: process all timesteps serially, updating state
                invokeSerialGatedDeltaRule(out_ptr,
                                           q_ptr_final,
                                           k_ptr_final,
                                           v_ptr_final,
                                           beta_ptr,
                                           g_ptr,
                                           recurrent_state_ptr,
                                           1,
                                           seq_len,
                                           num_v_heads_,
                                           key_head_dim_,
                                           value_head_dim_,
                                           stream);
            }
            sync_check_cuda_error();

            // ----- 3g. RMSNormGated -----
            const int N = seq_len * num_v_heads_;
            invokeRMSNormGated(out_ptr, z_ptr, weights.norm.data<T>(), norm_eps_, N, value_head_dim_, stream);
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
