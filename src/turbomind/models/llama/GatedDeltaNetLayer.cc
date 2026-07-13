#include "src/turbomind/models/llama/GatedDeltaNetLayer.h"

#include <cstdio>
#include <cstdlib>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/engine/block.h"
#include "src/turbomind/models/llama/gated_delta_net_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

auto get_lc_state_size(const DeltaNetWeight& weights, int tp)
{
    int num_k_heads    = weights.num_k_heads / tp;
    int num_v_heads    = weights.num_v_heads / tp;
    int key_head_dim   = weights.key_head_dim;
    int value_head_dim = weights.value_head_dim;
    int d_conv         = weights.d_conv;
    int key_dim        = num_k_heads * key_head_dim;
    int value_dim      = num_v_heads * value_head_dim;
    int conv_dim       = key_dim * 2 + value_dim;
    return std::make_pair(num_v_heads * key_head_dim * value_head_dim, conv_dim * d_conv);
}

GatedDeltaNetLayer::GatedDeltaNetLayer(std::vector<DeltaNetWeight*> weights,
                                       CacheRegistry&               registry,
                                       const EngineParam&           engine,
                                       const Context&               context,
                                       int                          phases):
    tp_size_{engine.attn_tp_size * engine.attn_cp_size}, state_dtype_{engine.data_type}, linear_{*context.linear}
{
    TM_CHECK(!weights.empty());
    layer_num_ = static_cast<int>(weights.size());

    const auto [l_state_size, c_state_size] = get_lc_state_size(*weights[0], tp_size_);

    const int num_v_heads = weights[0]->num_v_heads / tp_size_;
    const int cell_elems =
        weights[0]->key_head_dim * weights[0]->value_head_dim;  // one (layer, head) state, in elements
    TM_CHECK_EQ(l_state_size, num_v_heads * cell_elems);        // sanity: get_lc_state_size agrees

    // Block unit (L_b layers x H_b v_heads). Unset env => one part per layer,
    // no head-grouping == today's behavior.
    int L_b = 1;
    int H_b = num_v_heads;
    if (const char* e = std::getenv("TM_GDN_BLOCK_CONFIG")) {
        TM_CHECK_EQ(std::sscanf(e, "%d,%d", &L_b, &H_b), 2) << "expected TM_GDN_BLOCK_CONFIG=l,h (e.g. 4,16)";
    }
    TM_CHECK_GT(L_b, 0);
    TM_CHECK_GT(H_b, 0);

    auto cdiv_i       = [](int a, int b) { return (a + b - 1) / b; };
    layers_per_block_ = L_b;
    heads_per_block_  = H_b;
    num_head_groups_  = cdiv_i(num_v_heads, H_b);  // == 1 when H_b >= num_v_heads
    num_layer_groups_ = cdiv_i(layer_num_, L_b);   // == layer_num_ when L_b == 1
    num_blocks_       = num_layer_groups_ * num_head_groups_;
    block_bytes_      = byte_size(state_dtype_, (size_t)L_b * H_b * cell_elems);

    // recurrent: num_blocks_ uniform parts, base part id == 1
    rec_base_ = registry.checkpoint().Register({{block_bytes_, 1, static_cast<size_t>(num_blocks_)}});

    // conv: accumulation -> part 0; ELEMENT offsets kept exactly as today.
    size_t off = 0;
    for (int i = 0; i < layer_num_; ++i) {
        weights[i]->conv_state_offset = off;
        off += c_state_size;
    }
    conv_total_bytes_ = byte_size(state_dtype_, off);
    registry.checkpoint().Register(conv_total_bytes_, /*alignment=*/1);  // reserves part 0

    // Visibility: slot-level interchange with the prefix object requires
    // block_bytes_ == prefix object bytes. Not enforced (optimal sizing is
    // out of scope); just log. Attention registers prefix before this ctor.
    const size_t prefix_bytes = registry.prefix().accumulation_bytes();
    // Logger is fmtlib-style ({} placeholders), matching slab.h's TM_LOG_WARN.
    TM_LOG_INFO("[GDN] block config L_b={} H_b={} -> num_layer_groups={} num_head_groups={} "
                "num_blocks={} block_bytes={} prefix_object_bytes={} ({})",
                L_b,
                H_b,
                num_layer_groups_,
                num_head_groups_,
                num_blocks_,
                block_bytes_,
                prefix_bytes,
                (prefix_bytes != 0 && block_bytes_ == prefix_bytes) ? "slab-shared" : "separate-slab-class");

    for (int L = 0; L < layer_num_; ++L) {
        // in-block row offset (elements) for this layer within its block-row;
        // == 0 when L_b == 1 (today's behavior).
        weights[L]->linear_state_offset = (L % L_b) * H_b * cell_elems;
        layer_index_[weights[L]]        = L;  // weight ptr -> GDN-local layer index
    }

    // Staging buffers: conv stays [batch]; recurrent becomes a [layer_group][batch][head_group] table.
    conv_state_ptrs_buf_      = {engine.max_batch_size, kCPUpinned};
    recurrent_state_ptrs_buf_ = {(ssize_t)num_layer_groups_ * engine.max_batch_size * num_head_groups_, kCPUpinned};

    for (int i = 0; i < phases; ++i) {
        data_.emplace_back();
        data_.at(i).conv_state_ptrs      = empty_like(conv_state_ptrs_buf_, kDEVICE);
        data_.at(i).recurrent_state_ptrs = empty_like(recurrent_state_ptrs_buf_, kDEVICE);
    }

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device);
    work_counter_ = {1, kDEVICE};

    TM_CUDA_CHECK(cudaStreamCreateWithPriority(&aux_stream_, cudaStreamNonBlocking, -1));
    TM_CUDA_CHECK(cudaEventCreateWithFlags(&ev_before_, cudaEventDisableTiming));
    TM_CUDA_CHECK(cudaEventCreateWithFlags(&ev_after_, cudaEventDisableTiming));
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
        Buffer_<Sequence*> rc = env.at("requests").buffer();
        for (int i = 0; i < rc.size(); ++i) {}
    }
    else if (op == BatchOp::kSetup) {
        Setup(phase, env);
    }
    else if (op == BatchOp::kPrepare) {
        auto& d     = data_.at(phase);
        d.q_offsets = env.at("q_offsets").buffer().borrow();
        d.k_offsets = env.at("k_offsets").buffer().borrow();
        d.finished  = env.at("finished").buffer().borrow();
        for (const auto& [ptr, bytes] : d.reset_ptrs) {
            Clear(Buffer_<uint8_t>{ptr, static_cast<ssize_t>(bytes), kDEVICE});
        }
        d.reset_ptrs.clear();
    }
}

void GatedDeltaNetLayer::Setup(int phase, TensorMap& env)
{
    auto& d = data_.at(phase);

    Buffer_<Sequence*> rc = env.at("requests").buffer();

    d.batch_size = rc.size();
    d.input_lens.resize(d.batch_size);
    d.reset_ptrs.clear();

    for (int i = 0; i < d.batch_size; ++i) {
        auto& s         = *rc[i];
        d.input_lens[i] = s.input_len;

        const CacheBlock& cb = *TM_CHECK_NOTNULL(s.frontier.get());
        TM_CHECK_NOTNULL(cb.allocation.a);

        conv_state_ptrs_buf_[i] = cb.base(0);  // conv accumulation part
        // One pointer per (layer-group, head-group) == per recurrent part; the L_b
        // layers of a block-row share this base (differ only by linear_state_offset).
        for (int lg = 0; lg < num_layer_groups_; ++lg) {
            for (int hg = 0; hg < num_head_groups_; ++hg) {
                const int part = rec_base_ + lg * num_head_groups_ + hg;
                recurrent_state_ptrs_buf_[(lg * d.batch_size + i) * num_head_groups_ + hg] = cb.base(part);
            }
        }

        // The forward for this batch starts at history_len + inflight_input_len.
        // Reset only when the true start position is 0; clear every part
        // (including any rounding padding -- harmless, never read by kernels).
        if (s.history_len + s.inflight_input_len == 0) {
            d.reset_ptrs.push_back({reinterpret_cast<uint8_t*>(cb.base(0)), conv_total_bytes_});
            for (int blk = 0; blk < num_blocks_; ++blk) {
                d.reset_ptrs.push_back({reinterpret_cast<uint8_t*>(cb.base(rec_base_ + blk)), block_bytes_});
            }
        }
    }

    Copy(conv_state_ptrs_buf_, d.batch_size, d.conv_state_ptrs);
    Copy(recurrent_state_ptrs_buf_,
         (ssize_t)num_layer_groups_ * d.batch_size * num_head_groups_,
         d.recurrent_state_ptrs);
}

void GatedDeltaNetLayer::Forward(ForwardParam p)
{
    TM_FUNCTION_SCOPE();

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
        Tensor    all_proj;
        TM_SCOPE_CALL(linear_.Forward(p.input, *weights.in_proj_all, all_proj));

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

        TM_CUDA_CHECK(cudaGetLastError());

        // =================================================================
        // 3. Process all requests at once via batched kernel launches
        // =================================================================
        Tensor attn_out{{token_num, value_dim}, dtype, device};
        Tensor conv_out{{token_num, conv_dim}, dtype, device};

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
                              pd.finished,
                              pd.batch_size,
                              weights.conv_state_offset,
                              sm_count_,
                              work_counter_.data(),
                              stream);
        TM_CUDA_CHECK(cudaGetLastError());

        // ----- 3b. Gated Delta Rule -----
        // Requests are sorted by input_len: decode (seq_len==1) first, prefill last.
        // Find the split point and dispatch each half to its optimal kernel.
        // When both are present, run them concurrently on separate streams.
        const int lg = layer_index_.at(p.weights) / layers_per_block_;  // layer-group (block row)
        auto      layer_rec =
            pd.recurrent_state_ptrs.slice(lg * pd.batch_size * num_head_groups_, pd.batch_size * num_head_groups_);
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
                TM_CUDA_CHECK(cudaEventRecord(ev_before_, stream));
                TM_CUDA_CHECK(cudaStreamWaitEvent(aux_stream_, ev_before_));

                // Decode on main stream
                auto dc_state = layer_rec.slice(0, decode_count * num_head_groups_);
                auto dc_q     = pd.q_offsets.slice(0, decode_count + 1);
                auto dc_done  = pd.finished.slice(0, decode_count);
                invokeGatedDeltaRuleBatched_v3(attn_out,
                                               conv_out,
                                               beta,
                                               g,
                                               dc_state,
                                               dc_q,
                                               dc_done,
                                               decode_count,
                                               num_k_heads,
                                               weights.linear_state_offset,
                                               state_dtype_,
                                               sm_count_,
                                               work_counter_.data(),
                                               stream,
                                               num_head_groups_,
                                               heads_per_block_);

                // Prefill on aux stream (higher priority)
                auto pf_state = layer_rec.slice(decode_count * num_head_groups_, prefill_count * num_head_groups_);
                auto pf_q     = pd.q_offsets.slice(decode_count, prefill_count + 1);
                auto pf_done  = pd.finished.slice(decode_count, prefill_count);
                invokeChunkedGatedDeltaRuleBatched(attn_out,
                                                   conv_out,
                                                   beta,
                                                   g,
                                                   pf_state,
                                                   pf_q,
                                                   pf_done,
                                                   prefill_count,
                                                   num_k_heads,
                                                   weights.linear_state_offset,
                                                   state_dtype_,
                                                   sm_count_,
                                                   work_counter_.data(),
                                                   aux_stream_,
                                                   num_head_groups_,
                                                   heads_per_block_);

                // Join: main stream waits for prefill to finish
                TM_CUDA_CHECK(cudaEventRecord(ev_after_, aux_stream_));
                TM_CUDA_CHECK(cudaStreamWaitEvent(stream, ev_after_));
            }
            else if (decode_count > 0) {
                auto state_slice = layer_rec.slice(0, decode_count * num_head_groups_);
                auto q_slice     = pd.q_offsets.slice(0, decode_count + 1);
                auto done_slice  = pd.finished.slice(0, decode_count);
                invokeGatedDeltaRuleBatched_v3(attn_out,
                                               conv_out,
                                               beta,
                                               g,
                                               state_slice,
                                               q_slice,
                                               done_slice,
                                               decode_count,
                                               num_k_heads,
                                               weights.linear_state_offset,
                                               state_dtype_,
                                               sm_count_,
                                               work_counter_.data(),
                                               stream,
                                               num_head_groups_,
                                               heads_per_block_);
            }
            else if (prefill_count > 0) {
                auto state_slice = layer_rec.slice(decode_count * num_head_groups_, prefill_count * num_head_groups_);
                auto q_slice     = pd.q_offsets.slice(decode_count, prefill_count + 1);
                auto done_slice  = pd.finished.slice(decode_count, prefill_count);
                invokeChunkedGatedDeltaRuleBatched(attn_out,
                                                   conv_out,
                                                   beta,
                                                   g,
                                                   state_slice,
                                                   q_slice,
                                                   done_slice,
                                                   prefill_count,
                                                   num_k_heads,
                                                   weights.linear_state_offset,
                                                   state_dtype_,
                                                   sm_count_,
                                                   work_counter_.data(),
                                                   stream,
                                                   num_head_groups_,
                                                   heads_per_block_);
                // invokeChunkedGatedDeltaRuleBatched
            }
        }
        TM_CUDA_CHECK(cudaGetLastError());

        // ----- 3c. RMSNormGated (all tokens at once) -----
        // Gate (z) lives at column conv_dim of all_proj with row-stride all_col.
        Tensor gate        = all_proj.slice({0, conv_dim}, {-1, value_dim});
        Tensor hidden_view = attn_out.view({token_num * num_v_heads, value_head_dim});
        invokeRMSNormGated(hidden_view, gate, weights.norm->weight, weights.norm->norm_eps_, stream);
        TM_CUDA_CHECK(cudaGetLastError());

        // =================================================================
        // 4. Output projection (all tokens at once)
        // =================================================================
        TM_SCOPE_CALL(linear_.Forward(attn_out, *weights.out_proj, p.output));
    };

    if (dtype == kHalf) {
        dispatch(half{});
    }
    else if (dtype == kBfloat16) {
        dispatch(nv_bfloat16{});
    }
    else {
        TM_LOG_FATAL("Unsupported dtype for GatedDeltaNetLayer");
    }
}

}  // namespace turbomind
