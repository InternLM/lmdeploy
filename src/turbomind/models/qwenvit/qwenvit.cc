// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/qwenvit/qwenvit.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/kernels/norm/layer_norm.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/layer_norm_weight.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/models/qwenvit/qwenvit_block_weight.h"
#include "src/turbomind/models/qwenvit/qwenvit_input.h"
#include "src/turbomind/models/qwenvit/qwenvit_kernels.h"
#include "src/turbomind/models/qwenvit/qwenvit_weight.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"

#include <algorithm>
#include <cmath>

namespace turbomind {

struct QwenVit::Impl {
    const QwenVitWeight&        weights_;
    const core::QwenVitConfig&  config_;
    LlamaLinear&                linear_;
    const comm::HostComm&       h_tp_group;
    comm::DeviceCommImpl* const d_comm_;
    const int                   tp_group_;
    const DataType              engine_data_type_;

    Buffer_<int> grid_thws_buf_;     // (t, h, w)
    Buffer_<int> grid_offsets_buf_;  // (token offset, natural offset)
    Buffer_<int> window_idx_buf_;
    Buffer_<int> cu_window_seqlens_buf_;
    Buffer_<int> attn_cu_seqlens_buf_;

    struct Data {
        Tensor                           batch_input;
        int                              batch_size;
        std::vector<std::array<int, 3>>  grid_thws_host;
        std::vector<std::pair<int, int>> image_embeds_coords;  // (size, pos) for image embeddings
        std::vector<std::pair<int, int>> input_embeds_coords;  // (size, pos) for input embeddings

        // for RoPE / pos-embed interpolation
        Tensor_<int> grid_thws;
        Tensor_<int> grid_offsets;
        Tensor_<int> mapped_idx;
        Tensor_<int> window_idx;
        Tensor_<int> window_mapped_idx;
        int          total_hw;
        int          merge_unit_count;

        // for full attention, one sequence per temporal frame
        Tensor_<int>  attn_cu_seqlens;
        Tensor_<int>  cu_window_seqlens;
        Tensor_<bool> attn_finished;
        Tensor_<bool> window_attn_finished;
        int           attn_batch_size;
        int           max_attn_len;
        int           window_attn_batch_size;
        int           max_window_attn_len;

        // mrope position-id scratch (per-phase pinned host buffers for fast H2D)
        Buffer_<int> mrope_segs_host;  // reinterpreted as MropeSegment[], kMropeSegInts ints per segment
        Buffer_<int> mrope_length_host;
        Buffer_<int> mrope_delta_host;
        Buffer_<int> mrope_offsets_host;

        // mrope outputs - owned here so UnifiedAttentionLayer can safely borrow() across env clears.
        Buffer_<int> mrope_segs_dev;          // device-side segment scratch, grown alongside host
        Tensor_<int> mrope_position_ids;      // (max_forward_token_num, 3), flat current-forward table
        Tensor_<int> mrope_length;            // (bsz,)
        Tensor_<int> mrope_position_delta;    // (bsz,)
        Tensor_<int> mrope_position_offsets;  // (bsz,), flat row offset for each request slot

        void Clear()
        {
            batch_size             = 0;
            total_hw               = 0;
            merge_unit_count       = 0;
            attn_batch_size        = 0;
            max_attn_len           = 0;
            window_attn_batch_size = 0;
            max_window_attn_len    = 0;
            grid_thws_host.clear();
            image_embeds_coords.clear();
            input_embeds_coords.clear();
        }
    };

    static constexpr int kMropeSegInts = sizeof(MropeSegment) / sizeof(int);
    static_assert(sizeof(MropeSegment) % sizeof(int) == 0);

    std::vector<Data> data_;

    Impl(const EngineParam& engine, const Context& ctx, const QwenVitWeight& weights, int phases):
        weights_{weights},
        config_{weights.config()},
        linear_{*ctx.linear},
        h_tp_group{ctx.comm.h_comm},
        d_comm_{ctx.comm.d_comm},
        tp_group_{ctx.comm.d_tp_group},
        engine_data_type_{engine.data_type}
    {
        for (int i = 0; i < phases; ++i) {
            auto& d              = data_.emplace_back();
            d.mrope_length_host  = {engine.max_batch_size, kCPUpinned};
            d.mrope_delta_host   = {engine.max_batch_size, kCPUpinned};
            d.mrope_offsets_host = {engine.max_batch_size, kCPUpinned};

            // mrope outputs at worst-case current-forward shape so Setup() never reallocates them.
            d.mrope_length           = Tensor_<int>{{engine.max_batch_size}, kDEVICE};
            d.mrope_position_delta   = Tensor_<int>{{engine.max_batch_size}, kDEVICE};
            d.mrope_position_offsets = Tensor_<int>{{engine.max_batch_size}, kDEVICE};
            d.mrope_position_ids     = Tensor_<int>{{engine.max_forward_token_num, 3}, kDEVICE};
        }
    }

    struct WindowShape {
        int vit_window{};
        int llm_h{};
        int llm_w{};
        int win_h{};
        int win_w{};
    };

    static void EnsureTensor(Tensor& tensor, Layout layout, DataType dtype, DeviceType device)
    {
        if (tensor.size() < layout.cosize()) {
            tensor = Tensor{std::move(layout), dtype, device};
        }
    }

    template<class T>
    static void EnsureTensor(Tensor_<T>& tensor, Layout layout, DeviceType device)
    {
        if (tensor.size() < layout.cosize()) {
            tensor = Tensor_<T>{std::move(layout), device};
        }
    }

    template<class T>
    static void EnsureBuffer(Buffer_<T>& buffer, ssize_t size, DeviceType device)
    {
        if (buffer.size() < size) {
            buffer = Buffer_<T>{size, device};
        }
    }

    WindowShape GetWindowShape(int h, int w) const
    {
        const int S          = config_.spatial_merge_size;
        const int vit_window = config_.window_size / S / config_.patch_size;
        TM_CHECK_GT(vit_window, 0);

        const int llm_h = h / S;
        const int llm_w = w / S;
        return WindowShape{
            vit_window, llm_h, llm_w, (llm_h + vit_window - 1) / vit_window, (llm_w + vit_window - 1) / vit_window};
    }

    void CollectPrefillInputs(Data& d, const BatchData& b, std::vector<Tensor>& pixel_values) const
    {
        const auto& cfg = config_;

        int input_ids_offsets    = 0;
        int image_embeds_offsets = 0;
        for (int i = 0; i < b.rc.size(); ++i) {
            const auto& c = *b.rc[i];
            const auto& s = *c.seq;

            if ((not c.autoregres) && (not s.multimodal_inputs.empty())) {
                Interval text{c.history_len + c.alpha, Interval::Size{c.input_len}};
                for (const auto& mm : s.multimodal_inputs) {
                    auto o = mm->interval & text;
                    if (auto size = (int)o.size()) {
                        pixel_values.push_back(mm->data);
                        d.batch_size += mm->data.shape(0);

                        const int text_offset  = input_ids_offsets + o.begin() - text.begin();
                        const int image_offset = image_embeds_offsets + o.begin() - mm->interval.begin();
                        d.input_embeds_coords.emplace_back(size, text_offset);
                        d.image_embeds_coords.emplace_back(size, image_offset);

                        const auto& grid_thw = mm->grid_thw;
                        d.grid_thws_host.emplace_back(grid_thw);
                        const auto& [t, h, w] = grid_thw;
                        const int prod        = t * h * w;
                        image_embeds_offsets += prod / cfg.spatial_merge_size / cfg.spatial_merge_size;
                    }
                }
            }

            input_ids_offsets += c.autoregres ? 1 : c.input_len;
        }
    }

    void ComputeSetupStats(Data& d) const
    {
        const auto& cfg        = config_;
        const int   S          = cfg.spatial_merge_size;
        const int   merge_unit = S * S;

        d.attn_batch_size        = 0;
        d.max_attn_len           = 0;
        d.merge_unit_count       = 0;
        d.window_attn_batch_size = 0;
        d.max_window_attn_len    = 0;

        for (const auto& [t, h, w] : d.grid_thws_host) {
            TM_CHECK(h % S == 0);
            TM_CHECK(w % S == 0);

            const int hw = h * w;
            d.attn_batch_size += t;
            d.max_attn_len = std::max(d.max_attn_len, hw);
            d.merge_unit_count += t * (h / S) * (w / S);

            if (cfg.use_window_attention) {
                const auto win = GetWindowShape(h, w);
                d.window_attn_batch_size += t * win.win_h * win.win_w;
                d.max_window_attn_len = std::max(d.max_window_attn_len, win.vit_window * win.vit_window * merge_unit);
            }
        }
    }

    void EnsureSetupStorage(Data& d)
    {
        const auto& cfg       = config_;
        const int   num_grids = (int)d.grid_thws_host.size();

        core::ContextGuard ctx{Allocator{kCPUpinned}};
        EnsureTensor(d.batch_input, {d.batch_size, cfg.patch_in_dim}, cfg.data_type, kCPUpinned);
        EnsureBuffer(grid_thws_buf_, (ssize_t)num_grids * 3, kCPUpinned);
        EnsureBuffer(grid_offsets_buf_, (ssize_t)num_grids * 2, kCPUpinned);
        EnsureBuffer(attn_cu_seqlens_buf_, (ssize_t)d.attn_batch_size + 1, kCPUpinned);

        EnsureTensor(d.grid_thws, {num_grids, 3}, kDEVICE);
        EnsureTensor(d.grid_offsets, {num_grids, 2}, kDEVICE);
        EnsureTensor(d.mapped_idx, {d.batch_size}, kDEVICE);
        EnsureTensor(d.attn_cu_seqlens, {d.attn_batch_size + 1}, kDEVICE);
        EnsureTensor(d.attn_finished, {d.attn_batch_size}, kDEVICE);

        if (cfg.use_window_attention) {
            EnsureBuffer(window_idx_buf_, (ssize_t)d.merge_unit_count, kCPUpinned);
            EnsureBuffer(cu_window_seqlens_buf_, (ssize_t)d.window_attn_batch_size + 1, kCPUpinned);
            EnsureTensor(d.window_idx, {d.merge_unit_count}, kDEVICE);
            EnsureTensor(d.window_mapped_idx, {d.batch_size}, kDEVICE);
            EnsureTensor(d.cu_window_seqlens, {d.window_attn_batch_size + 1}, kDEVICE);
            EnsureTensor(d.window_attn_finished, {d.window_attn_batch_size}, kDEVICE);
        }
    }

    void StagePixelValues(Data& d, const std::vector<Tensor>& pixel_values) const
    {
        if (d.batch_size == 0) {
            return;
        }

        ssize_t batch_offset = 0;
        for (const auto& pixel_value : pixel_values) {
            TM_CHECK_EQ(pixel_value.size(), pixel_value.shape(0) * config_.patch_in_dim);
            TM_CHECK_EQ(pixel_value.dtype(), d.batch_input.dtype());
            Copy(pixel_value, d.batch_input.slice(batch_offset, pixel_value.shape(0)));
            batch_offset += pixel_value.shape(0);
        }
        TM_CHECK_EQ(batch_offset, d.batch_size);
    }

    void BuildHostAttentionMeta(Data& d)
    {
        const auto& cfg        = config_;
        const int   S          = cfg.spatial_merge_size;
        const int   merge_unit = S * S;
        const int   num_grids  = (int)d.grid_thws_host.size();

        int token_offset                            = 0;
        int natural_offset                          = 0;
        int attn_seq_idx                            = 0;
        int attn_offset                             = 0;
        attn_cu_seqlens_buf_.data()[attn_seq_idx++] = 0;
        int window_group_pos                        = 0;
        int window_id_base                          = 0;
        int window_seq_idx                          = 0;
        int window_offset                           = 0;
        if (cfg.use_window_attention) {
            cu_window_seqlens_buf_.data()[window_seq_idx++] = 0;
        }

        for (int i = 0; i < num_grids; ++i) {
            const auto& [t, h, w] = d.grid_thws_host[i];

            const int hw = h * w;
            for (int tt = 0; tt < t; ++tt) {
                attn_offset += hw;
                attn_cu_seqlens_buf_.data()[attn_seq_idx++] = attn_offset;
            }

            grid_thws_buf_.data()[i * 3]        = t;
            grid_thws_buf_.data()[i * 3 + 1]    = h;
            grid_thws_buf_.data()[i * 3 + 2]    = w;
            grid_offsets_buf_.data()[i * 2]     = token_offset;
            grid_offsets_buf_.data()[i * 2 + 1] = natural_offset;

            TM_CHECK_LE(token_offset + t * h * w, d.batch_size);

            if (cfg.use_window_attention) {
                const auto win = GetWindowShape(h, w);
                for (int tt = 0; tt < t; ++tt) {
                    for (int wh = 0; wh < win.win_h; ++wh) {
                        for (int ww = 0; ww < win.win_w; ++ww) {
                            int valid_cells = 0;
                            for (int ih = 0; ih < win.vit_window; ++ih) {
                                const int llm_i = wh * win.vit_window + ih;
                                for (int iw = 0; iw < win.vit_window; ++iw) {
                                    const int llm_j = ww * win.vit_window + iw;
                                    if (llm_i >= win.llm_h || llm_j >= win.llm_w) {
                                        continue;
                                    }
                                    const int local_group = (tt * win.llm_h + llm_i) * win.llm_w + llm_j;
                                    const int orig_group  = window_id_base + local_group;
                                    window_idx_buf_.data()[window_group_pos++] = orig_group;
                                    ++valid_cells;
                                }
                            }
                            window_offset += valid_cells * merge_unit;
                            if (cu_window_seqlens_buf_.data()[window_seq_idx - 1] != window_offset) {
                                cu_window_seqlens_buf_.data()[window_seq_idx++] = window_offset;
                            }
                        }
                    }
                }
                window_id_base += t * win.llm_h * win.llm_w;
            }

            token_offset += t * h * w;
            natural_offset += h * w;
        }

        TM_CHECK_EQ(token_offset, d.batch_size);
        TM_CHECK_EQ(attn_offset, d.batch_size);
        TM_CHECK_EQ(attn_seq_idx, d.attn_batch_size + 1);
        if (cfg.use_window_attention) {
            TM_CHECK_EQ(window_group_pos, d.merge_unit_count);
            TM_CHECK_EQ(window_offset, d.batch_size);
            TM_CHECK_EQ(window_seq_idx, d.window_attn_batch_size + 1);
        }

        d.total_hw = natural_offset;
    }

    void PublishHostMetadataCopies(BatchCopy& copy, Data& d)
    {
        const int num_grids = (int)d.grid_thws_host.size();

        copy(grid_thws_buf_.data(), num_grids * 3, d.grid_thws.data());
        copy(grid_offsets_buf_.data(), num_grids * 2, d.grid_offsets.data());
        copy(attn_cu_seqlens_buf_.data(), d.attn_batch_size + 1, d.attn_cu_seqlens.data());
        Clear(d.attn_finished.slice(0, d.attn_batch_size));

        if (config_.use_window_attention) {
            copy(window_idx_buf_.data(), d.merge_unit_count, d.window_idx.data());
            copy(cu_window_seqlens_buf_.data(), d.window_attn_batch_size + 1, d.cu_window_seqlens.data());
            Clear(d.window_attn_finished.slice(0, d.window_attn_batch_size));
        }
    }

    void AllReduceSum(Tensor& tensor, cudaStream_t stream) const
    {
        if (d_comm_) {
            d_comm_->AllReduceSum(
                tensor.raw_data(), tensor.raw_data(), tensor.size(), tensor.dtype(), tp_group_, stream);
            TM_CUDA_CHECK(cudaGetLastError());
        }
    }

    void ApplyNorm(Tensor& out, const Tensor& input, const core::Module& norm, NormType norm_type) const
    {
        auto stream = core::Context::stream().handle();
        switch (norm_type) {
            case NormType::kLayerNorm: {
                const auto& ln = static_cast<const LayerNormWeight&>(norm);
                invokeLayerNorm(out, input, ln.weight, ln.bias, ln.norm_eps_, stream);
                break;
            }
            case NormType::kRMSNorm: {
                const auto& rms = static_cast<const NormWeight&>(norm);
                invokeRMSNorm(out, input, rms.weight, rms.norm_eps_, stream);
                break;
            }
            default:
                TM_LOG_FATAL("unsupported QwenVit norm type: {}", (int)norm_type);
        }
        TM_CUDA_CHECK(cudaGetLastError());
    }

    void ResidualBiasNorm(Tensor&             hidden_states,
                          Tensor&             residual,
                          const Tensor&       residual_bias,
                          const core::Module& norm,
                          NormType            norm_type) const
    {
        auto stream = core::Context::stream().handle();
        switch (norm_type) {
            case NormType::kLayerNorm: {
                const auto& ln = static_cast<const LayerNormWeight&>(norm);
                invokeResidualBiasLayerNorm(hidden_states.raw_data(),
                                            residual.raw_data(),
                                            ln.weight.raw_data(),
                                            ln.bias.data_or((void*)nullptr),
                                            residual_bias.data_or((void*)nullptr),
                                            hidden_states.dtype(),
                                            config_.hidden_dim,
                                            hidden_states.shape(0),
                                            ln.norm_eps_,
                                            stream);
                break;
            }
            case NormType::kRMSNorm: {
                const auto& rms = static_cast<const NormWeight&>(norm);
                invokeResidualBiasRMSNorm(hidden_states.raw_data(),
                                          residual.raw_data(),
                                          rms.weight.raw_data(),
                                          residual_bias.data_or((void*)nullptr),
                                          hidden_states.dtype(),
                                          config_.hidden_dim,
                                          hidden_states.shape(0),
                                          rms.norm_eps_,
                                          stream);
                break;
            }
            default:
                TM_LOG_FATAL("unsupported QwenVit norm type: {}", (int)norm_type);
        }
        TM_CUDA_CHECK(cudaGetLastError());
    }

    // Qwen3.5: precompute the bilinear-interpolation gather indices/weights for the
    // learned position-embedding table, then gather the 4 neighbour rows. Consumed by
    // `invokeFusedPosEmbedMerge` in Forward(). No-op for models without pos_embed.
    void FastPosEmbedInterpolate(Data& d, TensorMap& env)
    {
        auto& cfg    = weights_.config();
        auto  stream = core::Context::stream().handle();

        const int num_grid_per_side = (int)std::sqrt(cfg.num_position_embeddings);
        TM_CHECK_EQ(num_grid_per_side * num_grid_per_side, cfg.num_position_embeddings);
        TM_CHECK_EQ(weights_.pos_embed.shape(0), cfg.num_position_embeddings);
        TM_CHECK_EQ(weights_.pos_embed.shape(1), cfg.hidden_dim);

        Buffer_<int> pos_embed_idx     = {d.total_hw * 4, kDEVICE};
        Tensor       pos_embed_weights = {{d.total_hw, 4}, cfg.data_type, kDEVICE};
        invokeFastPosEmbedIdxWeight(pos_embed_idx.data(),
                                    pos_embed_weights.raw_data(),
                                    cfg.data_type,
                                    d.grid_thws.data(),
                                    d.grid_offsets.data(),
                                    (int)d.grid_thws_host.size(),
                                    d.total_hw,
                                    num_grid_per_side,
                                    stream);
        TM_CUDA_CHECK(cudaGetLastError());

        Tensor pos_embeds = {{d.total_hw * 4, cfg.hidden_dim}, cfg.data_type, kDEVICE};
        invokeEmbeddingLookup(pos_embeds, pos_embed_idx, weights_.pos_embed, stream);
        TM_CUDA_CHECK(cudaGetLastError());

        env.produce("pos_embeds", pos_embeds);
        env.produce("pos_embed_weights", pos_embed_weights);
    }

    void RotPosEmb(Data& d, TensorMap& env)
    {
        auto& cfg = weights_.config();

        const int head_dim = cfg.hidden_dim / cfg.head_num;
        // produce rotary_pos_emb: [total_hw, head_dim] with interleaved (c,s,c,s,...) pairs,
        // keyed by the same natural flat index that `mapped_idx` already carries. Vision q/k
        // are reordered into this adjacent-pair layout at export time.
        Tensor rotary_pos_emb = {{d.total_hw, head_dim}, cfg.data_type, kDEVICE};
        invokeQwenVitRotaryPosEmb(rotary_pos_emb.raw_data(),
                                  cfg.data_type,
                                  d.grid_thws.data(),
                                  d.grid_offsets.data(),
                                  (int)d.grid_thws_host.size(),
                                  d.total_hw,
                                  head_dim,
                                  /*theta=*/10000.0f,
                                  core::Context::stream().handle());
        TM_CUDA_CHECK(cudaGetLastError());
        env.produce("rotary_pos_emb", rotary_pos_emb);
    }

    Tensor PatchEmbedding(Data& d)
    {
        Tensor host_input = d.batch_input.slice(0, d.batch_size);
        Tensor input      = empty_like(host_input, kDEVICE);

        Copy(host_input, input);
        TM_CUDA_CHECK(cudaGetLastError());

        Tensor output;
        TM_SCOPE_CALL(linear_.Forward(input, *weights_.patch_embed, output));
        return output;
    }

    int Add(RequestCache& c)
    {
        const auto& [r, s] = std::tie(*c.req, *c.seq);
        if (r.mm_inputs) {
            if ((not r.session.start_flag) or (not r.session.end_flag)) {
                // only support non-interactive inference
                return Request::kInvalid;
            }

            const auto mm_inputs = std::dynamic_pointer_cast<multimodal::QwenVitInput>(r.mm_inputs);
            if (!mm_inputs) {
                return Request::kInvalid;
            }

            for (const auto& item : mm_inputs->items) {
                if (item.modality != multimodal::Modality::kImage && item.modality != multimodal::Modality::kVideo) {
                    return Request::kInvalid;
                }

                const int tokens = item.token_end - item.token_begin;
                if (tokens <= 0) {
                    return Request::kInvalid;
                }

                auto mm_item = std::make_shared<MultiModalData>(
                    MultiModalData{item.data, Interval{item.token_begin, Interval::Size{tokens}}, item.grid_thw});
                s.multimodal_inputs.push_back(mm_item);
            }
        }

        return Request::kOk;
    }

    void Add(int phase, TensorMap& env)
    {
        // convert model-specific multimodal inputs to internal MultiModalData
        const Buffer_<RequestCache*> rc = env.at("requests").buffer();
        for (int i = 0; i < rc.size(); ++i) {
            auto& c = *TM_CHECK_NOTNULL(rc[i]);
            if (c.status == 0) {
                c.status = Add(c);
            }
        }
    }

    // Build the mrope tensors consumed by `UnifiedAttentionLayer` and publish them to env.
    //
    // Per-forward layout: one flat row in `(max_forward_token_num, 3)` for each current token.
    // Prefill slots with multimodal_inputs get their active range written by
    // `invokeMropePositionIds` from a clipped list of MropeSegment descriptors. All other slots
    // (decode + text-only prefill) get `length[i] = 0` so FastRoPE falls through to the closed-
    // form `timestep + delta` path and never reads the stale rows.
    //
    // The output tensors live on `Data` (allocated worst-case in the ctor). env shares
    // ownership via shared_ptr; UAL borrows safely across env clears.
    void SetupMrope(int phase, TensorMap& env, BatchCopy& copy)
    {
        auto& d  = data_.at(phase);
        auto& b  = *env.at("batch").data<BatchData*>()[0];
        auto& rc = b.rc;

        const int bsz = (int)rc.size();
        if (bsz <= 0) {
            return;
        }

        const int S = weights_.config().spatial_merge_size;

        // 1) One pass to upper-bound segment count, build flat forward offsets, then size scratch.
        //    Worst case per prefill slot with mrope: 2*num_images + 1 segments.
        int upper_segs     = 0;
        int total_q_tokens = 0;
        for (int i = 0; i < bsz; ++i) {
            const auto& c                  = *rc[i];
            d.mrope_offsets_host.data()[i] = total_q_tokens;
            total_q_tokens += c.autoregres ? 1 : c.input_len;
            if (!c.autoregres && !c.seq->multimodal_inputs.empty()) {
                upper_segs += 2 * (int)c.seq->multimodal_inputs.size() + 1;
            }
        }
        TM_CHECK_LE(total_q_tokens, d.mrope_position_ids.shape(0));

        const ssize_t upper_ints = (ssize_t)upper_segs * kMropeSegInts;
        if (upper_ints > d.mrope_segs_host.size()) {
            core::ContextGuard ctx{Allocator{kCPUpinned}};
            d.mrope_segs_host = Buffer_<int>{upper_ints, kCPUpinned};
            d.mrope_segs_dev  = Buffer_<int>{upper_ints, kDEVICE};
        }

        // 2) Unified per-request walk - always advance mm_off; emit segments only for needs_table.
        auto* segs        = reinterpret_cast<MropeSegment*>(d.mrope_segs_host.data_or(nullptr));
        int   n_segs      = 0;
        int   max_seg_len = 0;

        for (int i = 0; i < bsz; ++i) {
            const auto& c            = *rc[i];
            const auto& s            = *c.seq;
            const bool  needs_table  = !c.autoregres && !s.multimodal_inputs.empty();
            const int   active_start = c.history_len + c.alpha;
            const int   active_end   = active_start + c.input_len;
            const int   q_offset     = d.mrope_offsets_host.data()[i];

            auto emit = [&](int run_start, int run_n, int run_base, int h2, int w2) {
                const int a = std::max(run_start, active_start);
                const int b = std::min(run_start + run_n, active_end);
                if (a >= b) {
                    return;
                }
                const int local_off = a - run_start;
                segs[n_segs++]      = MropeSegment{
                    q_offset + (a - active_start),
                    b - a,
                    /*base_pos=*/(h2 == 0) ? run_base + local_off : run_base,
                    h2,
                    w2,
                    /*k_offset=*/(h2 == 0) ? 0 : local_off,
                };
                max_seg_len = std::max(max_seg_len, b - a);
            };

            int row = 0, pos = 0, mm_off = 0;
            for (const auto& mm : s.multimodal_inputs) {
                const auto& [t, h, w] = mm->grid_thw;
                const int h2 = h / S, w2 = w / S, n_tok = t * h2 * w2;
                TM_CHECK_EQ(n_tok, (int)mm->interval.size()) << "image token count mismatches grid_thw";
                const int img_start = mm->interval.begin();
                const int img_base  = img_start + mm_off;
                if (needs_table) {
                    if (img_start > row) {
                        emit(row, img_start - row, pos, /*h2=*/0, /*w2=*/0);
                    }
                    emit(img_start, n_tok, img_base, h2, w2);
                }
                row               = img_start + n_tok;
                const int new_pos = std::max(t, std::max(h2, w2));
                pos               = img_base + new_pos;
                mm_off += new_pos - n_tok;
            }
            if (needs_table && row < active_end) {
                emit(row, active_end - row, pos, /*h2=*/0, /*w2=*/0);
            }

            d.mrope_length_host.data()[i] = needs_table ? c.input_len : 0;
            d.mrope_delta_host.data()[i]  = mm_off;
        }

        // 3) Copy the bsz prefix of length / delta / flat offsets into the pre-allocated tensors.
        //    Rows beyond bsz are untouched (UAL never reads them).
        copy(d.mrope_length_host, bsz, d.mrope_length.buffer());
        copy(d.mrope_delta_host, bsz, d.mrope_position_delta.buffer());
        copy(d.mrope_offsets_host, bsz, d.mrope_position_offsets.buffer());

        // 4) Populate position_ids only when a slot actually needs the table. Rows for slots
        //    with length[i] == 0 are unreachable from FastRoPE, so leaving them stale is safe.
        if (n_segs > 0) {
            const ssize_t segs_ints = (ssize_t)n_segs * kMropeSegInts;
            Copy(d.mrope_segs_host.slice(0, segs_ints), d.mrope_segs_dev.slice(0, segs_ints));
            invokeMropePositionIds(d.mrope_position_ids.data(),
                                   reinterpret_cast<const MropeSegment*>(d.mrope_segs_dev.data()),
                                   n_segs,
                                   max_seg_len,
                                   core::Context::stream().handle());
            TM_CUDA_CHECK(cudaGetLastError());
        }

        // 5) Publish all tensors - the consumer relies on this contract unconditionally.
        env.produce("mrope_length", d.mrope_length);
        env.produce("mrope_position_delta", d.mrope_position_delta);
        env.produce("mrope_position_offsets", d.mrope_position_offsets);
        env.produce("mrope_position_ids", d.mrope_position_ids);
    }

    void Setup(int phase, TensorMap& env)
    {
        auto& d    = data_.at(phase);
        auto& b    = *env.at("batch").data<BatchData*>()[0];
        auto& copy = *env.at("copy").data<BatchCopy*>()[0];

        d.Clear();
        std::vector<Tensor> pixel_values;
        CollectPrefillInputs(d, b, pixel_values);

        if (d.batch_size > 0) {
            ComputeSetupStats(d);
            EnsureSetupStorage(d);
            StagePixelValues(d, pixel_values);

            BuildHostAttentionMeta(d);
            PublishHostMetadataCopies(copy, d);
        }

        SetupMrope(phase, env, copy);
        h_tp_group->Sync();
    }

    void Prepare(int phase, TensorMap& env)
    {
        auto& d = data_.at(phase);
        if (d.batch_size == 0) {
            return;
        }

        auto stream = core::Context::stream().handle();
        invokeQwenVitBuildMappedIdx(d.mapped_idx.data(),
                                    d.grid_thws.data(),
                                    d.grid_offsets.data(),
                                    (int)d.grid_thws_host.size(),
                                    config_.spatial_merge_size,
                                    stream);
        if (config_.use_window_attention) {
            invokeQwenVitBuildWindowMappedIdx(d.window_mapped_idx.data(),
                                              d.mapped_idx.data(),
                                              d.window_idx.data(),
                                              config_.spatial_merge_size * config_.spatial_merge_size,
                                              d.merge_unit_count,
                                              stream);
        }

        // Qwen3.5 learned positional embedding (bilinear interpolation of a fixed grid).
        if (config_.num_position_embeddings > 0) {
            FastPosEmbedInterpolate(d, env);
        }

        RotPosEmb(d, env);
    }

    void Forward(int phase, TensorMap& args)
    {
        auto& d = data_.at(phase);
        if (d.batch_size == 0) {
            return;
        }

        auto& cfg = weights_.config();

        auto residual       = PatchEmbedding(d);
        auto rotary_pos_emb = args.consume("rotary_pos_emb");
        auto stream         = core::Context::stream().handle();

        // Qwen3.5: fused pos-embed gather/merge into the patch_embed output (with bias folded in):
        //   residual[pos, d] += Σ_k w[k] * pos_embeds[mapped*4+k, d] + bias[d]
        if (cfg.num_position_embeddings > 0) {
            auto pos_embeds        = args.consume("pos_embeds");
            auto pos_embed_weights = args.consume("pos_embed_weights");
            invokeFusedPosEmbedMerge(residual.raw_data(),
                                     pos_embeds.raw_data(),
                                     pos_embed_weights.raw_data(),
                                     d.mapped_idx.data(),
                                     weights_.patch_embed->bias ? weights_.patch_embed->bias.raw_data() : nullptr,
                                     d.batch_size,
                                     cfg.hidden_dim,
                                     cfg.data_type,
                                     stream);
            TM_CUDA_CHECK(cudaGetLastError());
        }

        if (cfg.use_window_attention) {
            Tensor reordered{{d.batch_size, cfg.hidden_dim}, cfg.data_type, kDEVICE};
            invokeQwenVitWindowReorder(reordered,
                                       residual,
                                       d.window_idx.data(),
                                       cfg.spatial_merge_size * cfg.spatial_merge_size,
                                       d.merge_unit_count,
                                       stream);
            residual = std::move(reordered);
        }

        Buffer symm_buf = args.contains("symm_buf") ? args.at("symm_buf").buffer() : Buffer{};

        Tensor hidden_states = [&]() {
            if (symm_buf) {
                return Tensor{symm_buf.view(cfg.data_type), {d.batch_size, cfg.hidden_dim}};
            }
            else {
                return Tensor{{d.batch_size, cfg.hidden_dim}, cfg.data_type, kDEVICE};
            }
        }();

        ApplyNorm(hidden_states, residual, *weights_.block(0)->norm1, cfg.norm_type);

        for (int layer_id = 0; layer_id < cfg.depth; ++layer_id) {
            auto* block = weights_.block(layer_id);

            // attn
            auto invoke = [&](auto t) {
                using T = decltype(t);
                Attn<T>(hidden_states, hidden_states, d, layer_id, rotary_pos_emb);
            };
            TM_DISPATCH_PRIMARY_DTYPES(hidden_states.dtype(), invoke);

            if (block->attention->tp_size > 1) {
                AllReduceSum(hidden_states, stream);
            }

            ResidualBiasNorm(hidden_states, residual, block->attention->wo->bias, *block->norm2, cfg.norm_type);

            // mlp
            Mlp(hidden_states, hidden_states, d, layer_id);
            AllReduceSum(hidden_states, stream);

            const auto* next_norm =
                layer_id + 1 < cfg.depth ? weights_.block(layer_id + 1)->norm1.get() : weights_.merger_norm.get();
            TM_CHECK_NOTNULL(next_norm);
            ResidualBiasNorm(hidden_states, residual, block->mlp_fc2->bias, *next_norm, cfg.norm_type);
        }

        Tensor image_embeds = Merger(hidden_states, symm_buf);
        if (cfg.use_window_attention) {
            Tensor reordered{{d.merge_unit_count, cfg.out_hidden_dim}, image_embeds.dtype(), kDEVICE};
            invokeQwenVitReverseWindow(reordered, image_embeds, d.window_idx.data(), d.merge_unit_count, stream);
            image_embeds = std::move(reordered);
        }

        // ViT may run in its own dtype (e.g. bf16) while the text engine runs
        // in fp16 (AWQ-forced). PatchMultimodalEmbedding merges this buffer
        // into the text embedding stream via a byte-level copy, so the dtypes
        // must match before publishing.
        EnsureFloatDtype(image_embeds, engine_data_type_);

        args.produce("multimodal",
                     MultiModalEmbeddingData{image_embeds, d.image_embeds_coords, d.input_embeds_coords}.buf());
    }

    template<typename T>
    AttentionParams<T> CreateVitAttentionParams(
        Tensor& attn_context, Tensor& qkv, Tensor& kv, Data& d, const AttentionWeight& attn, int layer_id)
    {
        const bool use_full_attn  = IsFullAttentionLayer(layer_id);
        const int  local_head_num = attn.head_num / attn.tp_size;
        const int  head_dim       = attn.head_dim;
        const int  token_num      = d.batch_size;

        AttentionParams<T> params{};
        params.out = (T*)attn_context.raw_data();
        params.q   = (T*)qkv.raw_data();

        params.stride = (int64_t)local_head_num * 3 * head_dim;

        params.cu_q_len = use_full_attn ? d.attn_cu_seqlens.data() : d.cu_window_seqlens.data();
        params.cu_k_len = params.cu_q_len;
        params.finished = use_full_attn ? d.attn_finished.data() : d.window_attn_finished.data();

        params.linear_iter_params = LinearIteratorParams{
            kv.raw_data(),
            2 * token_num * head_dim,
            token_num * head_dim,
        };

        params.token_num  = token_num;
        params.batch_size = use_full_attn ? d.attn_batch_size : d.window_attn_batch_size;
        params.max_q_len  = use_full_attn ? d.max_attn_len : d.max_window_attn_len;
        params.max_k_len  = params.max_q_len;

        params.num_heads     = local_head_num;
        params.num_kv_heads  = local_head_num;
        params.size_per_head = head_dim;
        params.causal        = false;
        params.layer_id      = layer_id;

        double scaling = 1.;
        if (attn.softmax_scale) {
            scaling *= attn.softmax_scale;
        }
        else {
            scaling /= std::sqrt((float)head_dim);
        }
        params.inv_sqrt_dh = scaling * std::log2(std::exp(1.));

        params.window_size     = 0;
        params.rope_param.type = RopeType::kNull;
        params.max_split_k     = 1;
        params.cp_size         = 1;
        params.stream          = core::Context::stream().handle();

        return params;
    }

    bool IsFullAttentionLayer(int layer_id) const
    {
        if (!config_.use_window_attention) {
            return true;
        }
        return std::find(config_.fullatt_block_indexes.begin(), config_.fullatt_block_indexes.end(), layer_id)
               != config_.fullatt_block_indexes.end();
    }

    template<typename T>
    void Attn(Tensor& input, Tensor& output, Data& d, int layer_id, const Tensor& rotary_pos_emb)
    {
        auto& vit_cfg = weights_.config();
        auto* attn    = weights_.block(layer_id)->attention.get();

        Tensor qkv;
        TM_SCOPE_CALL(linear_.Forward(input, *attn->w_qkv, qkv));
        TM_CUDA_CHECK(cudaGetLastError());

        const int local_head_num = attn->head_num / attn->tp_size;
        const int head_dim       = attn->head_dim;                         // may be padded
        const int rope_head_dim  = vit_cfg.hidden_dim / vit_cfg.head_num;  // model's real per-head dim
        const int token_num      = d.batch_size;

        const int* mapped_idx = (config_.use_window_attention ? d.window_mapped_idx.data() : d.mapped_idx.data());

        Tensor tmp_kv{{local_head_num, 2, d.batch_size, head_dim}, qkv.dtype(), qkv.device()};
        invokeQwenVitPrepareQKV(qkv.raw_data(),
                                tmp_kv.raw_data(),
                                attn->w_qkv->bias.raw_data(),
                                rotary_pos_emb.raw_data(),
                                mapped_idx,
                                qkv.dtype(),
                                token_num,
                                local_head_num,
                                head_dim,
                                rope_head_dim,
                                core::Context::stream().handle());
        TM_CUDA_CHECK(cudaGetLastError());

        Tensor attn_output{{token_num, local_head_num * head_dim}, qkv.dtype(), qkv.device()};
        auto   params = CreateVitAttentionParams<T>(attn_output, qkv, tmp_kv, d, *attn, layer_id);
        dispatchAttention<T>(params);
        TM_CUDA_CHECK(cudaGetLastError());

        TM_SCOPE_CALL(linear_.Forward(attn_output, *attn->wo, output));
        TM_CUDA_CHECK(cudaGetLastError());
    }

    void Mlp(Tensor& input, Tensor& output, Data& d, int layer_id)
    {
        auto* block  = weights_.block(layer_id);
        auto  stream = core::Context::stream().handle();

        if (config_.gated_mlp) {
            TM_CHECK(block->mlp_gate);
            Tensor gate;
            Tensor up;
            TM_SCOPE_CALL(linear_.Forward(input, *block->mlp_gate, gate));
            TM_SCOPE_CALL(linear_.Forward(input, *block->mlp_fc1, up));
            TM_CUDA_CHECK(cudaGetLastError());

            ApplyBias(gate, block->mlp_gate->bias, stream);
            ApplyBias(up, block->mlp_fc1->bias, stream);
            Activation(gate, up, ActivationType::kSilu, stream);
            TM_CUDA_CHECK(cudaGetLastError());

            TM_SCOPE_CALL(linear_.Forward(gate, *block->mlp_fc2, output));
        }
        else {
            Tensor inter;
            TM_SCOPE_CALL(linear_.Forward(input, *block->mlp_fc1, inter));
            TM_CUDA_CHECK(cudaGetLastError());

            // Qwen2-VL/2.5 use the erf GELU; Qwen3.5 uses the tanh approximation.
            const ActivationType act = config_.gelu_tanh ? ActivationType::kGeluPytorchTanh : ActivationType::kGelu;
            invokeAddBiasActivation(inter, block->mlp_fc1->bias, act, stream);
            TM_CUDA_CHECK(cudaGetLastError());

            TM_SCOPE_CALL(linear_.Forward(inter, *block->mlp_fc2, output));
        }
        TM_CUDA_CHECK(cudaGetLastError());
    }

    Tensor Merger(Tensor& input, Buffer symm_buf)
    {
        auto& cfg    = config_;
        auto  stream = core::Context::stream().handle();

        const int merge_area   = cfg.spatial_merge_size * cfg.spatial_merge_size;
        Tensor    merged_input = input.view({-1, cfg.hidden_dim * merge_area});

        Tensor inter;
        TM_SCOPE_CALL(linear_.Forward(merged_input, *weights_.merger_fc1, inter));
        TM_CUDA_CHECK(cudaGetLastError());

        invokeAddBiasActivation(inter, weights_.merger_fc1->bias, ActivationType::kGelu, stream);
        TM_CUDA_CHECK(cudaGetLastError());

        Tensor output;
        if (d_comm_) {
            output = {symm_buf.view(config_.data_type), {inter.shape(0), weights_.merger_fc2->output_dim}};
        }
        TM_SCOPE_CALL(linear_.Forward(inter, *weights_.merger_fc2, output));
        TM_CUDA_CHECK(cudaGetLastError());
        AllReduceSum(output, stream);
        ApplyBias(output, weights_.merger_fc2->bias, stream);
        TM_CUDA_CHECK(cudaGetLastError());
        if (d_comm_) {
            Tensor tmp = empty_like(output);
            Copy(output, tmp);
            output = tmp;
        }

        return output;
    }
};

QwenVit::QwenVit(const EngineParam& engine, const Context& ctx, const QwenVitWeight& weights, int phases):
    impl_{std::make_unique<Impl>(engine, ctx, weights, phases)}
{
}

QwenVit::~QwenVit() = default;

void QwenVit::Run(BatchOp op, int phase, TensorMap& env)
{
    TM_FUNCTION_SCOPE();
    switch (op) {
        case BatchOp::kAdd:
            return impl_->Add(phase, env);
        case BatchOp::kSetup:
            return impl_->Setup(phase, env);
        case BatchOp::kPrepare:
            return impl_->Prepare(phase, env);
        case BatchOp::kForward:
            return impl_->Forward(phase, env);
        default:
            return;
    }
}

}  // namespace turbomind
