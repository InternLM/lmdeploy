// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/internvit/internvit.h"

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/norm/layer_norm.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/internvit/internvit_block_weight.h"
#include "src/turbomind/models/internvit/internvit_input.h"
#include "src/turbomind/models/internvit/internvit_kernels.h"
#include "src/turbomind/models/internvit/internvit_weight.h"
#include "src/turbomind/models/layer_norm_weight.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"

#include <array>
#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

namespace turbomind {

struct InternVit::Impl {
    const InternVitWeight&       weights_;
    const core::InternVitConfig& config_;
    const comm::HostComm&        h_tp_group;
    LlamaLinear&                 linear_;
    comm::DeviceCommImpl* const  d_comm_;
    const int                    tp_group_;
    const int                    tp_size_;
    const DataType               engine_data_type_;

    Buffer_<int> attn_cu_seqlens_buf_;

    struct Data {
        Tensor                           batch_input;
        int                              batch_size{};
        std::vector<std::pair<int, int>> image_embeds_coords;
        std::vector<std::pair<int, int>> input_embeds_coords;
        Tensor_<int>                     attn_cu_seqlens;
        Tensor_<bool>                    attn_finished;
        int                              token_num{};
        int                              seq_len{};

        void Clear()
        {
            batch_size = 0;
            token_num  = 0;
            seq_len    = 0;
            image_embeds_coords.clear();
            input_embeds_coords.clear();
        }
    };

    std::vector<Data> data_;

    Impl(const EngineParam& engine, const Context& ctx, const InternVitWeight& weights, int phases):
        weights_{weights},
        config_{weights.config()},
        h_tp_group{ctx.comm.h_comm},
        linear_{*ctx.linear},
        d_comm_{ctx.comm.d_comm},
        tp_group_{ctx.comm.d_tp_group},
        tp_size_{ctx.comm.h_tp_group ? ctx.comm.h_tp_group->n_ranks() : 1},
        engine_data_type_{engine.data_type}
    {
        const auto& cfg = weights.config();
        for (int i = 0; i < phases; ++i) {
            auto& d           = data_.emplace_back();
            d.batch_input     = {{engine.max_forward_token_num, cfg.in_channels, cfg.image_height, cfg.image_width},
                             cfg.data_type,
                             kCPUpinned};
            d.attn_cu_seqlens = Tensor_<int>{{engine.max_forward_token_num + 1}, kDEVICE};
            d.attn_finished   = Tensor_<bool>{{engine.max_forward_token_num}, kDEVICE};
        }
        attn_cu_seqlens_buf_ = {engine.max_forward_token_num + 1, kCPUpinned};
    }

    void AllReduceSum(Tensor& tensor, cudaStream_t stream) const
    {
        if (d_comm_ && tp_size_ > 1) {
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
                TM_LOG_FATAL("unsupported InternVit norm type: {}", (int)norm_type);
        }
        TM_CUDA_CHECK(cudaGetLastError());
    }

    void ResidualScaleNorm(Tensor&             hidden_states,
                           Tensor&             residual,
                           const Tensor&       branch_output,
                           const Tensor&       branch_scale,
                           const Tensor&       branch_bias,
                           const core::Module* norm,
                           NormType            norm_type) const
    {
        auto stream = core::Context::stream().handle();
        switch (norm_type) {
            case NormType::kLayerNorm: {
                const auto& ln = static_cast<const LayerNormWeight&>(*norm);
                invokeInternVitResidualScaleNorm(hidden_states,
                                                 residual,
                                                 branch_output,
                                                 branch_scale,
                                                 branch_bias,
                                                 ln.weight,
                                                 ln.bias,
                                                 ln.norm_eps_,
                                                 norm_type,
                                                 stream);
                break;
            }
            case NormType::kRMSNorm: {
                const auto& rms = static_cast<const NormWeight&>(*norm);
                invokeInternVitResidualScaleNorm(hidden_states,
                                                 residual,
                                                 branch_output,
                                                 branch_scale,
                                                 branch_bias,
                                                 rms.weight,
                                                 {},
                                                 rms.norm_eps_,
                                                 norm_type,
                                                 stream);
                break;
            }
            case NormType::kNone: {
                invokeInternVitResidualScaleNorm(
                    hidden_states, residual, branch_output, branch_scale, branch_bias, {}, {}, 0.f, norm_type, stream);
                break;
            }
            default:
                TM_LOG_FATAL("unsupported InternVit norm type: {}", (int)norm_type);
        }
        TM_CUDA_CHECK(cudaGetLastError());
    }

    int Add(RequestCache& c)
    {
        const auto& [r, s] = std::tie(*c.req, *c.seq);
        if (!r.mm_inputs) {
            return Request::kOk;
        }

        if ((not r.session.start_flag) or (not r.session.end_flag)) {
            return Request::kInvalid;
        }

        const auto mm_inputs = std::dynamic_pointer_cast<multimodal::InternVitInput>(r.mm_inputs);
        if (!mm_inputs) {
            return Request::kInvalid;
        }

        for (const auto& item : mm_inputs->items) {
            if (item.modality != multimodal::Modality::kImage) {
                return Request::kInvalid;
            }

            const int tokens = item.token_end - item.token_begin;
            if (tokens <= 0) {
                return Request::kInvalid;
            }

            auto mm_item = std::make_shared<MultiModalData>(
                MultiModalData{item.data, Interval{item.token_begin, Interval::Size{tokens}}, std::array<int, 3>{}});
            s.multimodal_inputs.push_back(mm_item);
        }

        return Request::kOk;
    }

    void Add(int /*phase*/, TensorMap& env)
    {
        const Buffer_<RequestCache*> rc = env.at("requests").buffer();
        for (int i = 0; i < rc.size(); ++i) {
            auto& c = *TM_CHECK_NOTNULL(rc[i]);
            if (c.status == 0) {
                c.status = Add(c);
            }
        }
    }

    void Setup(int phase, TensorMap& env)
    {
        auto&       d    = data_.at(phase);
        auto&       b    = *env.at("batch").data<BatchData*>()[0];
        auto&       copy = *env.at("copy").data<BatchCopy*>()[0];
        const auto& cfg  = config_;

        int input_ids_offsets    = 0;
        int image_embeds_offsets = 0;
        d.Clear();
        std::vector<Tensor> pixel_values;

        const auto& rc = b.rc;
        for (int i = 0; i < rc.size(); ++i) {
            const auto& c = *rc[i];
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

                        image_embeds_offsets += (int)mm->interval.size();
                    }
                }
            }

            input_ids_offsets += c.autoregres ? 1 : c.input_len;
        }

        if (d.batch_size > 0) {
            // batch input
            if (d.batch_size > d.batch_input.shape(0)) {
                core::ContextGuard ctx{Allocator{kCPUpinned}};
                Layout             layout{d.batch_size, cfg.in_channels, cfg.image_height, cfg.image_width};
                d.batch_input = {layout, cfg.data_type, kCPUpinned};
            }

            ssize_t batch_offset = 0;
            for (const auto& pixel_value : pixel_values) {
                TM_CHECK_EQ(pixel_value.ndim(), 4);
                TM_CHECK_EQ(pixel_value.shape(1), cfg.in_channels);
                TM_CHECK_EQ(pixel_value.shape(2), cfg.image_height);
                TM_CHECK_EQ(pixel_value.shape(3), cfg.image_width);
                TM_CHECK_EQ(pixel_value.dtype(), d.batch_input.dtype());
                Copy(pixel_value, d.batch_input.slice(batch_offset, pixel_value.shape(0)));
                batch_offset += pixel_value.shape(0);
            }
            TM_CHECK_EQ(batch_offset, d.batch_size);

            // attention meta
            d.seq_len   = cfg.num_patches + 1;
            d.token_num = d.batch_size * d.seq_len;

            if (d.attn_cu_seqlens.size() < d.batch_size + 1) {
                d.attn_cu_seqlens = Tensor_<int>{{d.batch_size + 1}, kDEVICE};
            }
            if (d.attn_finished.size() < d.batch_size) {
                d.attn_finished = Tensor_<bool>{{d.batch_size}, kDEVICE};
            }
            if (attn_cu_seqlens_buf_.size() < d.batch_size + 1) {
                core::ContextGuard ctx{Allocator{kCPUpinned}};
                attn_cu_seqlens_buf_ = {d.batch_size + 1, kCPUpinned};
            }

            for (int i = 0; i <= d.batch_size; ++i) {
                attn_cu_seqlens_buf_[i] = i * d.seq_len;
            }
            copy(attn_cu_seqlens_buf_.data(), d.batch_size + 1, d.attn_cu_seqlens.data());
            Clear(d.attn_finished.slice(0, d.batch_size));
        }
        h_tp_group->Sync();
    }

    Tensor PatchEmbedding(Data& d)
    {
        const auto& cfg    = config_;
        auto        stream = core::Context::stream().handle();

        Tensor host_input = d.batch_input.slice(0, d.batch_size);
        Tensor input      = empty_like(host_input, kDEVICE);
        Copy(host_input, input);
        TM_CUDA_CHECK(cudaGetLastError());

        Tensor patches{{d.batch_size * cfg.num_patches, cfg.patch_in_dim}, cfg.data_type, kDEVICE};
        invokeInternVitPatchify(patches,
                                input,
                                d.batch_size,
                                cfg.in_channels,
                                cfg.image_height,
                                cfg.image_width,
                                cfg.patch_height,
                                cfg.patch_width,
                                stream);

        Tensor patch_embeds;
        TM_SCOPE_CALL(linear_.Forward(patches, *weights_.patch_embed, patch_embeds));
        TM_CUDA_CHECK(cudaGetLastError());

        Tensor hidden{{d.token_num, cfg.hidden_dim}, cfg.data_type, kDEVICE};
        invokeInternVitAddEmbeddings(hidden,
                                     patch_embeds,
                                     weights_.patch_embed->bias,
                                     weights_.cls_token,
                                     weights_.position_embeddings,
                                     d.batch_size,
                                     cfg.num_patches,
                                     cfg.hidden_dim,
                                     stream);
        return hidden;
    }

    template<typename T>
    AttentionParams<T> CreateVitAttentionParams(
        Tensor& attn_output, Tensor& qkv, Tensor& kv, const Data& d, const AttentionWeight& attn, int layer_id)
    {
        const int local_head_num = attn.head_num / attn.tp_size;
        const int head_dim       = attn.head_dim;
        const int token_num      = d.token_num;

        AttentionParams<T> params{};
        params.out = (T*)attn_output.raw_data();
        params.q   = (T*)qkv.raw_data();

        params.stride = (int64_t)local_head_num * 3 * head_dim;

        params.cu_q_len = d.attn_cu_seqlens.data();
        params.cu_k_len = d.attn_cu_seqlens.data();
        params.finished = d.attn_finished.data();

        params.linear_iter_params = LinearIteratorParams{
            kv.raw_data(),
            2 * token_num * head_dim,
            token_num * head_dim,
        };

        params.token_num  = token_num;
        params.batch_size = d.batch_size;
        params.max_q_len  = d.seq_len;
        params.max_k_len  = d.seq_len;

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

    template<typename T>
    void Attn(Tensor& input, Tensor& output, Data& d, int layer_id)
    {
        auto* attn   = weights_.block(layer_id)->attention.get();
        auto  stream = core::Context::stream().handle();

        Tensor qkv;
        TM_SCOPE_CALL(linear_.Forward(input, *attn->w_qkv, qkv));
        TM_CUDA_CHECK(cudaGetLastError());

        const int local_head_num = attn->head_num / attn->tp_size;
        const int head_dim       = attn->head_dim;
        const int local_dim      = local_head_num * head_dim;

        ApplyBias(qkv, attn->w_qkv->bias, stream);

        if (attn->q_norm && attn->k_norm) {
            Tensor sums{output.buffer().view(kFloat), {2, d.token_num}};
            invokeInternVitPreRMSNorm(sums, qkv, local_dim, stream);
            if (attn->tp_size > 1) {
                AllReduceSum(sums, stream);
            }
            invokeInternVitPostRMSNorm(qkv,
                                       sums,
                                       attn->q_norm->weight,
                                       attn->k_norm->weight,
                                       local_dim,
                                       config_.hidden_dim,
                                       attn->q_norm->norm_eps_,
                                       stream);
        }

        Tensor kv{{local_head_num, 2, d.token_num, head_dim}, qkv.dtype(), qkv.device()};
        invokeInternVitPrepareQKV(kv, qkv, local_head_num, head_dim, stream);

        Tensor attn_output{{d.token_num, local_dim}, qkv.dtype(), qkv.device()};
        auto   params = CreateVitAttentionParams<T>(attn_output, qkv, kv, d, *attn, layer_id);
        dispatchAttention<T>(params);
        TM_CUDA_CHECK(cudaGetLastError());

        TM_SCOPE_CALL(linear_.Forward(attn_output, *attn->wo, output));
        TM_CUDA_CHECK(cudaGetLastError());
        if (attn->tp_size > 1) {
            AllReduceSum(output, stream);
        }
    }

    void Mlp(Tensor& input, Tensor& output, int layer_id)
    {
        auto* block  = weights_.block(layer_id);
        auto  stream = core::Context::stream().handle();

        Tensor inter;
        TM_SCOPE_CALL(linear_.Forward(input, *block->mlp_fc1, inter));
        TM_CUDA_CHECK(cudaGetLastError());

        invokeAddBiasActivation(inter, block->mlp_fc1->bias, ActivationType::kGelu, stream);

        TM_SCOPE_CALL(linear_.Forward(inter, *block->mlp_fc2, output));
        TM_CUDA_CHECK(cudaGetLastError());
        AllReduceSum(output, stream);
    }

    Tensor Projector(Tensor& hidden, Data& d, Buffer symm_buf)
    {
        const auto& cfg    = config_;
        auto        stream = core::Context::stream().handle();

        const int grid_size = (int)std::sqrt(cfg.num_patches);
        TM_CHECK_EQ(grid_size * grid_size, cfg.num_patches);
        TM_CHECK_EQ(cfg.image_seq_length, (grid_size / 2) * (grid_size / 2));

        Tensor shuffled{{d.batch_size * cfg.image_seq_length, cfg.hidden_dim * 4}, cfg.data_type, kDEVICE};
        invokeInternVitPixelShuffle(shuffled, hidden, grid_size, stream);

        Tensor projector_normed{{d.batch_size * cfg.image_seq_length, cfg.hidden_dim * 4}, cfg.data_type, kDEVICE};
        invokeLayerNorm(projector_normed,
                        shuffled,
                        weights_.projector_norm->weight,
                        weights_.projector_norm->bias,
                        weights_.projector_norm->norm_eps_,
                        stream);
        TM_CUDA_CHECK(cudaGetLastError());

        Tensor inter;
        TM_SCOPE_CALL(linear_.Forward(projector_normed, *weights_.projector_fc1, inter));
        TM_CUDA_CHECK(cudaGetLastError());

        invokeAddBiasActivation(inter, weights_.projector_fc1->bias, ActivationType::kGelu, stream);

        Tensor output;
        if (tp_size_ > 1) {
            output = {symm_buf.view(config_.data_type), {inter.shape(0), weights_.projector_fc2->output_dim}};
        }
        TM_SCOPE_CALL(linear_.Forward(inter, *weights_.projector_fc2, output));
        TM_CUDA_CHECK(cudaGetLastError());
        AllReduceSum(output, stream);

        Tensor result = empty_like(output);
        ApplyBias(result, output, weights_.projector_fc2->bias, stream);
        TM_CUDA_CHECK(cudaGetLastError());

        return result;
    }

    void Forward(int phase, TensorMap& args)
    {
        const auto& cfg = config_;
        auto&       d   = data_.at(phase);
        if (d.batch_size == 0) {
            return;
        }

        auto stream   = core::Context::stream().handle();
        auto residual = PatchEmbedding(d);

        Buffer symm_buf = args.contains("symm_buf") ? args.at("symm_buf").buffer() : Buffer{};

        Tensor hidden_states = [&]() {
            if (symm_buf) {
                return Tensor{symm_buf.view(cfg.data_type), {d.token_num, cfg.hidden_dim}};
            }
            else {
                return Tensor{{d.token_num, cfg.hidden_dim}, cfg.data_type, kDEVICE};
            }
        }();

        ApplyNorm(hidden_states, residual, *weights_.block(0)->norm1, config_.norm_type);

        for (int layer_id = 0; layer_id < cfg.depth; ++layer_id) {
            auto* block = weights_.block(layer_id);

            auto invoke = [&](auto t) {
                using T = decltype(t);
                Attn<T>(hidden_states, hidden_states, d, layer_id);
            };
            TM_DISPATCH_PRIMARY_DTYPES(hidden_states.dtype(), invoke);
            ResidualScaleNorm(hidden_states,
                              residual,
                              hidden_states,
                              block->lambda_1,
                              block->attention->wo->bias,
                              block->norm2.get(),
                              config_.norm_type);

            Mlp(hidden_states, hidden_states, layer_id);

            const bool is_last_layer = layer_id + 1 == cfg.depth;
            ResidualScaleNorm(hidden_states,
                              residual,
                              hidden_states,
                              block->lambda_2,
                              block->mlp_fc2->bias,
                              is_last_layer ? nullptr : weights_.block(layer_id + 1)->norm1.get(),
                              is_last_layer ? NormType::kNone : config_.norm_type);
        }

        Tensor image_embeds = Projector(residual, d, symm_buf);
        EnsureFloatDtype(image_embeds, engine_data_type_);

        args.produce("multimodal",
                     MultiModalEmbeddingData{image_embeds, d.image_embeds_coords, d.input_embeds_coords}.buf());
    }
};

InternVit::InternVit(const EngineParam& engine, const Context& ctx, const InternVitWeight& weights, int phases):
    impl_{std::make_unique<Impl>(engine, ctx, weights, phases)}
{
}

InternVit::~InternVit() = default;

void InternVit::Run(BatchOp op, int phase, TensorMap& env)
{
    switch (op) {
        case BatchOp::kAdd:
            return impl_->Add(phase, env);
        case BatchOp::kSetup:
            return impl_->Setup(phase, env);
        case BatchOp::kForward:
            return impl_->Forward(phase, env);
        default:
            return;
    }
}

}  // namespace turbomind
