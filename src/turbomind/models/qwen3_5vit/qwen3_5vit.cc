// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/qwen3_5vit/qwen3_5vit.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/kernels/norm/layer_norm.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/layer_norm_weight.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/qwen3_5vit/bias_gelu.h"
#include "src/turbomind/models/qwen3_5vit/fast_pos_embed.h"
#include "src/turbomind/models/qwen3_5vit/fast_rotary_pos_emb.h"
#include "src/turbomind/models/qwen3_5vit/fused_embed_merge.h"
#include "src/turbomind/models/qwen3_5vit/mrope_position_ids.h"
#include "src/turbomind/models/qwen3_5vit/qkv_preprocess.h"
#include "src/turbomind/models/qwen3_5vit/qwen3_5vit_block_weight.h"
#include "src/turbomind/models/qwen3_5vit/qwen3_5vit_weight.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>

namespace turbomind {

namespace {

std::string TensorShapeString(const Tensor& tensor)
{
    std::string str   = "[";
    const auto& shape = tensor.shape();
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) {
            str += ", ";
        }
        str += std::to_string(shape[i]);
    }
    str += "]";
    return str;
}

[[maybe_unused]] void DumpTensorToBin(const Tensor& tensor, const std::string& output_path)
{
    TM_CHECK(tensor) << "Cannot dump an empty tensor to " << output_path;
    TM_CHECK(tensor.is_contiguous()) << "Only contiguous tensors can be dumped: " << tensor;
    TM_LOG_ERROR("DumpTensorToBin: file={}, shape={}", output_path, TensorShapeString(tensor));

    Tensor host_tensor{tensor.layout(), tensor.dtype(), kCPU};
    Copy(tensor, host_tensor);
    core::Context::stream().Sync();

    std::ofstream ofs(output_path, std::ios::binary);
    TM_CHECK(ofs.is_open()) << "Failed to open " << output_path << " for writing";

    if (const auto bytes = host_tensor.byte_size()) {
        ofs.write(static_cast<const char*>(host_tensor.raw_data()), bytes);
        TM_CHECK(ofs.good()) << "Failed to write tensor to " << output_path;
    }
}

[[maybe_unused]] void ReadTensorFromBin(Tensor& tensor, const std::string& input_path)
{
    TM_CHECK(tensor) << "Cannot read an empty tensor from " << input_path;
    TM_CHECK(tensor.is_contiguous()) << "Only contiguous tensors can be read: " << tensor;
    TM_LOG_ERROR("ReadTensorFromBin: file={}, shape={}", input_path, TensorShapeString(tensor));

    const bool is_host_tensor = tensor.device().type == kCPU || tensor.device().type == kCPUpinned;
    Tensor     host_tensor    = is_host_tensor ? tensor : Tensor{tensor.layout(), tensor.dtype(), kCPU};
    const auto expected_bytes = host_tensor.byte_size();

    std::ifstream ifs(input_path, std::ios::binary | std::ios::ate);
    TM_CHECK(ifs.is_open()) << "Failed to open " << input_path << " for reading";

    const std::streamoff actual_bytes = ifs.tellg();
    TM_CHECK_GE(actual_bytes, 0) << "Failed to get size of " << input_path;
    TM_CHECK_EQ(actual_bytes, expected_bytes)
        << "Unexpected tensor file size for " << input_path << ", tensor " << tensor;

    ifs.seekg(0, std::ios::beg);
    TM_CHECK(ifs.good()) << "Failed to seek " << input_path;

    if (expected_bytes) {
        ifs.read(static_cast<char*>(host_tensor.raw_data()), expected_bytes);
        TM_CHECK(ifs.good()) << "Failed to read tensor from " << input_path;
    }

    if (!is_host_tensor) {
        Copy(host_tensor, tensor);
        core::Context::stream().Sync();
    }
}

}  // namespace

struct Qwen3_5Vit::Impl {
    const Qwen3_5VitWeight&       weights_;
    const core::Qwen3_5VitConfig& config_;
    int                           phases_;
    LlamaLinear&                  linear_;
    comm::DeviceCommImpl* const   d_comm_;
    const int                     tp_group_;

    Buffer_<int> grid_thws_buf_;     // (t, h, w)
    Buffer_<int> grid_offsets_buf_;  // (t*h*w, h*w)
    Buffer_<int> mapped_idx_buf_;    // [batch]
    Buffer_<int> attn_cu_seqlens_buf_;

    struct Data {
        Tensor_<float>                   batch_input;
        int                              batch_size;
        std::vector<std::array<int, 3>>  grid_thws_host;
        std::vector<std::pair<int, int>> image_embeds_coords;  // (size, pos) for image embeddings
        std::vector<std::pair<int, int>> input_embeds_coords;  // (size, pos) for input embeddings

        // for fast_pos_embed
        Tensor_<int> grid_thws;
        Tensor_<int> grid_offsets;
        Tensor_<int> mapped_idx;
        int          total_hw;

        // for full attention, one sequence per temporal frame
        Tensor_<int>  attn_cu_seqlens;
        Tensor_<bool> attn_finished;
        int           attn_batch_size;
        int           max_attn_len;

        // mrope position-id scratch (per-phase pinned host buffers for fast H2D)
        Buffer_<int> mrope_segs_host;  // reinterpreted as MropeSegment[], 7 ints per segment
        Buffer_<int> mrope_length_host;
        Buffer_<int> mrope_delta_host;

        // mrope outputs — owned here so UnifiedAttentionLayer can safely borrow() across env clears.
        Buffer_<int> mrope_segs_dev;        // device-side segment scratch, grown alongside host
        Tensor_<int> mrope_position_ids;    // (bsz, max_active_end, 3), empty when no slot needs table
        Tensor_<int> mrope_length;          // (bsz,)
        Tensor_<int> mrope_position_delta;  // (bsz,)

        void Clear()
        {
            batch_size      = 0;
            total_hw        = 0;
            attn_batch_size = 0;
            max_attn_len    = 0;
            grid_thws_host.clear();
            image_embeds_coords.clear();
            input_embeds_coords.clear();
        }
    };

    static constexpr int kMropeSegInts = sizeof(MropeSegment) / sizeof(int);
    static_assert(sizeof(MropeSegment) % sizeof(int) == 0);

    std::vector<Data> data_;

    Impl(const EngineParam& engine, const Context& ctx, const Qwen3_5VitWeight& weights, int phases):
        weights_{weights},
        config_{weights.config()},
        phases_{phases},
        linear_{*ctx.linear},
        d_comm_{ctx.comm.d_comm},
        tp_group_{ctx.comm.d_tp_group}
    {
        auto& cfg = weights.config();
        for (int i = 0; i < phases; ++i) {
            auto& d             = data_.emplace_back();
            d.batch_input       = {{engine.max_forward_token_num, cfg.patch_in_dim}, kCPUpinned};
            d.mrope_length_host = {engine.max_batch_size, kCPUpinned};
            d.mrope_delta_host  = {engine.max_batch_size, kCPUpinned};
            // Generous initial capacity: typical batches emit << bsz * 8 segments. Lazily grown below.
            d.mrope_segs_host = {engine.max_batch_size * 8 * (ssize_t)kMropeSegInts, kCPUpinned};

            // mrope outputs at worst-case shape so Setup() never reallocates them. Rows beyond
            // the current bsz (and rows whose length[i] == 0) are stale but unreachable from
            // FastRoPE — it only reads position_ids[3 * timestep] when `timestep < length[i]`.
            d.mrope_length         = Tensor_<int>{{engine.max_batch_size}, kDEVICE};
            d.mrope_position_delta = Tensor_<int>{{engine.max_batch_size}, kDEVICE};
            d.mrope_position_ids   = Tensor_<int>{{engine.max_batch_size, engine.session_len, 3}, kDEVICE};
        }

        // should be large enough to hold all patches
        grid_thws_buf_       = {engine.max_forward_token_num * 3, kCPUpinned};
        grid_offsets_buf_    = {engine.max_forward_token_num * 2, kCPUpinned};
        mapped_idx_buf_      = {engine.max_forward_token_num, kCPUpinned};
        attn_cu_seqlens_buf_ = {engine.max_forward_token_num + 1, kCPUpinned};
    }

    void AllReduceSum(Tensor& tensor, cudaStream_t stream) const
    {
        if (d_comm_) {
            d_comm_->AllReduceSum(
                tensor.raw_data(), tensor.raw_data(), tensor.size(), tensor.dtype(), tp_group_, stream);
            TM_CUDA_CHECK(cudaGetLastError());
        }
    }

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
                                    d.grid_thws.shape(0),
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
        // keyed by the same natural flat index that `mapped_idx` already carries. Visual q/k
        // are reordered into this adjacent-pair layout at export time.
        Tensor rotary_pos_emb = {{d.total_hw, head_dim}, cfg.data_type, kDEVICE};
        invokeQwen3VitRotaryPosEmb(rotary_pos_emb.raw_data(),
                                   cfg.data_type,
                                   d.grid_thws.data(),
                                   d.grid_offsets.data(),
                                   d.grid_thws.shape(0),
                                   d.total_hw,
                                   head_dim,
                                   /*theta=*/10000.0f,
                                   core::Context::stream().handle());
        TM_CUDA_CHECK(cudaGetLastError());
        env.produce("rotary_pos_emb", rotary_pos_emb);
    }

    Tensor PatchEmbedding(Data& d)
    {
        auto& cfg = weights_.config();

        Tensor host_input = d.batch_input.slice(0, d.batch_size);
        Tensor input      = empty_like(host_input, kDEVICE);

        Copy(host_input, input);
        TM_CUDA_CHECK(cudaGetLastError());

        EnsureFloatDtype(input, cfg.data_type);
        TM_CUDA_CHECK(cudaGetLastError());

        Tensor output;
        TM_SCOPE_CALL(linear_.Forward(input, *weights_.patch_embed, output));
        return output;
    }

    int Add(RequestCache& c)
    {
        TM_LOG_ERROR("add");
        const auto& [r, s] = std::tie(*c.req, *c.seq);
        if (r.mm_inputs && !r.mm_inputs->is_null()) {
            if ((not r.session.start_flag) or (not r.session.end_flag)) {
                // only support non-interactive inference
                return Request::kInvalid;
            }

            const auto& mm_inputs = r.mm_inputs->get<multimodal::Array>();
            for (const auto& item : mm_inputs) {
                const auto& doc      = item.get<multimodal::Document>();
                const auto& modality = doc.at("modality").get<std::string>();
                const auto& ranges   = doc.at("offset").get<multimodal::Array>();
                const int   offset   = ranges[0].get<int64_t>();
                const int   tokens   = ranges[1].get<int64_t>() - offset;

                if (modality != "image" && modality != "video") {
                    return Request::kInvalid;
                }
                const bool is_image = modality == "image";

                std::array<int, 3> grid_thw = [&]() {
                    const auto  key  = is_image ? "image_grid_thw" : "video_grid_thw";
                    const auto& ten  = doc.at(key).get<Tensor>();
                    auto        data = ten.data<int64_t>();
                    return std::array<int, 3>{(int)data[0], (int)data[1], (int)data[2]};
                }();

                auto data = [&]() {
                    const auto key = is_image ? "pixel_values" : "pixel_values_videos";
                    return doc.at(key).get<Tensor>();
                }();

                auto mm_item = std::make_shared<MultiModalData>(
                    MultiModalData{data, Interval{offset, Interval::Size{tokens}}, grid_thw});
                s.multimodal_inputs.push_back(mm_item);
            }
            TM_LOG_ERROR("add done");
        }

        return Request::kOk;
    }

    void Add(int phase, TensorMap& env)
    {
        // convert mm_inputs(list[dict]) to internal MultiModalData
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
    // Per-request layout: one row in `(max_batch_size, session_len, 3)` for each slot. Prefill
    // slots with multimodal_inputs get their active range written by `invokeMropePositionIds`
    // from a clipped list of MropeSegment descriptors (one per text/image run). All other slots
    // (decode + text-only prefill) get `length[i] = 0` so FastRoPE falls through to the closed-
    // form `timestep + delta` path and never reads the stale row.
    //
    // The three output tensors live on `Data` (allocated worst-case in the ctor). env shares
    // ownership via shared_ptr; UAL borrows safely across env clears.
    void SetupMrope(int phase, TensorMap& env)
    {
        auto& d  = data_.at(phase);
        auto& b  = *env.at("batch").data<BatchData*>()[0];
        auto& rc = b.rc;

        const int bsz = (int)rc.size();
        if (bsz <= 0) {
            return;
        }

        const int S = weights_.config().spatial_merge_size;

        // 1) One pass to upper-bound segment count, then size host + device scratch in one shot.
        //    Worst case per prefill slot with mrope: 2*num_images + 1 segments.
        int upper_segs = 0;
        for (int i = 0; i < bsz; ++i) {
            const auto& c = *rc[i];
            if (!c.autoregres && !c.seq->multimodal_inputs.empty()) {
                upper_segs += 2 * (int)c.seq->multimodal_inputs.size() + 1;
            }
        }
        const ssize_t upper_ints = (ssize_t)upper_segs * kMropeSegInts;
        if (upper_ints > d.mrope_segs_host.size()) {
            core::ContextGuard ctx{Allocator{kCPUpinned}};
            d.mrope_segs_host = Buffer_<int>{upper_ints, kCPUpinned};
        }
        if (upper_ints > d.mrope_segs_dev.size()) {
            d.mrope_segs_dev = Buffer_<int>{upper_ints, kDEVICE};
        }

        // 2) Unified per-request walk — always advance mm_off; emit segments only for needs_table.
        auto* segs        = reinterpret_cast<MropeSegment*>(d.mrope_segs_host.data());
        int   n_segs      = 0;
        int   max_seg_len = 0;

        for (int i = 0; i < bsz; ++i) {
            const auto& c            = *rc[i];
            const auto& s            = *c.seq;
            const int   seq_len      = (int)c.req->inputs.at("input_ids").shape(0);
            const bool  needs_table  = !c.autoregres && !s.multimodal_inputs.empty();
            const int   active_start = c.history_len + c.alpha;
            const int   active_end   = active_start + c.input_len;

            auto emit = [&](int run_start, int run_n, int run_base, int h2, int w2) {
                const int a = std::max(run_start, active_start);
                const int b = std::min(run_start + run_n, active_end);
                if (a >= b) {
                    return;
                }
                const int local_off = a - run_start;
                segs[n_segs++]      = MropeSegment{
                    i,
                    a,
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
                const int new_pos = std::max({t, h2, w2});
                pos               = img_base + new_pos;
                mm_off += new_pos - n_tok;
            }
            if (needs_table && row < seq_len) {
                emit(row, seq_len - row, pos, /*h2=*/0, /*w2=*/0);
            }

            d.mrope_length_host.data()[i] = needs_table ? seq_len : 0;
            d.mrope_delta_host.data()[i]  = mm_off;
        }

        // 3) Copy the bsz prefix of length / delta into the pre-allocated device tensors.
        //    Rows beyond bsz are untouched (UAL never reads them).
        Copy(d.mrope_length_host.slice(0, bsz), d.mrope_length.buffer().slice(0, bsz));
        Copy(d.mrope_delta_host.slice(0, bsz), d.mrope_position_delta.buffer().slice(0, bsz));

        // 4) Populate position_ids only when a slot actually needs the table. Rows for slots
        //    with length[i] == 0 are unreachable from FastRoPE, so leaving them stale is safe.
        if (n_segs > 0) {
            const ssize_t segs_ints = (ssize_t)n_segs * kMropeSegInts;
            Copy(d.mrope_segs_host.slice(0, segs_ints), d.mrope_segs_dev.slice(0, segs_ints));
            invokeMropePositionIds(d.mrope_position_ids.data(),
                                   (int)d.mrope_position_ids.stride(0),
                                   reinterpret_cast<const MropeSegment*>(d.mrope_segs_dev.data()),
                                   n_segs,
                                   max_seg_len,
                                   core::Context::stream().handle());
            TM_CUDA_CHECK(cudaGetLastError());
        }

        // 5) Publish all three — the consumer relies on this contract unconditionally.
        env.produce("mrope_length", d.mrope_length);
        env.produce("mrope_position_delta", d.mrope_position_delta);
        env.produce("mrope_position_ids", d.mrope_position_ids);
    }

    void Setup(int phase, TensorMap& env)
    {
        // create batch data according to scheduled sequences
        auto& d    = data_.at(phase);
        auto& b    = *env.at("batch").data<BatchData*>()[0];
        auto& copy = *env.at("copy").data<BatchCopy*>()[0];
        auto& cfg  = weights_.config();

        int input_ids_offsets    = 0;
        int image_embeds_offsets = 0;
        d.Clear();
        std::vector<Tensor> pixel_values;

        // collect image/video pixel values, grid_thws and embeds_coords
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

                        auto& grid_thw = mm->grid_thw;
                        d.grid_thws_host.emplace_back(grid_thw);
                        auto prod = std::accumulate(grid_thw.begin(), grid_thw.end(), 1, std::multiplies<int>());
                        image_embeds_offsets += (prod / cfg.spatial_merge_size / cfg.spatial_merge_size);
                    }
                }
            }

            input_ids_offsets += c.autoregres ? 1 : c.input_len;
        }

        // copy pixel values to batch input
        if (d.batch_size > 0) {
            if (d.batch_size > d.batch_input.shape(0)) {
                core::ContextGuard ctx{Allocator{kCPUpinned}};
                d.batch_input = {{d.batch_size, cfg.patch_in_dim}, kCPUpinned};
            }
            auto embed_ptr = d.batch_input.data();
            for (const auto& pixel_value : pixel_values) {
                embed_ptr = std::copy_n(pixel_value.data<float>(), pixel_value.size(), embed_ptr);
            }
        }

        // setup fast_pos_embed
        if (const int num_grids = (int)d.grid_thws_host.size(); num_grids > 0) {
            for (const auto& [t, h, w] : d.grid_thws_host) {
                d.attn_batch_size += t;
                d.max_attn_len = std::max(d.max_attn_len, h * w);
            }
            if (grid_thws_buf_.size() < (ssize_t)num_grids * 3) {
                core::ContextGuard ctx{Allocator{kCPUpinned}};
                grid_thws_buf_    = Buffer_<int>{num_grids * 3, kCPUpinned};
                grid_offsets_buf_ = Buffer_<int>{num_grids * 2, kCPUpinned};
            }
            if (mapped_idx_buf_.size() < d.batch_size) {
                core::ContextGuard ctx{Allocator{kCPUpinned}};
                mapped_idx_buf_ = Buffer_<int>{d.batch_size, kCPUpinned};
            }
            if (attn_cu_seqlens_buf_.size() < (ssize_t)d.attn_batch_size + 1) {
                core::ContextGuard ctx{Allocator{kCPUpinned}};
                attn_cu_seqlens_buf_ = Buffer_<int>{d.attn_batch_size + 1, kCPUpinned};
            }
            d.grid_thws       = {{num_grids, 3}, kDEVICE};
            d.grid_offsets    = {{num_grids, 2}, kDEVICE};
            d.mapped_idx      = {{d.batch_size}, kDEVICE};
            d.attn_cu_seqlens = {{d.attn_batch_size + 1}, kDEVICE};
            d.attn_finished   = {{d.attn_batch_size}, kDEVICE};

            std::pair<int, int> offset{};
            int                 attn_seq_idx            = 0;
            int                 attn_offset             = 0;
            attn_cu_seqlens_buf_.data()[attn_seq_idx++] = 0;
            for (int i = 0; i < num_grids; ++i) {
                const auto& [t, h, w] = d.grid_thws_host[i];
                TM_CHECK(h % cfg.spatial_merge_size == 0);
                TM_CHECK(w % cfg.spatial_merge_size == 0);
                const int hw = h * w;
                for (int tt = 0; tt < t; ++tt) {
                    attn_offset += hw;
                    attn_cu_seqlens_buf_.data()[attn_seq_idx++] = attn_offset;
                }

                grid_thws_buf_.data()[i * 3]        = t;
                grid_thws_buf_.data()[i * 3 + 1]    = h;
                grid_thws_buf_.data()[i * 3 + 2]    = w;
                grid_offsets_buf_.data()[i * 2]     = offset.first;
                grid_offsets_buf_.data()[i * 2 + 1] = offset.second;

                // compute mapped_idx
                TM_CHECK(offset.first + t * h * w <= d.batch_size);
                const int S   = cfg.spatial_merge_size;
                int*      buf = mapped_idx_buf_.data();
                int       pos = offset.first;
                for (int h_outer = 0; h_outer < h / S; ++h_outer) {
                    for (int w_outer = 0; w_outer < w / S; ++w_outer) {
                        for (int h_inner = 0; h_inner < S; ++h_inner) {
                            for (int w_inner = 0; w_inner < S; ++w_inner) {
                                const int ii = h_outer * S + h_inner;
                                const int jj = w_outer * S + w_inner;
                                buf[pos++]   = offset.second + ii * w + jj;
                            }
                        }
                    }
                }
                for (int tt = 1; tt < t; ++tt) {
                    std::memcpy(buf + offset.first + tt * hw, buf + offset.first, hw * sizeof(int));
                }
                pos = offset.first + t * hw;
                TM_CHECK_EQ(pos, offset.first + t * h * w);
                offset.first += t * h * w;
                offset.second += h * w;
            }
            TM_CHECK_EQ(offset.first, d.batch_size);
            TM_CHECK_EQ(attn_offset, d.batch_size);
            TM_CHECK_EQ(attn_seq_idx, d.attn_batch_size + 1);
            d.total_hw = offset.second;
            copy(grid_thws_buf_.data(), num_grids * 3, d.grid_thws.data());
            copy(grid_offsets_buf_.data(), num_grids * 2, d.grid_offsets.data());
            copy(mapped_idx_buf_.data(), d.batch_size, d.mapped_idx.data());
            copy(attn_cu_seqlens_buf_.data(), d.attn_batch_size + 1, d.attn_cu_seqlens.data());
            Clear(d.attn_finished);
        }

        SetupMrope(phase, env);
    }

    void Prepare(int phase, TensorMap& env)
    {
        auto& d = data_.at(phase);
        if (d.batch_size == 0) {
            return;
        }

        // produce non-merged pos-embeds and weights
        FastPosEmbedInterpolate(d, env);

        // produce rotary_pos_emb
        RotPosEmb(d, env);
    }

    void Forward(int phase, TensorMap& args)
    {
        auto& d = data_.at(phase);
        if (d.batch_size == 0) {
            return;
        }

        auto& cfg = weights_.config();

        // 1) patch_embed (Linear without bias, the bias will be folded into the fused kernel)
        auto residual          = PatchEmbedding(d);
        auto pos_embeds        = args.consume("pos_embeds");
        auto pos_embed_weights = args.consume("pos_embed_weights");
        auto rotary_pos_emb    = args.consume("rotary_pos_emb");
        auto stream            = core::Context::stream().handle();

        // 2) fused pos-embed gather/merge:
        // residual[pos, d] += Σ_k w[k] * pos_embeds[mapped*4+k, d] + bias[d]
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

        // DumpTensorToBin(pos_embeds, "pos_embeds_" + std::to_string(d_comm_->rank(tp_group_)) + ".bin");
        // DumpTensorToBin(pos_embed_weights, "pos_embed_weights_" + std::to_string(d_comm_->rank(tp_group_)) + ".bin");

        // DumpTensorToBin(residual, "residual_" + std::to_string(d_comm_->rank(tp_group_)) + ".bin");
        // TM_LOG_ERROR("residual {} {}, dtype {}", residual.shape(0), residual.shape(1), (int)residual.dtype());

        // 3) decoder
        Tensor hidden_states = [&]() {
            Buffer symm_buf = args.contains("symm_buf") ? args.at("symm_buf").buffer() : Buffer{};
            if (symm_buf && d.batch_size * cfg.hidden_dim <= symm_buf.size() / turbomind::byte_size(cfg.data_type)) {
                return Tensor{symm_buf.view(cfg.data_type), {d.batch_size, cfg.hidden_dim}};
            }
            else {
                return Tensor{{d.batch_size, cfg.hidden_dim}, cfg.data_type, kDEVICE};
            }
        }();

        invokeLayerNorm(hidden_states,
                        residual,
                        weights_.block(0)->norm1->weight,
                        weights_.block(0)->norm1->bias,
                        weights_.block(0)->norm1->norm_eps_,
                        stream);
        TM_CUDA_CHECK(cudaGetLastError());

        for (int layer_id = 0; layer_id < cfg.depth; ++layer_id) {

            // attn
            auto invoke = [&](auto t) {
                using T = decltype(t);
                Attn<T>(hidden_states, hidden_states, d, layer_id, rotary_pos_emb);
            };
            TM_DISPATCH_PRIMARY_DTYPES(hidden_states.dtype(), invoke);

            AllReduceSum(hidden_states, stream);

            auto* block = weights_.block(layer_id);
            invokeResidualBiasLayerNorm(hidden_states.raw_data(),
                                        residual.raw_data(),
                                        block->norm2->weight.raw_data(),
                                        block->norm2->bias.data_or((void*)nullptr),
                                        block->attention->wo->bias.data_or((void*)nullptr),
                                        hidden_states.dtype(),
                                        cfg.hidden_dim,
                                        d.batch_size,
                                        block->norm2->norm_eps_,
                                        stream);
            TM_CUDA_CHECK(cudaGetLastError());

            // mlp
            Mlp(hidden_states, hidden_states, d, layer_id);
            AllReduceSum(hidden_states, stream);

            const auto* next_norm =
                layer_id + 1 < cfg.depth ? weights_.block(layer_id + 1)->norm1.get() : weights_.merger_norm.get();
            TM_CHECK_NOTNULL(next_norm);
            invokeResidualBiasLayerNorm(hidden_states.raw_data(),
                                        residual.raw_data(),
                                        next_norm->weight.raw_data(),
                                        next_norm->bias.data_or((void*)nullptr),
                                        block->mlp_fc2->bias.data_or((void*)nullptr),
                                        hidden_states.dtype(),
                                        cfg.hidden_dim,
                                        d.batch_size,
                                        next_norm->norm_eps_,
                                        stream);
            TM_CUDA_CHECK(cudaGetLastError());
        }

        // ReadTensorFromBin(hidden_states, "merger_input.bin");
        Tensor image_embeds = Merger(hidden_states);

        args.produce("multimodal",
                     MultiModalEmbeddingData{image_embeds, d.image_embeds_coords, d.input_embeds_coords}.buf());
    }

    template<typename T>
    AttentionParams<T> CreateVitAttentionParams(
        Tensor& attn_context, Tensor& qkv, Tensor& kv, Data& d, const AttentionWeight& attn, int layer_id)
    {
        const int local_head_num = attn.head_num / attn.tp_size;
        const int head_dim       = attn.head_dim;
        const int token_num      = d.batch_size;

        AttentionParams<T> params{};
        params.out = (T*)attn_context.raw_data();
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
        params.batch_size = d.attn_batch_size;
        params.max_q_len  = d.max_attn_len;
        params.max_k_len  = d.max_attn_len;

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

        Tensor tmp_kv{{local_head_num, 2, d.batch_size, head_dim}, qkv.dtype(), qkv.device()};
        invokeQwen3_5VitPrepareQKV(qkv.raw_data(),
                                   tmp_kv.raw_data(),
                                   attn->w_qkv->bias.raw_data(),
                                   rotary_pos_emb.raw_data(),
                                   d.mapped_idx.data(),
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

        TM_CHECK(block);
        TM_CHECK_EQ(input.shape(0), d.batch_size);
        TM_CHECK_EQ(input.shape(1), config_.hidden_dim);

        Tensor inter;
        TM_SCOPE_CALL(linear_.Forward(input, *block->mlp_fc1, inter));
        TM_CUDA_CHECK(cudaGetLastError());

        invokeQwen3_5VitBiasActivation(inter, block->mlp_fc1->bias, ActivationType::kGeluPytorchTanh, stream);
        TM_CUDA_CHECK(cudaGetLastError());

        TM_SCOPE_CALL(linear_.Forward(inter, *block->mlp_fc2, output));
        TM_CUDA_CHECK(cudaGetLastError());
    }

    Tensor Merger(Tensor& input)
    {
        auto& cfg    = config_;
        auto  stream = core::Context::stream().handle();

        const int merge_area   = cfg.spatial_merge_size * cfg.spatial_merge_size;
        Tensor    merged_input = input.view({-1, cfg.hidden_dim * merge_area});

        Tensor inter;
        TM_SCOPE_CALL(linear_.Forward(merged_input, *weights_.merger_fc1, inter));
        TM_CUDA_CHECK(cudaGetLastError());

        invokeQwen3_5VitBiasActivation(inter, weights_.merger_fc1->bias, ActivationType::kGelu, stream);
        TM_CUDA_CHECK(cudaGetLastError());

        Tensor output;
        TM_SCOPE_CALL(linear_.Forward(inter, *weights_.merger_fc2, output));
        TM_CUDA_CHECK(cudaGetLastError());

        AllReduceSum(output, stream);

        ApplyBias(output, weights_.merger_fc2->bias, stream);
        TM_CUDA_CHECK(cudaGetLastError());

        return output;
    }
};

Qwen3_5Vit::Qwen3_5Vit(const EngineParam& engine, const Context& ctx, const Qwen3_5VitWeight& weights, int phases):
    impl_{std::make_unique<Impl>(engine, ctx, weights, phases)}
{
}

Qwen3_5Vit::~Qwen3_5Vit() = default;

void Qwen3_5Vit::Run(BatchOp op, int phase, TensorMap& env)
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
