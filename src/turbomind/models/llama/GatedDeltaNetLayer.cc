#include "src/turbomind/models/llama/GatedDeltaNetLayer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/engine/block.h"
#include "src/turbomind/kernels/copy/copy.h"
#include "src/turbomind/models/llama/gated_delta_net_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {
namespace {

using linear_attn::delta_rule::ContextParallelLevel;

ContextParallelLevel GetCPLevel()
{
    const char* value = std::getenv("TM_GDR_CP_LEVEL");
    if (value == nullptr || std::strcmp(value, "2") == 0) {
        return ContextParallelLevel::kAll;
    }
    if (std::strcmp(value, "1") == 0) {
        return ContextParallelLevel::kExact;
    }
    if (std::strcmp(value, "0") == 0) {
        return ContextParallelLevel::kOff;
    }
    TM_CHECK(false) << "TM_GDR_CP_LEVEL must be 0 (off), 1 (exact), or 2 (all), got " << value;
    return ContextParallelLevel::kAll;
}

}  // namespace

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
    tp_size_{engine.attn_tp_size * engine.attn_cp_size},
    recurrent_state_dtype_{engine.state_dtype},
    gdr_cp_level_{GetCPLevel()},
    linear_{*context.linear}
{
    TM_CHECK(!weights.empty());
    const auto& first = *TM_CHECK_NOTNULL(weights.front());
    layer_num_        = static_cast<int>(weights.size());

    arch_     = getSMVersion() * 10;
    sm_count_ = getSMCount();

    TM_CHECK_EQ(first.num_k_heads % tp_size_, 0);
    TM_CHECK_EQ(first.num_v_heads % tp_size_, 0);
    TM_CHECK_EQ(first.key_head_dim, 128);
    TM_CHECK_EQ(first.value_head_dim, 128);
    for (const auto* weight_ptr : weights) {
        const auto& weight = *TM_CHECK_NOTNULL(weight_ptr);
        TM_CHECK_EQ(weight.num_k_heads, first.num_k_heads);
        TM_CHECK_EQ(weight.num_v_heads, first.num_v_heads);
        TM_CHECK_EQ(weight.key_head_dim, first.key_head_dim);
        TM_CHECK_EQ(weight.value_head_dim, first.value_head_dim);
        TM_CHECK_EQ(weight.d_conv, first.d_conv);
        TM_CHECK_EQ(weight.data_type, first.data_type);
        TM_CHECK_EQ(weight.num_k_heads % tp_size_, 0);
        TM_CHECK_EQ(weight.num_v_heads % tp_size_, 0);
    }

    input_dtype_ = first.data_type;
    num_k_heads_ = first.num_k_heads / tp_size_;
    num_v_heads_ = first.num_v_heads / tp_size_;
    head_dim_    = first.key_head_dim;
    gate_stride_ = num_v_heads_;
    d_conv_      = first.d_conv;
    conv_dim_    = 2 * num_k_heads_ * head_dim_ + num_v_heads_ * head_dim_;
    TM_CHECK_EQ(num_v_heads_ % num_k_heads_, 0);
    TM_CHECK(recurrent_state_dtype_ == kFloat32 || recurrent_state_dtype_ == input_dtype_)
        << "GDN recurrent state dtype must be float32 or match the input dtype, got state_dtype="
        << recurrent_state_dtype_ << " input_dtype=" << input_dtype_;

    const auto [linear_state_size, conv_state_size] = get_lc_state_size(first, tp_size_);
    const int cell_elements                         = first.key_head_dim * first.value_head_dim;
    TM_CHECK_EQ(linear_state_size, num_v_heads_ * cell_elements);

    int layers_per_block = 1;
    int heads_per_block  = num_v_heads_;
    if (const char* value = std::getenv("TM_GDN_BLOCK_CONFIG")) {
        TM_CHECK_EQ(std::sscanf(value, "%d,%d", &layers_per_block, &heads_per_block), 2)
            << "expected TM_GDN_BLOCK_CONFIG=l,h (e.g. 4,16)";
    }
    TM_CHECK_GT(layers_per_block, 0);
    TM_CHECK_GT(heads_per_block, 0);

    auto ceil_div     = [](int value, int divisor) { return (value + divisor - 1) / divisor; };
    layers_per_block_ = layers_per_block;
    heads_per_block_  = heads_per_block;
    num_head_groups_  = ceil_div(num_v_heads_, heads_per_block_);
    num_layer_groups_ = ceil_div(layer_num_, layers_per_block_);
    num_blocks_       = num_layer_groups_ * num_head_groups_;
    block_bytes_      = byte_size(recurrent_state_dtype_, size_t(layers_per_block_) * heads_per_block_ * cell_elements);

    auto require_mode = [&](linear_attn::delta_rule::GdrMode mode) {
        using namespace linear_attn::delta_rule;
        PlanningContext planning{};
        planning.arch              = arch_;
        planning.sm_count          = sm_count_;
        planning.input_dtype       = input_dtype_;
        planning.state_dtype       = recurrent_state_dtype_;
        planning.physical_batch    = 1;
        planning.token_slots       = mode == GdrMode::kRecurrent ? 1 : 16;
        planning.hq                = num_k_heads_;
        planning.hv                = num_v_heads_;
        planning.head_dim          = head_dim_;
        planning.gate_stride       = gate_stride_;
        planning.gate_batch_stride = int64_t(planning.token_slots) * gate_stride_;
        planning.beta_stride       = planning.gate_stride;
        planning.beta_batch_stride = planning.gate_batch_stride;
        planning.num_head_groups   = num_head_groups_;
        planning.heads_per_block   = heads_per_block_;
        if (mode == GdrMode::kChunked) {
            planning.q_offsets = {0, 16};
        }
        Operation operation{};
        operation.mode     = mode;
        operation.cp_level = gdr_cp_level_;
        Plan plan;
        TM_CHECK(delta_rule_.Plan(operation, planning, &plan));
    };
    require_mode(linear_attn::delta_rule::GdrMode::kRecurrent);
    require_mode(linear_attn::delta_rule::GdrMode::kChunked);

    rec_base_ = registry.checkpoint().Register({{block_bytes_, 1, static_cast<size_t>(num_blocks_)}});

    size_t conv_offset = 0;
    for (int layer = 0; layer < layer_num_; ++layer) {
        weights[layer]->conv_state_offset = conv_offset;
        conv_offset += conv_state_size;
    }
    conv_total_bytes_ = byte_size(input_dtype_, conv_offset);
    registry.checkpoint().Register(conv_total_bytes_, 1);

    const size_t prefix_bytes = registry.prefix().accumulation_bytes();
    TM_LOG_INFO("[GDN] input_dtype={} state_dtype={} gdr_cp_level={} block config L_b={} H_b={} -> "
                "num_layer_groups={} num_head_groups={} num_blocks={} block_bytes={} "
                "prefix_object_bytes={} ({})",
                input_dtype_,
                recurrent_state_dtype_,
                static_cast<int>(gdr_cp_level_),
                layers_per_block_,
                heads_per_block_,
                num_layer_groups_,
                num_head_groups_,
                num_blocks_,
                block_bytes_,
                prefix_bytes,
                (prefix_bytes != 0 && block_bytes_ == prefix_bytes) ? "slab-shared" : "separate-slab-class");

    for (int layer = 0; layer < layer_num_; ++layer) {
        weights[layer]->linear_state_offset = (layer % layers_per_block_) * heads_per_block_ * cell_elements;
        layer_index_[weights[layer]]        = layer;
    }

    conv_state_ptrs_buf_             = {engine.max_batch_size, kCPUpinned};
    recurrent_state_ptrs_buf_        = {core::ssize_t(num_layer_groups_) * engine.max_batch_size * num_head_groups_,
                                 kCPUpinned};
    speculative_request_indices_buf_ = {engine.max_batch_size, kCPUpinned};
    conv_state_offsets_buf_          = {layer_num_, kCPUpinned};
    for (int layer = 0; layer < layer_num_; ++layer) {
        conv_state_offsets_buf_[layer] = static_cast<int>(weights[layer]->conv_state_offset);
    }

    for (int phase = 0; phase < phases; ++phase) {
        data_.emplace_back();
        data_.at(phase).conv_state_ptrs             = empty_like(conv_state_ptrs_buf_, kDEVICE);
        data_.at(phase).recurrent_state_ptrs        = empty_like(recurrent_state_ptrs_buf_, kDEVICE);
        data_.at(phase).state_store_suppressed      = {engine.max_batch_size, kDEVICE};
        data_.at(phase).speculative_request_indices = {engine.max_batch_size, kDEVICE};
        data_.at(phase).conv_state_offsets          = {layer_num_, kDEVICE};
        // Engine-thread kSetup has no pinned allocator. Keep each phase's
        // staging storage alive until its executor-stream copy completes.
        data_.at(phase).commit_state_ptrs_buf = {
            core::ssize_t(layer_num_) * engine.max_batch_size * num_head_groups_, kCPUpinned};
    }

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
    if (op == BatchOp::kSetup) {
        Setup(phase, env);
    }
    else if (op == BatchOp::kPrepare) {
        auto& data                 = data_.at(phase);
        data.q_offsets             = env.at("q_offsets").buffer().borrow();
        data.k_offsets             = env.at("k_offsets").buffer().borrow();
        data.entry_sequence_length = env.at("sequence_length").buffer().borrow();
        // Verification rows in the batch — counted at kSetup from the
        // submitted rows — select the store-suppressed path. The
        // speculative-row flag itself flows to the mask kernel as data.
        if (data.speculative_request_count != 0) {
            data.finished_on_entry      = env.at("finished").buffer().borrow();
            data.speculative_row        = env.at("speculative_row").buffer().borrow();
            data.finished               = data.state_store_suppressed.slice(0, data.batch_size);
            data.build_state_store_mask = true;
        }
        else {
            data.finished               = env.at("finished").buffer().borrow();
            data.finished_on_entry      = {};
            data.speculative_row        = {};
            data.build_state_store_mask = false;
        }

        if (data.speculative_request_count != 0) {
            const int L            = layer_num_;
            const int Sspec        = data.speculative_request_count;
            const int K            = data.verify_positions;
            data.journal.raw_conv  = {{L, Sspec, K, conv_dim_}, input_dtype_, kDEVICE};
            data.journal.key       = {{L, Sspec, K, num_k_heads_, 128}, input_dtype_, kDEVICE};
            data.journal.value     = {{L, Sspec, K, num_v_heads_, 128}, input_dtype_, kDEVICE};
            data.journal.log_decay = {{L, Sspec, K, num_v_heads_}, kFloat32, kDEVICE};
            data.journal.beta      = {{L, Sspec, K, num_v_heads_}, kFloat32, kDEVICE};
        }
        for (const auto& [ptr, bytes] : data.reset_ptrs) {
            Clear(Buffer_<uint8_t>{ptr, static_cast<core::ssize_t>(bytes), kDEVICE});
        }
        data.reset_ptrs.clear();

        if (data.commit_plan) {
            const core::ssize_t count = core::ssize_t(layer_num_) * data.speculative_request_count;
            auto host_ptrs = data.commit_state_ptrs_buf.slice(0, count * num_head_groups_);
            data.commit_state_ptrs = Tensor{empty_like(host_ptrs, kDEVICE),
                                            core::Layout{{count, num_head_groups_}}};
            data.commit_state_tma_descs = {{count, num_head_groups_, 128}, kUint8, kDEVICE};
            data.commit_lengths         = {{count}, kInt32, kDEVICE};
            Copy(host_ptrs, data.commit_state_ptrs.buffer());

            // Each flattened request points at one layer slice of its cache part.
            auto state_ptrs  = data.commit_state_ptrs.view({1, count, num_head_groups_});
            auto state_descs = data.commit_state_tma_descs.view({1, count, num_head_groups_, 128});
            delta_rule_.PrepareState(state_ptrs,
                                     state_descs,
                                     1,
                                     1,
                                     *data.commit_plan,
                                     core::Context::stream().handle());
        }

        if (data.verify_plan) {
            core::Tensor state_ptrs{data.recurrent_state_ptrs,
                                    core::Layout{{num_layer_groups_, data.verify_count, num_head_groups_},
                                                 {data.batch_size * num_head_groups_, num_head_groups_, 1}},
                                    core::Tensor::PreserveBufferCapacity{}};
            core::Tensor state_descs{data.verify_state_tma_descs,
                                     core::Layout{{num_layer_groups_, data.verify_count, num_head_groups_, 128}}};
            delta_rule_.PrepareState(state_ptrs,
                                     state_descs,
                                     num_layer_groups_,
                                     layers_per_block_,
                                     *data.verify_plan,
                                     core::Context::stream().handle());
        }
        if (data.recurrent_plan) {
            const core::ssize_t base = core::ssize_t(data.verify_count) * num_head_groups_;
            const auto          tail = data.recurrent_state_ptrs.slice(base, data.recurrent_state_ptrs.size() - base);
            core::Tensor        state_ptrs{tail,
                                    core::Layout{{num_layer_groups_, data.decode_count, num_head_groups_},
                                                 {data.batch_size * num_head_groups_, num_head_groups_, 1}},
                                    core::Tensor::PreserveBufferCapacity{}};
            core::Tensor        state_descs;
            if (data.recurrent_state_tma_descs) {
                state_descs = core::Tensor{data.recurrent_state_tma_descs,
                                           core::Layout{{num_layer_groups_, data.decode_count, num_head_groups_, 128}}};
            }
            delta_rule_.PrepareState(state_ptrs,
                                     state_descs,
                                     num_layer_groups_,
                                     layers_per_block_,
                                     *data.recurrent_plan,
                                     core::Context::stream().handle());
        }
    }
}

void GatedDeltaNetLayer::Setup(int phase, TensorMap& env)
{
    auto&              data     = data_.at(phase);
    Buffer_<Sequence*> requests = env.at("requests").buffer();

    data.batch_size = requests.size();
    data.input_lens.resize(data.batch_size);
    data.reset_ptrs.clear();
    data.speculative_request_count = 0;
    data.verify_positions    = env.at("verification_positions").data<int>()[0];

    std::vector<int32_t> host_offsets(data.batch_size + 1, 0);
    for (int sequence = 0; sequence < data.batch_size; ++sequence) {
        const Sequence&     request = *requests[sequence];
        const SubmittedRow& row     = *request.submitted;
        data.input_lens[sequence]   = row.input_len;
        if (row.is_verification_row()) {
            speculative_request_indices_buf_[data.speculative_request_count++] = sequence;
        }
        host_offsets[sequence + 1] = host_offsets[sequence] + data.input_lens[sequence];
    }
    const int token_slots   = *env.at("token_num").data<int>();
    data.verify_count = 0;
    data.decode_count       = 0;
    data.prefill_count      = 0;
    data.verify_plan.reset();
    data.commit_plan.reset();
    data.recurrent_plan.reset();
    data.chunked_plan.reset();
    data.chunked_workspace            = {};
    data.verify_state_tma_descs = {};
    data.recurrent_state_tma_descs    = {};

    auto make_context = [&] {
        linear_attn::delta_rule::PlanningContext planning{};
        planning.arch            = arch_;
        planning.sm_count        = sm_count_;
        planning.input_dtype     = input_dtype_;
        planning.state_dtype     = recurrent_state_dtype_;
        planning.hq              = num_k_heads_;
        planning.hv              = num_v_heads_;
        planning.head_dim        = head_dim_;
        planning.gate_stride     = gate_stride_;
        planning.beta_stride     = gate_stride_;
        planning.num_head_groups = num_head_groups_;
        planning.heads_per_block = heads_per_block_;
        return planning;
    };

    const bool verify_candidate = arch_ == 900 && input_dtype_ == kBfloat16 && head_dim_ == 128
                                        && data.speculative_request_count != 0 && data.verify_positions <= 16;
    if (verify_candidate) {
        auto planning              = make_context();
        planning.physical_batch    = data.speculative_request_count;
        planning.token_slots       = data.verify_positions;
        planning.gate_batch_stride = int64_t(data.verify_positions) * gate_stride_;
        planning.beta_batch_stride = planning.gate_batch_stride;
        linear_attn::delta_rule::Operation operation{};
        operation.mode       = linear_attn::delta_rule::GdrMode::kVerify;
        operation.chunk_size = data.verify_positions <= 8 ? 8 : 16;
        operation.cp_level   = ContextParallelLevel::kOff;
        linear_attn::delta_rule::Plan plan;
        if (delta_rule_.Plan(operation, planning, &plan)) {
            data.verify_plan.emplace(std::move(plan));
            data.verify_count = data.speculative_request_count;
        }
    }

    const bool commit_candidate = arch_ == 900 && input_dtype_ == kBfloat16 && head_dim_ == 128
                                  && data.speculative_request_count != 0
                                  && data.verify_positions >= 1 && data.verify_positions <= 16;
    if (commit_candidate) {
        auto planning              = make_context();
        planning.physical_batch    = layer_num_ * data.speculative_request_count;
        planning.token_slots       = data.verify_positions;
        planning.gate_stride       = num_v_heads_;
        planning.beta_stride       = num_v_heads_;
        planning.gate_batch_stride = int64_t(data.verify_positions) * num_v_heads_;
        planning.beta_batch_stride = planning.gate_batch_stride;
        linear_attn::delta_rule::Operation operation{};
        operation.mode       = linear_attn::delta_rule::GdrMode::kCommit;
        operation.chunk_size = data.verify_positions <= 8 ? 8 : 16;
        operation.cp_level   = ContextParallelLevel::kOff;
        linear_attn::delta_rule::Plan plan;
        if (delta_rule_.Plan(operation, planning, &plan)) {
            data.commit_plan.emplace(std::move(plan));
        }
    }

    while (data.verify_count + data.decode_count < data.batch_size
           && data.input_lens[data.verify_count + data.decode_count] == 1) {
        ++data.decode_count;
    }
    data.prefill_count = data.batch_size - data.verify_count - data.decode_count;

    if (data.decode_count != 0) {
        auto planning              = make_context();
        planning.physical_batch    = data.decode_count;
        planning.token_slots       = 1;
        planning.gate_batch_stride = gate_stride_;
        planning.beta_batch_stride = gate_stride_;
        linear_attn::delta_rule::Operation operation{};
        operation.mode = linear_attn::delta_rule::GdrMode::kRecurrent;
        linear_attn::delta_rule::Plan plan;
        TM_CHECK(delta_rule_.Plan(operation, planning, &plan));
        data.recurrent_plan.emplace(std::move(plan));
    }

    if (data.prefill_count != 0) {
        auto planning              = make_context();
        planning.physical_batch    = 1;
        planning.token_slots       = token_slots;
        planning.gate_batch_stride = int64_t(token_slots) * gate_stride_;
        planning.beta_batch_stride = planning.gate_batch_stride;
        const int first_prefill    = data.verify_count + data.decode_count;
        planning.q_offsets.assign(host_offsets.begin() + first_prefill, host_offsets.end());
        linear_attn::delta_rule::Operation operation{};
        operation.mode     = linear_attn::delta_rule::GdrMode::kChunked;
        operation.cp_level = gdr_cp_level_;
        linear_attn::delta_rule::Plan plan;
        TM_CHECK(delta_rule_.Plan(operation, planning, &plan));
        data.chunked_plan.emplace(std::move(plan));
    }

    if (data.chunked_plan && data.chunked_plan->workspace_bytes != 0) {
        data.chunked_workspace = core::Tensor{
            core::Layout{{static_cast<core::ssize_t>(data.chunked_plan->workspace_bytes)}}, kUint8, kDEVICE};
    }
    if (data.verify_plan && data.verify_plan->state_tma_desc_bytes_per_layer_group != 0) {
        const core::ssize_t descriptor_bytes =
            core::ssize_t(num_layer_groups_) * data.verify_plan->state_tma_desc_bytes_per_layer_group;
        data.verify_state_tma_descs = {descriptor_bytes, kDEVICE};
    }
    if (data.recurrent_plan && data.recurrent_plan->state_tma_desc_bytes_per_layer_group != 0) {
        const core::ssize_t descriptor_bytes =
            core::ssize_t(num_layer_groups_) * data.recurrent_plan->state_tma_desc_bytes_per_layer_group;
        data.recurrent_state_tma_descs = {descriptor_bytes, kDEVICE};
    }

    for (int sequence = 0; sequence < data.batch_size; ++sequence) {
        auto&               request = *requests[sequence];
        const SubmittedRow& row     = *request.submitted;

        const CacheBlock& block = *TM_CHECK_NOTNULL(request.frontier.get());
        TM_CHECK_NOTNULL(block.allocation.a);

        conv_state_ptrs_buf_[sequence] = block.base(0);
        for (int layer_group = 0; layer_group < num_layer_groups_; ++layer_group) {
            for (int head_group = 0; head_group < num_head_groups_; ++head_group) {
                const int part = rec_base_ + layer_group * num_head_groups_ + head_group;
                recurrent_state_ptrs_buf_[(layer_group * data.batch_size + sequence) * num_head_groups_ + head_group] =
                    block.base(part);
            }
        }

        if (row.history_len + request.inflight_input_len == 0) {
            data.reset_ptrs.push_back({reinterpret_cast<uint8_t*>(block.base(0)), conv_total_bytes_});
            for (int recurrent_block = 0; recurrent_block < num_blocks_; ++recurrent_block) {
                data.reset_ptrs.push_back(
                    {reinterpret_cast<uint8_t*>(block.base(rec_base_ + recurrent_block)), block_bytes_});
            }
        }
    }

    if (data.commit_plan) {
        const int requests = data.speculative_request_count;
        // contracts.scheduler-output puts speculative rows first in executor order.
        for (int request = 0; request < requests; ++request) {
            TM_CHECK_EQ(speculative_request_indices_buf_[request], request);
        }
        for (int layer = 0; layer < layer_num_; ++layer) {
            const int layer_group = layer / layers_per_block_;
            const size_t layer_bytes = byte_size(recurrent_state_dtype_,
                size_t(layer % layers_per_block_) * heads_per_block_ * head_dim_ * head_dim_);
            for (int request = 0; request < requests; ++request) {
                for (int head_group = 0; head_group < num_head_groups_; ++head_group) {
                    const core::ssize_t source =
                        (core::ssize_t(layer_group) * data.batch_size + request) * num_head_groups_ + head_group;
                    const core::ssize_t destination =
                        (core::ssize_t(layer) * requests + request) * num_head_groups_ + head_group;
                    data.commit_state_ptrs_buf[destination] =
                        static_cast<uint8_t*>(recurrent_state_ptrs_buf_[source]) + layer_bytes;
                }
            }
        }
    }

    Copy(conv_state_ptrs_buf_, data.batch_size, data.conv_state_ptrs);
    Copy(recurrent_state_ptrs_buf_,
         core::ssize_t(num_layer_groups_) * data.batch_size * num_head_groups_,
         data.recurrent_state_ptrs);

    auto& copy = *env.at("copy").data<core::BatchCopy*>()[0];
    copy(speculative_request_indices_buf_, data.speculative_request_count, data.speculative_request_indices);
    copy(conv_state_offsets_buf_, layer_num_, data.conv_state_offsets);
}

void GatedDeltaNetLayer::Forward(ForwardParam param)
{
    TM_FUNCTION_SCOPE();

    const int token_num = param.input.shape(0);
    if (token_num == 0) {
        return;
    }

    const auto  dtype      = param.input.dtype();
    const auto  device     = param.input.device();
    const auto  stream     = core::Context::stream().handle();
    const auto& weights    = *param.weights;
    auto&       phase_data = data_.at(param.phase);
    const int   layer      = layer_index_.at(param.weights);

    if (phase_data.build_state_store_mask && layer == 0) {
        linear_attn::delta_rule::invokeBuildGdnStateStoreMask(phase_data.state_store_suppressed.data(),
                                                              phase_data.finished_on_entry.data(),
                                                              phase_data.speculative_row.data(),
                                                              phase_data.batch_size,
                                                              stream);
    }

    TM_CHECK(dtype == kHalf || dtype == kBfloat16);

    const int key_dim   = num_k_heads_ * head_dim_;
    const int value_dim = num_v_heads_ * head_dim_;
    const int conv_dim  = key_dim * 2 + value_dim;

    Tensor all_proj = MakePaddedOutput(token_num, *weights.in_proj_all, device);
    TM_SCOPE_CALL(linear_.Forward(param.input, *weights.in_proj_all, all_proj));

    const int value_heads       = num_v_heads_;
    const int value_gate_offset = conv_dim + value_dim;
    const int decay_gate_offset = value_gate_offset + value_heads;

    const core::ssize_t gate_capacity = core::ssize_t(token_num) * gate_stride_;
    const core::Layout  gate_layout{{1, token_num, num_v_heads_}, {gate_capacity, gate_stride_, 1}};
    Tensor beta{core::Buffer{gate_capacity, kFloat32, device}, gate_layout, Tensor::PreserveBufferCapacity{}};
    Tensor g{core::Buffer{gate_capacity, kFloat32, device}, gate_layout, Tensor::PreserveBufferCapacity{}};

    Tensor beta_projection  = all_proj.slice({0, value_gate_offset}, {-1, value_heads});
    Tensor decay_projection = all_proj.slice({0, decay_gate_offset}, {-1, value_heads});
    ComputeBetaG(beta, g, beta_projection, decay_projection, weights.A_log, weights.dt_bias, stream);

    Tensor attn_out{{token_num, value_dim}, dtype, device};
    Tensor conv_out{{token_num, conv_dim}, dtype, device};

    invokeFusedConv1dSiLU(conv_out,
                          all_proj,
                          weights.conv1d,
                          Tensor{},
                          phase_data.conv_state_ptrs,
                          phase_data.q_offsets,
                          phase_data.k_offsets,
                          phase_data.finished,
                          phase_data.batch_size,
                          weights.conv_state_offset,
                          sm_count_,
                          work_counter_.data(),
                          stream);

    auto make_view = [](const Tensor& storage, core::ssize_t offset, core::Layout layout) {
        return Tensor{storage.buffer().slice(offset, storage.buffer().size() - offset),
                      std::move(layout),
                      Tensor::PreserveBufferCapacity{}};
    };

    const core::Layout qk_layout{{1, token_num, num_k_heads_, 128}, {int64_t(token_num) * conv_dim, conv_dim, 128, 1}};
    const core::Layout v_layout{{1, token_num, num_v_heads_, 128}, {int64_t(token_num) * conv_dim, conv_dim, 128, 1}};
    const core::Layout out_layout{{1, token_num, num_v_heads_, 128},
                                  {int64_t(token_num) * value_dim, value_dim, 128, 1}};
    Tensor             q = make_view(conv_out, 0, qk_layout);
    Tensor             k = make_view(conv_out, key_dim, qk_layout);
    Tensor             v = make_view(conv_out, 2 * key_dim, v_layout);
    Tensor             out{attn_out.buffer(), out_layout};
    invokeL2NormalizeQK(q, k, 1e-6f, stream);

    if (phase_data.speculative_request_count != 0) {
        linear_attn::delta_rule::invokeCaptureGdnTransitions(
            all_proj.slice({0, 0}, {-1, conv_dim}),
            k,
            v,
            g,
            beta,
            phase_data.q_offsets,
            phase_data.speculative_request_indices.slice(0, phase_data.speculative_request_count),
            layer,
            phase_data.verify_positions,
            phase_data.journal,
            stream);
    }

    const int     layer_group        = layer / layers_per_block_;
    const int64_t state_layer_offset = weights.linear_state_offset;

    auto pointer_view = [&](int first_sequence, int sequence_count) {
        const core::ssize_t offset =
            (core::ssize_t(layer_group) * phase_data.batch_size + first_sequence) * num_head_groups_;
        const core::ssize_t count = core::ssize_t(sequence_count) * num_head_groups_;
        return Tensor{phase_data.recurrent_state_ptrs.slice(offset, count),
                      core::Layout{{sequence_count, num_head_groups_}}};
    };

    const int S = phase_data.verify_count;
    const int K = phase_data.verify_positions;

    linear_attn::delta_rule::Arguments verify_args{};
    Tensor                             verify_out;
    if (phase_data.verify_plan) {
        const core::Layout verify_qk_layout{{S, K, num_k_heads_, 128}, {int64_t(K) * conv_dim, conv_dim, 128, 1}};
        const core::Layout verify_v_layout{{S, K, num_v_heads_, 128}, {int64_t(K) * conv_dim, conv_dim, 128, 1}};
        const core::Layout verify_gate_layout{{S, K, num_v_heads_}, {int64_t(K) * gate_stride_, gate_stride_, 1}};
        const core::Layout verify_out_layout{{S, K, num_v_heads_, 128}, {int64_t(K) * value_dim, value_dim, 128, 1}};
        verify_out             = Tensor{out.buffer(), verify_out_layout, Tensor::PreserveBufferCapacity{}};
        verify_args.q          = Tensor{q.buffer(), verify_qk_layout, Tensor::PreserveBufferCapacity{}};
        verify_args.k          = Tensor{k.buffer(), verify_qk_layout, Tensor::PreserveBufferCapacity{}};
        verify_args.v          = Tensor{v.buffer(), verify_v_layout, Tensor::PreserveBufferCapacity{}};
        verify_args.g          = Tensor{g.buffer(), verify_gate_layout, Tensor::PreserveBufferCapacity{}};
        verify_args.beta       = Tensor{beta.buffer(), verify_gate_layout, Tensor::PreserveBufferCapacity{}};
        verify_args.state_ptrs = pointer_view(0, S);
        const core::ssize_t descriptor_count  = core::ssize_t(S) * num_head_groups_ * 128;
        const core::ssize_t descriptor_offset = core::ssize_t(layer_group) * descriptor_count;
        verify_args.state_tma_descs =
            Tensor{phase_data.verify_state_tma_descs.slice(descriptor_offset, descriptor_count),
                   core::Layout{{S, num_head_groups_, 128}}};
        verify_args.out                = &verify_out;
        verify_args.state_layer_offset = state_layer_offset;
    }

    linear_attn::delta_rule::Arguments recurrent_args{};
    Tensor                             recurrent_out;
    if (phase_data.recurrent_plan) {
        const int ordinary_token_begin = S * K;
        auto      token_tail           = [ordinary_token_begin](const Tensor& tensor) {
            const core::ssize_t first = core::ssize_t(ordinary_token_begin) * tensor.stride(1);
            return tensor.buffer().slice(first, tensor.buffer().size() - first);
        };
        const core::Layout recurrent_qk_layout{{phase_data.decode_count, 1, num_k_heads_, 128},
                                               {conv_dim, conv_dim, 128, 1}};
        const core::Layout recurrent_v_layout{{phase_data.decode_count, 1, num_v_heads_, 128},
                                              {conv_dim, conv_dim, 128, 1}};
        const core::Layout recurrent_gate_layout{{phase_data.decode_count, 1, num_v_heads_},
                                                 {gate_stride_, gate_stride_, 1}};
        const core::Layout recurrent_out_layout{{phase_data.decode_count, 1, num_v_heads_, 128},
                                                {value_dim, value_dim, 128, 1}};
        recurrent_out             = Tensor{token_tail(out), recurrent_out_layout, Tensor::PreserveBufferCapacity{}};
        recurrent_args.q          = Tensor{token_tail(q), recurrent_qk_layout, Tensor::PreserveBufferCapacity{}};
        recurrent_args.k          = Tensor{token_tail(k), recurrent_qk_layout, Tensor::PreserveBufferCapacity{}};
        recurrent_args.v          = Tensor{token_tail(v), recurrent_v_layout, Tensor::PreserveBufferCapacity{}};
        recurrent_args.g          = Tensor{token_tail(g), recurrent_gate_layout, Tensor::PreserveBufferCapacity{}};
        recurrent_args.beta       = Tensor{token_tail(beta), recurrent_gate_layout, Tensor::PreserveBufferCapacity{}};
        recurrent_args.state_ptrs = pointer_view(S, phase_data.decode_count);
        if (phase_data.recurrent_state_tma_descs) {
            const core::ssize_t descriptor_count  = core::ssize_t(phase_data.decode_count) * num_head_groups_ * 128;
            const core::ssize_t descriptor_offset = core::ssize_t(layer_group) * descriptor_count;
            recurrent_args.state_tma_descs =
                Tensor{phase_data.recurrent_state_tma_descs.slice(descriptor_offset, descriptor_count),
                       core::Layout{{phase_data.decode_count, num_head_groups_, 128}}};
        }
        recurrent_args.finished =
            Tensor{phase_data.finished.slice(S, phase_data.decode_count), core::Layout{{phase_data.decode_count}}};
        recurrent_args.out                = &recurrent_out;
        recurrent_args.state_layer_offset = state_layer_offset;
    }

    const bool has_main     = phase_data.verify_plan.has_value() || phase_data.recurrent_plan.has_value();
    const bool fork_chunked = has_main && phase_data.chunked_plan.has_value();
    if (fork_chunked) {
        TM_CUDA_CHECK(cudaEventRecord(ev_before_, stream));
        TM_CUDA_CHECK(cudaStreamWaitEvent(aux_stream_, ev_before_));
    }

    if (phase_data.verify_plan) {
        delta_rule_.Run(verify_args, *phase_data.verify_plan, stream);
    }
    if (phase_data.recurrent_plan) {
        delta_rule_.Run(recurrent_args, *phase_data.recurrent_plan, stream);
    }

    if (phase_data.chunked_plan) {
        const int first_prefill    = S + phase_data.decode_count;
        Tensor    chunk_state_ptrs = pointer_view(first_prefill, phase_data.prefill_count);
        Tensor    chunk_finished{phase_data.finished.slice(first_prefill, phase_data.prefill_count),
                              core::Layout{{phase_data.prefill_count}}};
        Tensor    chunk_q_offsets{phase_data.q_offsets.slice(first_prefill, phase_data.prefill_count + 1),
                               core::Layout{{phase_data.prefill_count + 1}}};

        linear_attn::delta_rule::Arguments arguments{};
        arguments.q                  = q;
        arguments.k                  = k;
        arguments.v                  = v;
        arguments.g                  = g;
        arguments.beta               = beta;
        arguments.state_ptrs         = chunk_state_ptrs;
        arguments.q_offsets          = chunk_q_offsets;
        arguments.finished           = chunk_finished;
        arguments.out                = &out;
        arguments.workspace          = phase_data.chunked_workspace ? &phase_data.chunked_workspace : nullptr;
        arguments.state_layer_offset = state_layer_offset;
        delta_rule_.Run(arguments, *phase_data.chunked_plan, fork_chunked ? aux_stream_ : stream);
    }

    if (fork_chunked) {
        TM_CUDA_CHECK(cudaEventRecord(ev_after_, aux_stream_));
        TM_CUDA_CHECK(cudaStreamWaitEvent(stream, ev_after_));
    }

    Tensor gate        = all_proj.slice({0, conv_dim}, {-1, value_dim});
    Tensor hidden_view = attn_out.view({token_num * num_v_heads_, head_dim_});
    invokeRMSNormGated(hidden_view, gate, weights.norm->weight, weights.norm->norm_eps_, stream);

    TM_SCOPE_CALL(linear_.Forward(attn_out, *weights.out_proj, param.output));
}

void GatedDeltaNetLayer::CommitAcceptedState(int phase, const Buffer_<int>& accept_len)
{
    Data& data = data_.at(phase);
    if (data.speculative_request_count == 0) {
        return;
    }

    const cudaStream_t stream          = core::Context::stream().handle();
    const auto         request_indices = data.speculative_request_indices.slice(0, data.speculative_request_count);

    linear_attn::delta_rule::invokeCommitAcceptedConvState(data.journal.raw_conv,
                                                           data.conv_state_ptrs,
                                                           request_indices,
                                                           data.entry_sequence_length,
                                                           accept_len,
                                                           data.conv_state_offsets,
                                                           conv_dim_,
                                                           d_conv_,
                                                           stream);

    if (data.commit_plan) {
        const int requests       = data.speculative_request_count;
        const int positions      = data.verify_positions;
        const core::ssize_t count = core::ssize_t(layer_num_) * requests;
        Tensor accepted_lengths{accept_len.slice(0, requests),
                                 core::Layout{{layer_num_, requests}, {0, 1}}};
        auto commit_lengths = data.commit_lengths.view({layer_num_, requests});
        core::GenericCopy(accepted_lengths, commit_lengths, stream);

        linear_attn::delta_rule::Arguments args{};
        args.k               = data.journal.key.view({count, positions, num_k_heads_, 128});
        args.v               = data.journal.value.view({count, positions, num_v_heads_, 128});
        args.g               = data.journal.log_decay.view({count, positions, num_v_heads_});
        args.beta            = data.journal.beta.view({count, positions, num_v_heads_});
        args.state_ptrs      = data.commit_state_ptrs;
        args.state_tma_descs = data.commit_state_tma_descs;
        args.commit_lengths  = data.commit_lengths;
        // Final accept_len is already terminal-clamped. Newly finished rows
        // still commit their accepted prefix; no forward suppression mask applies.
        // Adjusted pointer bases make every logical request a single-layer state.
        args.state_layer_offset = 0;
        delta_rule_.Run(args, *data.commit_plan, stream);
    }
    else {
        Tensor recurrent_state_ptrs{data.recurrent_state_ptrs,
                                    core::Layout{{num_layer_groups_, data.batch_size, num_head_groups_},
                                                 {data.batch_size * num_head_groups_, num_head_groups_, 1}},
                                    Tensor::PreserveBufferCapacity{}};

        linear_attn::delta_rule::AcceptedPrefixArguments args{};
        args.key                  = data.journal.key;
        args.value                = data.journal.value;
        args.log_decay            = data.journal.log_decay;
        args.beta                 = data.journal.beta;
        args.recurrent_state_ptrs = recurrent_state_ptrs;
        args.request_indices      = Tensor{request_indices, core::Layout{{data.speculative_request_count}}};
        args.accept_len           = Tensor{accept_len, core::Layout{{data.batch_size}}};
        args.layer_count          = layer_num_;
        args.speculative_count    = data.speculative_request_count;
        args.position_count       = data.verify_positions;
        args.hq                   = num_k_heads_;
        args.hv                   = num_v_heads_;
        args.num_head_groups      = num_head_groups_;
        args.layers_per_block     = layers_per_block_;
        args.heads_per_block      = heads_per_block_;
        args.sm_count             = sm_count_;
        delta_rule_.CommitAccepted(args, recurrent_state_dtype_, stream);
    }

    // These buffers were allocated during kPrepare on the executor stream.
    // Release them with the journal after their last queued consumers.
    data.journal                = {};
    data.commit_state_ptrs      = {};
    data.commit_state_tma_descs = {};
    data.commit_lengths         = {};
}

size_t GatedDeltaNetLayer::SpeculativeStateJournalBytes(int request_count, int verify_positions) const
{
    const size_t rows           = size_t(layer_num_) * request_count * verify_positions;
    const size_t input_elements = size_t(conv_dim_) + size_t(num_k_heads_ + num_v_heads_) * 128;
    const size_t gate_elements  = size_t(2) * num_v_heads_;
    size_t bytes = rows * (byte_size(input_dtype_, input_elements) + gate_elements * sizeof(float));
    if (arch_ == 900 && input_dtype_ == kBfloat16 && head_dim_ == 128
        && verify_positions >= 1 && verify_positions <= 16) {
        // Commit metadata has the same executor-stream lifetime as the journal.
        const size_t commit_rows = size_t(layer_num_) * request_count;
        bytes += commit_rows * (num_head_groups_ * (sizeof(void*) + 128) + sizeof(int));
    }
    return bytes;
}

}  // namespace turbomind
