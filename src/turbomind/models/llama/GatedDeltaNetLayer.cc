#include "src/turbomind/models/llama/GatedDeltaNetLayer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/engine/block.h"
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

    conv_state_ptrs_buf_      = {engine.max_batch_size, kCPUpinned};
    recurrent_state_ptrs_buf_ = {core::ssize_t(num_layer_groups_) * engine.max_batch_size * num_head_groups_,
                                 kCPUpinned};

    for (int phase = 0; phase < phases; ++phase) {
        data_.emplace_back();
        data_.at(phase).conv_state_ptrs      = empty_like(conv_state_ptrs_buf_, kDEVICE);
        data_.at(phase).recurrent_state_ptrs = empty_like(recurrent_state_ptrs_buf_, kDEVICE);
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
    if (op == BatchOp::kAdd) {
        Buffer_<Sequence*> requests = env.at("requests").buffer();
        for (int i = 0; i < requests.size(); ++i) {}
    }
    else if (op == BatchOp::kSetup) {
        Setup(phase, env);
    }
    else if (op == BatchOp::kPrepare) {
        auto& data     = data_.at(phase);
        data.q_offsets = env.at("q_offsets").buffer().borrow();
        data.k_offsets = env.at("k_offsets").buffer().borrow();
        data.finished  = env.at("finished").buffer().borrow();
        for (const auto& [ptr, bytes] : data.reset_ptrs) {
            Clear(Buffer_<uint8_t>{ptr, static_cast<core::ssize_t>(bytes), kDEVICE});
        }
        data.reset_ptrs.clear();

        if (data.recurrent_plan) {
            core::Tensor state_ptrs{data.recurrent_state_ptrs,
                                    core::Layout{{num_layer_groups_, data.decode_count, num_head_groups_},
                                                 {data.batch_size * num_head_groups_, num_head_groups_, 1}},
                                    core::Tensor::PreserveBufferCapacity{}};
            core::Tensor state_descs;
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

    std::vector<int32_t> host_offsets(data.batch_size + 1, 0);
    for (int sequence = 0; sequence < data.batch_size; ++sequence) {
        data.input_lens[sequence]  = requests[sequence]->input_len;
        host_offsets[sequence + 1] = host_offsets[sequence] + data.input_lens[sequence];
    }
    const int token_slots = *env.at("token_num").data<int>();
    data.decode_count     = 0;
    while (data.decode_count < data.batch_size && data.input_lens[data.decode_count] == 1) {
        ++data.decode_count;
    }
    data.prefill_count = data.batch_size - data.decode_count;

    data.recurrent_plan.reset();
    data.chunked_plan.reset();
    data.chunked_workspace         = {};
    data.recurrent_state_tma_descs = {};

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
        planning.q_offsets.assign(host_offsets.begin() + data.decode_count, host_offsets.end());
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
    if (data.recurrent_plan && data.recurrent_plan->state_tma_desc_bytes_per_layer_group != 0) {
        const core::ssize_t descriptor_bytes =
            core::ssize_t(num_layer_groups_) * data.recurrent_plan->state_tma_desc_bytes_per_layer_group;
        data.recurrent_state_tma_descs = {descriptor_bytes, kDEVICE};
    }

    for (int sequence = 0; sequence < data.batch_size; ++sequence) {
        auto& request = *requests[sequence];

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

        if (request.history_len + request.inflight_input_len == 0) {
            data.reset_ptrs.push_back({reinterpret_cast<uint8_t*>(block.base(0)), conv_total_bytes_});
            for (int recurrent_block = 0; recurrent_block < num_blocks_; ++recurrent_block) {
                data.reset_ptrs.push_back(
                    {reinterpret_cast<uint8_t*>(block.base(rec_base_ + recurrent_block)), block_bytes_});
            }
        }
    }

    Copy(conv_state_ptrs_buf_, data.batch_size, data.conv_state_ptrs);
    Copy(recurrent_state_ptrs_buf_,
         core::ssize_t(num_layer_groups_) * data.batch_size * num_head_groups_,
         data.recurrent_state_ptrs);
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

    TM_CHECK(dtype == kHalf || dtype == kBfloat16);

    const int key_dim   = num_k_heads_ * head_dim_;
    const int value_dim = num_v_heads_ * head_dim_;
    const int conv_dim  = key_dim * 2 + value_dim;

    Tensor all_proj;
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

    const int     layer              = layer_index_.at(param.weights);
    const int     layer_group        = layer / layers_per_block_;
    const int64_t state_layer_offset = weights.linear_state_offset;

    auto pointer_view = [&](int first_sequence, int sequence_count) {
        const core::ssize_t offset =
            (core::ssize_t(layer_group) * phase_data.batch_size + first_sequence) * num_head_groups_;
        const core::ssize_t count = core::ssize_t(sequence_count) * num_head_groups_;
        return Tensor{phase_data.recurrent_state_ptrs.slice(offset, count),
                      core::Layout{{sequence_count, num_head_groups_}}};
    };

    const bool mixed = phase_data.recurrent_plan.has_value() && phase_data.chunked_plan.has_value();
    if (mixed) {
        TM_CUDA_CHECK(cudaEventRecord(ev_before_, stream));
        TM_CUDA_CHECK(cudaStreamWaitEvent(aux_stream_, ev_before_));
    }
    const cudaStream_t chunk_stream = mixed ? aux_stream_ : stream;

    if (phase_data.recurrent_plan) {
        const core::Layout recurrent_qk_layout{{phase_data.decode_count, 1, num_k_heads_, 128},
                                               {conv_dim, conv_dim, 128, 1}};
        const core::Layout recurrent_v_layout{{phase_data.decode_count, 1, num_v_heads_, 128},
                                              {conv_dim, conv_dim, 128, 1}};
        const core::Layout recurrent_out_layout{{phase_data.decode_count, 1, num_v_heads_, 128},
                                                {value_dim, value_dim, 128, 1}};
        const core::Layout recurrent_gate_layout{{phase_data.decode_count, 1, num_v_heads_},
                                                 {gate_stride_, gate_stride_, 1}};
        Tensor             recurrent_q{q.buffer(), recurrent_qk_layout, Tensor::PreserveBufferCapacity{}};
        Tensor             recurrent_k{k.buffer(), recurrent_qk_layout, Tensor::PreserveBufferCapacity{}};
        Tensor             recurrent_v{v.buffer(), recurrent_v_layout, Tensor::PreserveBufferCapacity{}};
        Tensor             recurrent_out{out.buffer(), recurrent_out_layout, Tensor::PreserveBufferCapacity{}};
        Tensor             recurrent_g{g.buffer(), recurrent_gate_layout, Tensor::PreserveBufferCapacity{}};
        Tensor             recurrent_beta{beta.buffer(), recurrent_gate_layout, Tensor::PreserveBufferCapacity{}};
        Tensor             recurrent_state_ptrs = pointer_view(0, phase_data.decode_count);
        Tensor             recurrent_finished{phase_data.finished.slice(0, phase_data.decode_count),
                                  core::Layout{{phase_data.decode_count}}};
        Tensor             recurrent_state_descs;
        if (phase_data.recurrent_state_tma_descs) {
            const core::ssize_t descriptor_count  = core::ssize_t(phase_data.decode_count) * num_head_groups_ * 128;
            const core::ssize_t descriptor_offset = core::ssize_t(layer_group) * descriptor_count;
            recurrent_state_descs =
                Tensor{phase_data.recurrent_state_tma_descs.slice(descriptor_offset, descriptor_count),
                       core::Layout{{phase_data.decode_count, num_head_groups_, 128}}};
        }

        linear_attn::delta_rule::Arguments arguments{};
        arguments.q                  = recurrent_q;
        arguments.k                  = recurrent_k;
        arguments.v                  = recurrent_v;
        arguments.g                  = recurrent_g;
        arguments.beta               = recurrent_beta;
        arguments.state_ptrs         = recurrent_state_ptrs;
        arguments.state_tma_descs    = recurrent_state_descs;
        arguments.finished           = recurrent_finished;
        arguments.out                = &recurrent_out;
        arguments.state_layer_offset = state_layer_offset;
        delta_rule_.Run(arguments, *phase_data.recurrent_plan, stream);
    }

    if (phase_data.chunked_plan) {
        Tensor chunk_state_ptrs = pointer_view(phase_data.decode_count, phase_data.prefill_count);
        Tensor chunk_finished{phase_data.finished.slice(phase_data.decode_count, phase_data.prefill_count),
                              core::Layout{{phase_data.prefill_count}}};
        Tensor chunk_q_offsets{phase_data.q_offsets.slice(phase_data.decode_count, phase_data.prefill_count + 1),
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
        delta_rule_.Run(arguments, *phase_data.chunked_plan, chunk_stream);
    }

    if (mixed) {
        TM_CUDA_CHECK(cudaEventRecord(ev_after_, aux_stream_));
        TM_CUDA_CHECK(cudaStreamWaitEvent(stream, ev_after_));
    }

    Tensor gate        = all_proj.slice({0, conv_dim}, {-1, value_dim});
    Tensor hidden_view = attn_out.view({token_num * num_v_heads_, head_dim_});
    invokeRMSNormGated(hidden_view, gate, weights.norm->weight, weights.norm->norm_eps_, stream);

    TM_SCOPE_CALL(linear_.Forward(attn_out, *weights.out_proj, param.output));
}

}  // namespace turbomind
