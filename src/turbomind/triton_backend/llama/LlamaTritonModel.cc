/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/triton_backend/multi_gpu_gpt/ParallelGptTritonModel.cc

#include <cctype>
#include <optional>
#include <string>

#include <cuda_runtime.h>

#include <yaml-cpp/yaml.h>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/model_request.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaV2.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/cuda_utils.h"

#include "src/turbomind/triton_backend/llama/LlamaTritonModel.h"

namespace turbomind {

static std::optional<MoeParam::Method> get_moe_method()
{
    static const auto value = []() -> std::optional<MoeParam::Method> {
        const auto p = std::getenv("TM_MOE_METHOD");
        if (p) {
            std::string str(p);
            for (auto& x : str) {
                x = std::tolower(x);
            }
            if (str == "naive") {
                return MoeParam::kNaive;
            }
            else if (str == "fused") {
                return MoeParam::kFused;
            }
            else {
                std::cerr << "[WARNING] unrecognised MoE method: " << str << "\n";
            }
        }
        return {};
    }();
    return value;
}

static void parse_default_rope_param(const YAML::Node& node, RopeParam& param)
{
    param.base = node["base"].as<float>();
    param.dim  = node["dim"].as<int>();
    if (param.base == 0.f || param.dim == 0) {
        TM_LOG_ERROR("invalid rope param: base = %f, dim = %d", param.base, param.dim);
        FT_CHECK(0);
    }
}

static void parse_linear_rope_param(const YAML::Node& node, RopeParam& param)
{
    parse_default_rope_param(node, param);
    param.factor = node["factor"].as<float>();
}

static void parse_dynamic_rope_param(const YAML::Node& node, RopeParam& param)
{
    parse_linear_rope_param(node, param);
    param.max_position_embeddings = node["max_position_embeddings"].as<int>();
}

static void parse_yarn_rope_param(const YAML::Node& node, RopeParam& param)
{
    parse_dynamic_rope_param(node, param);
    param.yarn.attention_factor = node["attention_factor"].as<float>();
    param.yarn.beta_fast        = node["beta_fast"].as<float>();
    param.yarn.beta_slow        = node["beta_slow"].as<float>();
}

static void parse_llama3_rope_param(const YAML::Node& node, RopeParam& param)
{
    parse_linear_rope_param(node, param);
    param.llama3.low_freq_factor                  = node["low_freq_factor"].as<float>();
    param.llama3.high_freq_factor                 = node["high_freq_factor"].as<float>();
    param.llama3.original_max_position_embeddings = node["original_max_position_embeddings"].as<int>();
}

static void parse_mrope_rope_param(const YAML::Node& node, RopeParam& param)
{
    parse_default_rope_param(node, param);
    auto mrope_section = node["mrope_section"].as<std::vector<int>>();
    FT_CHECK(mrope_section.size() == 3);
    param.mrope.section = {mrope_section[0], mrope_section[1], mrope_section[2]};
}

static void parse_rope_param(const YAML::Node& node, RopeParam& rope)
{
    rope.type = GetRoPEType(node["type"].as<std::string>());

    switch (rope.type) {
        case RopeType::kDefault:
            parse_default_rope_param(node, rope);
            break;
        case RopeType::kLinear:
            parse_linear_rope_param(node, rope);
            break;
        case RopeType::kDynamic:
            parse_dynamic_rope_param(node, rope);
            break;
        case RopeType::kYarn:
            parse_yarn_rope_param(node, rope);
            break;
        case RopeType::kLlama3:
            parse_llama3_rope_param(node, rope);
            break;
        case RopeType::kMrope:
            parse_mrope_rope_param(node, rope);
            break;
        default:
            FT_CHECK(0);
            break;
    }
}

DataType data_type_from_string(std::string str)
{
    if (str == "fp16" || str == "float16") {
        return kFloat16;
    }
    else if (str == "bf16" || str == "bfloat16") {
        return kBfloat16;
    }
    else if (str == "fp32") {
        return kFloat32;
    }
    else if (str == "int8") {
        return kUint8;
    }
    else if (str == "int4") {
        return kUint4;
    }
    else if (str == "fp8") {
        return kFloat8_e4m3;
    }
    else if (str == "e2m1") {
        return kFloat4_e2m1;
    }
    TM_CHECK(0) << "unsupported weight type: " << str;
    return {};
}

template<typename T>
std::map<std::string, std::pair<std::regex, T>> getLoraPattern(std::string pattern, T (*func)(const std::string& s))
{
    std::map<std::string, std::pair<std::regex, T>> res;
    std::stringstream                               ss(pattern);
    std::string                                     kv;
    while (std::getline(ss, kv, ',')) {
        auto pos = kv.rfind(":");
        auto k   = kv.substr(0, pos);
        auto v   = func(kv.substr(pos + 1));
        res.emplace(k, std::make_pair(std::regex(k), v));
    }
    return res;
}

void LlamaTritonModel::handleMissingParams()
{
    if (model_param_.kv_head_num == 0) {
        model_param_.kv_head_num = model_param_.head_num;
        TM_LOG_WARNING("[LlamaTritonModel] `kv_head_num` is not set, default to `head_num` (%d).",
                       (int)model_param_.kv_head_num);
    }

    if (model_param_.embedding_size == 0) {
        model_param_.embedding_size = model_param_.vocab_size;
        TM_LOG_WARNING("[LlamaTritonModel] `embedding_size` is not set, default to `vocab_size` (%d).",
                       (int)model_param_.vocab_size);
    }

    if (!attn_param_.max_position_embeddings) {
        attn_param_.max_position_embeddings = 2048;
        TM_LOG_WARNING("[LlamaTritonModel] `max_position_embeddings` is not set, default to %d.",
                       (int)attn_param_.max_position_embeddings);
    }

    if (!engine_param_.max_batch_size) {
        engine_param_.max_batch_size = 64;
        TM_LOG_WARNING("[LlamaTritonModel] `max_batch_size` is not set, default to %d.",
                       (int)engine_param_.max_batch_size);
    }

    if (!engine_param_.session_len) {
        engine_param_.session_len = attn_param_.max_position_embeddings;
        TM_LOG_WARNING("[LlamaTritonModel] `session_len` is not set, default to %d.", (int)engine_param_.session_len);
    }

    if (!engine_param_.max_context_token_num) {
        engine_param_.max_context_token_num = engine_param_.session_len;
        TM_LOG_WARNING("[LlamaTritonModel] `max_context_token_num` is not set, default to %d.",
                       (int)engine_param_.max_context_token_num);
    }

    if (engine_param_.max_context_token_num <= engine_param_.max_batch_size) {
        engine_param_.max_context_token_num *= engine_param_.session_len;
        TM_LOG_WARNING("[LlamaTritonModel] `max_context_token_num` = %d.", (int)engine_param_.max_context_token_num);
    }

    if (!engine_param_.step_length) {
        engine_param_.step_length = 1;
    }

    if (!engine_param_.cache_max_block_count) {
        engine_param_.cache_max_block_count = .95f;
        TM_LOG_WARNING("[LlamaTritonModel] `cache_max_entry_count` is not set, default to %f.",
                       engine_param_.cache_max_block_count);
    }

    if (!attn_param_.cache_block_seq_len) {
        attn_param_.cache_block_seq_len = 128;
        TM_LOG_WARNING("[LlamaTritonModel] `cache_block_seq_len` is not set, default to %d.",
                       attn_param_.cache_block_seq_len);
    }

    if (!engine_param_.cache_chunk_size) {
        engine_param_.cache_chunk_size = engine_param_.cache_max_block_count;
        TM_LOG_WARNING("[LlamaTritonModel] `cache_chunk_size` is not set, default to %d.",
                       (int)engine_param_.cache_chunk_size);
    }

    if (!engine_param_.num_tokens_per_iter) {
        engine_param_.num_tokens_per_iter = engine_param_.max_context_token_num;
        TM_LOG_WARNING("[LlamaTritonModel] `num_tokens_per_iter` is not set, default to `max_context_token_num` (%d).",
                       (int)engine_param_.num_tokens_per_iter);
    }
}

LlamaTritonModel::~LlamaTritonModel()
{
    FT_CHECK(weights_.size() == engines_.size());

    if (gateway_) {
        gateway_->shutdown();
    }

    for (int device_id = 0; device_id < (int)engines_.size(); ++device_id) {
        // Set device id before destructing CUDA resources
        CudaDeviceGuard dev_guard(engine_param_.devices[device_id]);
        engines_[device_id].reset();
        weights_[device_id].reset();
        contexts_[device_id].reset();
        trim_default_mempool(engine_param_.devices[device_id]);
    }
}

LlamaTritonModel::LlamaTritonModel(std::string                            model_dir,
                                   std::string                            config,
                                   std::function<std::shared_ptr<void>()> ffi_ctx_factory):
    dtype_{}, model_param_{}, attn_param_{}, moe_param_{}, lora_param_{}, engine_param_{}
{
    FT_CHECK_WITH_INFO(!(config.empty() && model_dir.empty()), "invalid init options");

    YAML::Node reader;

    try {
        if (!model_dir.empty()) {
            model_dir_ = model_dir;
            const std::string config_file{model_dir + "/config.yaml"};
            reader = YAML::LoadFile(config_file);
        }

        if (!config.empty()) {
            reader = YAML::Load(config);
        }
    }
    catch (const YAML::Exception& e) {
        std::cerr << "Error reading YAML config: " << e.what() << std::endl;
        FT_CHECK(false);
    }

    const auto model_reader     = reader["model_config"];
    const auto attention_reader = reader["attention_config"];
    const auto lora_reader      = reader["lora_config"];
    const auto engine_reader    = reader["engine_config"];

    dtype_ = model_param_.data_type = data_type_from_string(model_reader["data_type"].as<std::string>());
    TM_CHECK(model_param_.data_type == kBfloat16 || model_param_.data_type == kHalf);

    model_name_                     = model_reader["model_name"].as<std::string>();
    model_param_.head_num           = model_reader["head_num"].as<int>();
    model_param_.head_dim           = model_reader["size_per_head"].as<int>();
    model_param_.kv_head_num        = model_reader["kv_head_num"].as<int>(0);
    model_param_.hidden_units       = model_reader["hidden_units"].as<int>();
    model_param_.layer_num          = model_reader["num_layer"].as<int>();
    model_param_.vocab_size         = model_reader["vocab_size"].as<int>();
    model_param_.embedding_size     = model_reader["embedding_size"].as<int>();
    model_param_.norm_eps           = model_reader["norm_eps"].as<float>();
    model_param_.tune_layer_num     = model_reader["tune_layer_num"].as<int>(1);
    model_param_.mla.q_lora_rank    = model_reader["q_lora_rank"].as<int>();
    model_param_.mla.kv_lora_rank   = model_reader["kv_lora_rank"].as<int>();
    model_param_.mla.qk_rope_dim    = model_reader["qk_rope_dim"].as<int>();
    model_param_.mla.v_head_dim     = model_reader["v_head_dim"].as<int>();
    attn_param_.cache_block_seq_len = attention_reader["cache_block_seq_len"].as<int>(0);
    model_param_.quant_policy       = engine_reader["quant_policy"].as<int>(0);

    auto inter_size = model_reader["inter_size"];
    for (auto it = inter_size.begin(); it != inter_size.end(); ++it) {
        model_param_.inter_size.push_back(it->as<int>());
    }
    model_param_.attn_sink = model_reader["attn_sink"].as<bool>();
    model_param_.mlp_bias  = model_reader["mlp_bias"].as<bool>();
    if (model_reader["activation_type"].as<std::string>("") == "gpt-oss") {
        model_param_.act_type = ActivationType::kSiluGptOss;
    }

    auto window_size = model_reader["window_size"];
    for (auto it = window_size.begin(); it != window_size.end(); ++it) {
        model_param_.window_size.push_back(it->as<int>());
    }

    model_param_.attn_bias  = model_reader["attn_bias"].as<int>(0);
    model_param_.qk_norm    = model_reader["qk_norm"].as<bool>();
    model_param_.group_size = model_reader["group_size"].as<int>(0);

    attn_param_.softmax_scale = attention_reader["softmax_scale"].as<float>(0);
    // logn attn for qwen model
    attn_param_.use_logn_attn           = attention_reader["use_logn_attn"].as<int>(0);
    attn_param_.max_position_embeddings = attention_reader["max_position_embeddings"].as<int>(0);
    // rotary embedding parameters
    parse_rope_param(attention_reader["rope_param"], attn_param_.rope);

    engine_param_.max_batch_size = engine_reader["max_batch_size"].as<int>(0);
    auto max_forward_token_num   = engine_reader["max_prefill_token_num"].as<int>(0);
    max_forward_token_num += engine_param_.max_batch_size;

    engine_param_.max_context_token_num = engine_reader["max_context_token_num"].as<int>(0);
    engine_param_.session_len           = model_reader["session_len"].as<int>(0);

    engine_param_.cache_max_block_count = engine_reader["cache_max_entry_count"].as<float>(0);
    engine_param_.cache_chunk_size      = engine_reader["cache_chunk_size"].as<int>(0);
    engine_param_.enable_prefix_caching = engine_reader["enable_prefix_caching"].as<bool>(false);
    engine_param_.enable_metrics        = engine_reader["enable_metrics"].as<bool>(false);

    engine_param_.num_tokens_per_iter = engine_reader["num_tokens_per_iter"].as<int>(0);
    engine_param_.max_prefill_iters   = engine_reader["max_prefill_iters"].as<int>(1);

    engine_param_.outer_dp_size = engine_reader["outer_dp_size"].as<int>();
    engine_param_.outer_dp_rank = 0;
    engine_param_.attn_dp_size  = engine_reader["attn_dp_size"].as<int>();
    engine_param_.attn_dp_rank  = 0;
    engine_param_.attn_tp_size  = engine_reader["attn_tp_size"].as<int>();
    engine_param_.attn_tp_rank  = 0;
    engine_param_.mlp_tp_size   = engine_reader["mlp_tp_size"].as<int>();
    engine_param_.mlp_tp_rank   = 0;

    engine_param_.devices = engine_reader["devices"].as<std::vector<int>>();

    {
        auto tp                             = engine_param_.attn_tp_size;
        engine_param_.max_forward_token_num = ((size_t)max_forward_token_num + tp - 1) / tp * tp;
    }

    comm_size_ = engine_param_.attn_dp_size * engine_param_.attn_tp_size;
    FT_CHECK(engine_param_.mlp_tp_size == comm_size_);

    communicator_ = engine_reader["communicator"].as<std::string>();

    lora_param_.policy        = getLoraPolicy(reader["lora_config"]["lora_policy"].as<std::string>(""));
    lora_param_.r             = lora_reader["lora_r"].as<int>(0);
    lora_param_.scale         = lora_reader["lora_scale"].as<float>(0);
    lora_param_.max_wo_r      = lora_reader["lora_max_wo_r"].as<int>(0);
    lora_param_.rank_pattern  = getLoraPattern<int>(lora_reader["lora_rank_pattern"].as<std::string>(""),
                                                   [](const std::string& s) { return std::stoi(s); });
    lora_param_.scale_pattern = getLoraPattern<float>(lora_reader["lora_scale_pattern"].as<std::string>(""),
                                                      [](const std::string& s) { return std::stof(s); });

    moe_param_.experts_per_token = model_reader["experts_per_token"].as<int>(0);
    moe_param_.inter_size        = model_reader["expert_inter_size"].as<int>(0);
    moe_param_.shared_gate       = model_reader["moe_shared_gate"].as<bool>();
    moe_param_.norm_topk_prob    = model_reader["norm_topk_prob"].as<bool>();
    moe_param_.routed_scale      = model_reader["routed_scale"].as<float>(1.f);
    moe_param_.topk_group        = model_reader["topk_group"].as<int>(1);
    moe_param_.topk_method       = model_reader["topk_method"].as<std::string>("greedy");
    moe_param_.n_group           = model_reader["moe_group_num"].as<int>(1);
    moe_param_.router_bias       = model_reader["expert_router_bias"].as<bool>();
    YAML::Node expert_num        = model_reader["expert_num"];
    for (auto it = expert_num.begin(); it != expert_num.end(); ++it) {
        moe_param_.expert_num.push_back(it->as<int>());
    }

    handleMissingParams();

    gateway_ = std::make_shared<Gateway>(engine_param_.outer_dp_size, engine_param_.attn_dp_size, ffi_ctx_factory);
    ffi_ctx_factory_ = ffi_ctx_factory;

    weights_.resize(engine_param_.devices.size());
    engines_.resize(engine_param_.devices.size());
    contexts_.resize(engine_param_.devices.size());

    model_param_.weight_type        = data_type_from_string(model_reader["weight_type"].as<std::string>());
    model_param_.expert_weight_type = data_type_from_string(model_reader["expert_weight_type"].as<std::string>());

    if (auto method = get_moe_method()) {
        moe_param_.method = *method;
    }
    else {
        moe_param_.method = MoeParam::kFused;
    }

    // NOTE: This runs on Python main thread
    group_ids_.resize(engine_param_.outer_dp_size);
    for (size_t i = 0; i < group_ids_.size(); ++i) {
        group_ids_[i] = comm::CreateHostGroupId("");
        group_ids_[i]->Initialize();
    }

    const int device_num = engine_param_.outer_dp_size * comm_size_;

    engine_params_.resize(device_num, engine_param_);
    for (int i = 0; i < device_num; ++i) {
        auto& e         = engine_params_[i];
        e.outer_dp_rank = i / comm_size_;
        e.attn_tp_rank  = i % comm_size_ % e.attn_tp_size;
        e.attn_dp_rank  = i % comm_size_ / e.attn_tp_size;
        e.mlp_tp_rank   = i % comm_size_;
    }

    TM_LOG_INFO("%s", toString().c_str());
}

std::unique_ptr<ModelRequest> LlamaTritonModel::createModelInstance(int device_id)
{
    FT_CHECK(engines_[device_id] != nullptr);

    return std::make_unique<ModelRequest>(
        gateway_.get(), dtype_, engine_param_.session_len, model_param_.vocab_size, model_param_.hidden_units);
}

void LlamaTritonModel::createSharedWeights(int device_id, int rank)
{
    CudaDeviceGuard dev_guard(engine_param_.devices[device_id]);
    weights_[rank] =
        std::make_shared<LlamaWeight>(dtype_, model_param_, engine_params_.at(rank), lora_param_, moe_param_);
}

TensorMap LlamaTritonModel::getParams(int device_id, int rank)
{
    const auto& tensor_ptr_map = TM_CHECK_NOTNULL(weights_[rank])->get_parameters();
    TensorMap   params;
    for (const auto& [name, tensor_ptr] : tensor_ptr_map) {
        params[name] = *tensor_ptr;
    }
    return params;
}

void LlamaTritonModel::processWeights(int device_id, int rank)
{
    CudaDeviceGuard dev_guard(engine_param_.devices[device_id]);
    FT_CHECK(weights_[device_id] != nullptr);

    cudaDeviceProp props{};
    check_cuda_error(cudaGetDeviceProperties(&props, engine_param_.devices[device_id]));

    weights_[device_id]->prepare(props);
    sync_check_cuda_error();
}

Communicators LlamaTritonModel::createCommSplits(int rank)
{
    Communicators comm{};

    const int outer_rank = rank / comm_size_;
    const int inner_rank = rank % comm_size_;

    comm.h_comm = group_ids_[outer_rank]->CreateCommunicator(comm_size_, inner_rank);

    comm.h_tp_group = comm.h_comm->Split(inner_rank / engine_param_.attn_tp_size, 0);
    comm.h_dp_group = comm.h_comm->Split(inner_rank % engine_param_.attn_tp_size, 0);

    if (comm_size_ > 1) {
        comm.d_comm = CreateDeviceCommunicator(communicator_, comm_size_, inner_rank, comm.h_comm);
        //
        comm.d_tp_group = 0;
        if (engine_param_.attn_tp_size != comm_size_) {
            comm.d_tp_group = comm.d_comm->Split(inner_rank / engine_param_.attn_tp_size, 0, 0);
        }
    }

    return comm;
}

void LlamaTritonModel::createEngine(int device_id, int rank)
{
    CudaDeviceGuard dev_guard(engine_param_.devices[device_id]);

    auto&      ctx          = contexts_[device_id];
    const bool first_create = (ctx == nullptr);
    if (first_create) {
        ctx       = std::make_shared<Context>(engine_param_.devices[device_id]);
        ctx->comm = createCommSplits(rank);
    }

    core::ContextGuard guard{ctx->core_stream, ctx->allocator, Allocator{kCPUpinned}};

    const auto& engine_param = engine_params_.at(rank);

    // Get `h_comm` first as ctx will be moved later
    const auto h_comm = ctx->comm.h_comm;

    h_comm->Sync();

    auto model = std::make_unique<LlamaV2>(dtype_,
                                           model_param_,  //
                                           engine_param,
                                           attn_param_,
                                           moe_param_,
                                           lora_param_,
                                           *ctx,
                                           engine_param_.max_batch_size,
                                           weights_[device_id]);

    h_comm->Sync();

    try {
        const int dp_rank   = engine_param.outer_dp_rank * engine_param.attn_dp_size + engine_param.attn_dp_rank;
        engines_[device_id] = std::make_unique<Engine>(dtype_,
                                                       engine_param,  //
                                                       std::move(model),
                                                       ctx,
                                                       gateway_,
                                                       engine_param_.devices[device_id],
                                                       dp_rank);
    }
    catch (const std::exception& e) {
        TM_LOG_ERROR("[Engine][Init] %s", e.what());
        throw;
    }

    // Wait for pinned buffers to be allocated for all ranks, otherwise tuning will hang
    // due to concurrent kernel launch & cudaMallocHost

    h_comm->Sync();

    auto& engine = *engines_[device_id];

    if (first_create) {
        try {
            engine.Warmup();
        }
        catch (const std::exception& e) {
            TM_LOG_ERROR("[Engine][Warmup] %s", e.what());
            throw;
        }
    }

    h_comm->Sync();

    engine.Start();
}

ScheduleMetrics LlamaTritonModel::getScheduleMetrics(int device_id, int rank)
{
    auto& engine = *engines_[device_id];

    return engine.getScheduleMetrics();
}

void LlamaTritonModel::sleep(int device_id, int level)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    CudaDeviceGuard dev_guard(engine_param_.devices[device_id]);

    if (level == 2) {
        // free weights
        weights_[device_id]->release();
    }
    else {
        // offload weights to CPU
        TM_CHECK(moe_param_.experts_per_token == 0) << "level 1 sleep not supported for MoE model";
        weights_[device_id]->to_device(kCPU);
    }

    // free model (kv cache and buffer)
    if (device_id == 0) {
        gateway_->shutdown();
        gateway_.reset();
    }
    engines_[device_id].reset();
    contexts_[device_id]->allocator->trim(0);
    trim_default_mempool(engine_param_.devices[device_id]);
}

void LlamaTritonModel::wakeup(int device_id, const std::vector<std::string>& tags, int rank)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    CudaDeviceGuard dev_guard(engine_param_.devices[device_id]);

    std::set<std::string> keys(tags.begin(), tags.end());

    if (keys.find("weights") != keys.end()) {
        TM_CHECK(weights_[device_id] != nullptr);
        if (weights_[device_id]->is_initialized()) {
            weights_[device_id]->to_device(kDEVICE);
        }
        else {
            weights_[device_id]->initialize();
        }
    }

    if (keys.find("kv_cache") != keys.end()) {
        if (device_id == 0) {
            gateway_ =
                std::make_shared<Gateway>(engine_param_.outer_dp_size, engine_param_.attn_dp_size, ffi_ctx_factory_);
        }
        TM_CHECK(contexts_[device_id] != nullptr);
        contexts_[device_id]->comm.h_comm->Sync();
        createEngine(device_id, rank);
    }
}

std::string LlamaTritonModel::toString()
{
    std::stringstream ss;
    ss << "Model: "  //
       << "\nhead_num: " << model_param_.head_num << "\nkv_head_num: " << model_param_.kv_head_num
       << "\nsize_per_head: "
       << model_param_.head_dim
       //    << "\ninter_size: " << model_param_.inter_size
       << "\nnum_layer: " << model_param_.layer_num << "\nvocab_size: " << model_param_.vocab_size
       << "\nattn_bias: " << model_param_.attn_bias << "\nqk_norm: " << model_param_.qk_norm
       << "\nmax_batch_size: " << engine_param_.max_batch_size
       << "\nmax_context_token_num: " << engine_param_.max_context_token_num
       << "\nnum_tokens_per_iter: " << engine_param_.num_tokens_per_iter
       << "\nmax_prefill_iters: " << engine_param_.max_prefill_iters << "\nsession_len: " << engine_param_.session_len
       << "\ncache_max_entry_count: " << engine_param_.cache_max_block_count
       << "\ncache_block_seq_len: " << attn_param_.cache_block_seq_len
       << "\ncache_chunk_size: " << engine_param_.cache_chunk_size << "\nenable_prefix_caching: "
       << engine_param_.enable_prefix_caching
       //    << "\ntensor_para_size: " << tensor_para_size_ << "\npipeline_para_size: " << pipeline_para_size_
       << "\nmodel_name: " << model_name_ << "\nmodel_dir: " << model_dir_
       << "\nquant_policy: " << model_param_.quant_policy << "\ngroup_size: "
       << model_param_.group_size
       //    << "\nexpert_num: " << moe_param_.expert_num
       << "\nexpert_per_token: " << moe_param_.experts_per_token << "\nmoe_method: " << moe_param_.method << std::endl;

    return ss.str();
}

int LlamaTritonModel::getTensorParaSize()
{
    return engine_param_.attn_tp_size;
}

int LlamaTritonModel::getPipelineParaSize()
{
    return 1;
}

}  // namespace turbomind
