// Copyright (c) OpenMMLab. All rights reserved.

#include <filesystem>
#include <future>
#include <random>

#include "src/turbomind/turbomind.h"

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/core.h"

#include "src/turbomind/engine/engine.h"
#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/model_executor.h"
#include "src/turbomind/engine/model_request.h"

#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"

#include "src/turbomind/kernels/gemm/tuner/params.h"

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/metrics.h"

#include <yaml-cpp/yaml.h>

// #include "dbg.h"

namespace turbomind {

using std::vector;
using std::string;
using std::shared_ptr;
using std::unique_ptr;

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

/// TODO: move config parsing to suitable place
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

static DataType data_type_from_string(std::string str)
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

struct TurboMind::Impl {
    DataType       data_type_;
    ModelParam     model_param_;
    AttentionParam attn_param_;
    MoeParam       moe_param_;
    EngineParam    engine_param_;
    size_t         comm_size_;

    vector<EngineParam> engine_params_;

    string communicator_type_;  // communicator backend

    unique_ptr<comm::HostGroupId> group_id_;

    shared_ptr<Gateway> gateway_;

    FFICtxFactory ffi_ctx_factory_;

    vector<int> global_rank_;

    // Weights & engine instances for the ranks
    vector<shared_ptr<LlamaWeight>> weights_;
    vector<shared_ptr<Context>>     contexts_;
    vector<Engine>                  engines_;

    string model_name_;
    string model_dir_;

    vector<int> queue_id_;
    int         n_queues_{0};

    int need_warm_up_{1};
    int phases_{1};

    ~Impl();

    Impl(string model_dir, string config, FFICtxFactory ffi_ctx_factory);

    unique_ptr<ModelRequest> CreateRequest()
    {
        return std::make_unique<ModelRequest>(gateway_.get(),  //
                                              data_type_,
                                              engine_param_.session_len,
                                              model_param_.vocab_size,
                                              model_param_.hidden_units);
    }

    void CreateWeights(int index)
    {
        CudaDeviceGuard dev_guard(engine_param_.devices[index]);

        CreateContext(index);

        weights_[index] = std::make_shared<LlamaWeight>(data_type_,  //
                                                        model_param_,
                                                        engine_params_.at(index),
                                                        moe_param_);
    }

    TensorMap GetWeights(int index)
    {
        const auto& tensor_ptr_map = TM_CHECK_NOTNULL(weights_[index])->get_parameters();
        TensorMap   params;
        for (const auto& [name, tensor_ptr] : tensor_ptr_map) {
            params[name] = *tensor_ptr;
        }
        return params;
    }

    void ProcessWeights(int index)
    {
        CudaDeviceGuard dev_guard(engine_param_.devices[index]);
        FT_CHECK(weights_[index] != nullptr);

        cudaDeviceProp props{};
        check_cuda_error(cudaGetDeviceProperties(&props, engine_param_.devices[index]));

        weights_[index]->prepare(props);
        sync_check_cuda_error();
    }

    void CreateEngine(int index);

    void CreateContext(int index);

    void WarmUp(int index);

    void Sleep(int index, int level)
    {
        CudaDeviceGuard dev_guard(engine_param_.devices[index]);

        if (level == 2) {
            // free weights
            weights_[index]->release();
        }
        else {
            // offload weights to CPU
            TM_CHECK(moe_param_.experts_per_token == 0) << "level 1 sleep not supported for MoE model";
            weights_[index]->to_device(kCPU);
        }

        // free model (kv cache and buffer)
        if (index == 0) {
            gateway_->shutdown();
            gateway_.reset();
        }

        engines_[index] = {};
        contexts_[index]->allocator->trim(0);

        trim_default_mempool(engine_param_.devices[index]);
    }

    void WakeUp(int index, const std::vector<std::string>& tags)
    {
        CudaDeviceGuard dev_guard(engine_param_.devices[index]);

        std::set<std::string> keys(tags.begin(), tags.end());

        auto& ctx = *TM_CHECK_NOTNULL(contexts_[index]);

        if (keys.find("weights") != keys.end()) {
            TM_CHECK(weights_[index] != nullptr);
            if (weights_[index]->is_initialized()) {
                weights_[index]->to_device(kDEVICE);
            }
            else {
                weights_[index]->initialize();
            }
        }

        if (keys.find("kv_cache") != keys.end()) {
            if (index == 0) {
                gateway_ = std::make_shared<Gateway>(n_queues_, ffi_ctx_factory_);
            }
            CreateEngine(index);
        }
    }

    void HandleMissingParams()
    {
        if (!engine_param_.max_context_token_num) {
            engine_param_.max_context_token_num = engine_param_.session_len;
            TM_LOG_WARNING("[TM] `max_context_token_num` is not set, default to %d.",
                           (int)engine_param_.max_context_token_num);
        }

        if (engine_param_.max_context_token_num <= engine_param_.max_batch_size) {
            engine_param_.max_context_token_num *= engine_param_.session_len;
            TM_LOG_WARNING("[TM] `max_context_token_num` = %d.", (int)engine_param_.max_context_token_num);
        }
    }
};

TurboMind::Impl::~Impl()
{
    TM_LOG_INFO(__PRETTY_FUNCTION__);
    if (gateway_) {
        gateway_->shutdown();
    }
    for (int i = 0; i < (int)engines_.size(); ++i) {
        /// TODO: make device part of core::Context
        CudaDeviceGuard device(engine_param_.devices[i]);
        {
            core::ContextGuard context{contexts_[i]->core_stream};
            engines_[i]  = {};
            contexts_[i] = {};
        }
        weights_[i] = {};
    }
}

TurboMind::Impl::Impl(string model_dir, string config, FFICtxFactory ffi_ctx_factory):
    data_type_{}, model_param_{}, attn_param_{}, moe_param_{}, engine_param_{}, ffi_ctx_factory_{ffi_ctx_factory}
{
    TM_CHECK(!config.empty());

    YAML::Node node;
    try {
        node = YAML::Load(config);
    }
    catch (const YAML::Exception& e) {
        TM_CHECK(0) << "Error loading YAML config: " << e.what() << "\nconfig:\n" << config;
    }

    /// TODO: move config parsing to suitable place
    const auto model     = node["model_config"];
    const auto attention = node["attention_config"];
    const auto engine    = node["engine_config"];

    data_type_ = model_param_.data_type = data_type_from_string(model["data_type"].as<std::string>());
    TM_CHECK(data_type_ == kBfloat16 || data_type_ == kHalf);

    model_name_                     = model["model_name"].as<std::string>();
    model_param_.head_num           = model["head_num"].as<int>();
    model_param_.head_dim           = model["size_per_head"].as<int>();
    model_param_.kv_head_num        = model["kv_head_num"].as<int>(0);
    model_param_.hidden_units       = model["hidden_units"].as<int>();
    model_param_.layer_num          = model["num_layer"].as<int>();
    model_param_.vocab_size         = model["vocab_size"].as<int>();
    model_param_.embedding_size     = model["embedding_size"].as<int>();
    model_param_.norm_eps           = model["norm_eps"].as<float>();
    model_param_.tune_layer_num     = model["tune_layer_num"].as<int>(1);
    model_param_.mla.q_lora_rank    = model["q_lora_rank"].as<int>();
    model_param_.mla.kv_lora_rank   = model["kv_lora_rank"].as<int>();
    model_param_.mla.qk_rope_dim    = model["qk_rope_dim"].as<int>();
    model_param_.mla.v_head_dim     = model["v_head_dim"].as<int>();
    attn_param_.cache_block_seq_len = attention["cache_block_seq_len"].as<int>(0);
    model_param_.quant_policy       = engine["quant_policy"].as<int>(0);

    auto inter_size = model["inter_size"];
    for (auto it = inter_size.begin(); it != inter_size.end(); ++it) {
        model_param_.inter_size.push_back(it->as<int>());
    }
    model_param_.attn_sink = model["attn_sink"].as<bool>();
    model_param_.mlp_bias  = model["mlp_bias"].as<bool>();
    if (model["activation_type"].as<std::string>("") == "gpt-oss") {
        model_param_.act_type = ActivationType::kSiluGptOss;
    }

    auto window_size = model["window_size"];
    for (auto it = window_size.begin(); it != window_size.end(); ++it) {
        model_param_.window_size.push_back(it->as<int>());
    }

    model_param_.attn_bias  = model["attn_bias"].as<int>(0);
    model_param_.qk_norm    = model["qk_norm"].as<bool>();
    model_param_.group_size = model["group_size"].as<int>(0);

    attn_param_.softmax_scale = attention["softmax_scale"].as<float>(0);
    // logn attn for qwen model
    attn_param_.use_logn_attn           = attention["use_logn_attn"].as<int>(0);
    attn_param_.max_position_embeddings = attention["max_position_embeddings"].as<int>(0);
    // rotary embedding parameters
    parse_rope_param(attention["rope_param"], attn_param_.rope);

    engine_param_.max_batch_size = engine["max_batch_size"].as<int>(0);
    auto max_forward_token_num   = engine["max_prefill_token_num"].as<int>(0);
    max_forward_token_num += engine_param_.max_batch_size;

    engine_param_.max_context_token_num = engine["max_context_token_num"].as<int>(0);
    engine_param_.session_len           = model["session_len"].as<int>(0);

    engine_param_.cache_max_block_count = engine["cache_max_entry_count"].as<float>(0);
    engine_param_.cache_chunk_size      = engine["cache_chunk_size"].as<int>(0);
    engine_param_.enable_prefix_caching = engine["enable_prefix_caching"].as<bool>(false);
    engine_param_.enable_metrics        = engine["enable_metrics"].as<bool>(false);

    engine_param_.num_tokens_per_iter = engine["num_tokens_per_iter"].as<int>(0);
    engine_param_.max_prefill_iters   = engine["max_prefill_iters"].as<int>(1);

    phases_ = engine["async_"].as<int>() ? 2 : 1;

    engine_param_.outer_dp_size = engine["outer_dp_size"].as<int>();

    engine_param_.attn_dp_size = engine["attn_dp_size"].as<int>();
    engine_param_.attn_tp_size = engine["attn_tp_size"].as<int>();
    engine_param_.attn_cp_size = engine["attn_cp_size"].as<int>();

    engine_param_.mlp_tp_size = engine["mlp_tp_size"].as<int>();

    engine_param_.devices = engine["devices"].as<std::vector<int>>();

    // multi-node information
    engine_param_.nnodes    = engine["nnodes"].as<int>();
    engine_param_.node_rank = engine["node_rank"].as<int>();

    {
        auto sp                             = engine_param_.attn_tp_size * engine_param_.attn_cp_size;
        engine_param_.max_forward_token_num = ((size_t)max_forward_token_num + sp - 1) / sp * sp;
    }

    comm_size_ = engine_param_.attn_dp_size * engine_param_.attn_tp_size * engine_param_.attn_cp_size;
    FT_CHECK(engine_param_.mlp_tp_size == comm_size_);

    communicator_type_ = engine["communicator"].as<std::string>();

    moe_param_.experts_per_token = model["experts_per_token"].as<int>(0);
    moe_param_.inter_size        = model["expert_inter_size"].as<int>(0);
    moe_param_.shared_gate       = model["moe_shared_gate"].as<bool>();
    moe_param_.norm_topk_prob    = model["norm_topk_prob"].as<bool>();
    moe_param_.routed_scale      = model["routed_scale"].as<float>(1.f);
    moe_param_.topk_group        = model["topk_group"].as<int>(1);
    moe_param_.topk_method       = model["topk_method"].as<std::string>("greedy");
    moe_param_.n_group           = model["moe_group_num"].as<int>(1);
    moe_param_.router_bias       = model["expert_router_bias"].as<bool>();
    YAML::Node expert_num        = model["expert_num"];
    for (auto it = expert_num.begin(); it != expert_num.end(); ++it) {
        moe_param_.expert_num.push_back(it->as<int>());
    }

    HandleMissingParams();

    weights_.resize(engine_param_.devices.size());
    engines_.resize(engine_param_.devices.size());
    contexts_.resize(engine_param_.devices.size());

    model_param_.weight_type        = data_type_from_string(model["weight_type"].as<std::string>());
    model_param_.expert_weight_type = data_type_from_string(model["expert_weight_type"].as<std::string>());

    if (auto method = get_moe_method()) {
        moe_param_.method = *method;
    }
    else {
        moe_param_.method = MoeParam::kFused;
    }

    // NOTE: This runs on Python main thread
    group_id_ = comm::CreateHostGroupId((engine_param_.nnodes == 1) ? "" : "hybrid");
    group_id_->Initialize();

    const int devices = engine_param_.devices.size();

    for (int i = 0; i < devices; ++i) {
        global_rank_.push_back(engine_param_.node_rank * devices + i);
    }

    queue_id_.resize(devices);
    engine_params_.resize(devices, engine_param_);
}

void TurboMind::Impl::CreateContext(int index)
{
    auto& p = engine_params_[index];

    CudaDeviceGuard dev_guard(p.devices[index]);

    TM_CHECK(contexts_[index] == nullptr);

    auto& ctx = contexts_[index] = std::make_shared<Context>(p.devices[index]);

    // Layout: (outer, dp, tp, cp)

    const int global_rank = global_rank_[index];

    const int outer_rank = global_rank / comm_size_;
    const int inner_rank = global_rank % comm_size_;

    p.outer_dp_rank = outer_rank;

    const int tp_cp_size = p.attn_tp_size * p.attn_cp_size;

    const int tp_color = inner_rank / tp_cp_size;
    const int dp_color = inner_rank % tp_cp_size;
    const int cp_color = inner_rank / p.attn_cp_size;

    auto& c = ctx->comm;

    c.h_global = group_id_->CreateCommunicator(comm_size_, global_rank, p.node_rank);

    c.h_comm = c.h_global->Split(outer_rank, 0);

    c.h_tp_group = c.h_comm->Split(tp_color, 0);
    c.h_dp_group = c.h_comm->Split(dp_color, 0);

    if (comm_size_ > 1) {
        c.d_comm = CreateDeviceCommunicator(communicator_type_, comm_size_, inner_rank, c.h_comm);

        c.d_tp_group = 0;
        c.d_cp_group = 0;

        if (p.attn_dp_size > 1) {  // has attn_dp
            c.d_tp_group   = c.d_comm->Split(tp_color, 0, 0);
            p.attn_dp_rank = c.h_dp_group->rank();
        }

        if (p.attn_cp_size > 1) {  // has attn_cp
            c.d_cp_group   = c.d_comm->Split(cp_color, 0, 0);
            p.attn_cp_rank = c.d_comm->rank(c.d_cp_group);
        }

        p.attn_tp_rank = c.d_comm->rank(c.d_tp_group) / p.attn_cp_size;
        p.mlp_tp_rank  = c.d_comm->rank(0);
    }

    if (c.h_tp_group->rank() == 0) {
        queue_id_[index] = 1;
    }

    c.h_global->Sync();

    if (index == 0) {
        n_queues_ = 0;
        for (size_t i = 0; i < queue_id_.size(); ++i) {
            queue_id_[i] = queue_id_[i] ? n_queues_++ : -1;
        }
        gateway_ = std::make_shared<Gateway>(n_queues_, ffi_ctx_factory_);
    }

    c.h_global->Sync();
}

void TurboMind::Impl::CreateEngine(int index)
{
    CudaDeviceGuard dev_guard(engine_param_.devices[index]);

    auto& ctx = *TM_CHECK_NOTNULL(contexts_[index]);

    core::ContextGuard guard{ctx.core_stream, ctx.allocator, Allocator{kCPUpinned}};

    const auto& param = engine_params_.at(index);

    ctx.comm.h_comm->Sync();

    // create model
    LanguageModel model{data_type_,  //
                        model_param_,
                        param,
                        attn_param_,
                        moe_param_,
                        ctx,
                        *weights_[index],
                        phases_};

    // create engine
    engines_[index] = Engine{data_type_,  //
                             param,
                             std::move(model),
                             ctx,
                             *gateway_,
                             engine_param_.devices[index],
                             queue_id_[index],
                             phases_};

    core::Context::stream().Sync();

    ctx.comm.h_comm->Sync();

    engines_[index].Start();

    if (need_warm_up_) {
        WarmUp(index);
    }
}

template<class Iter>
static std::string Join(Iter first, Iter last, const std::string& delim)
{
    if (first == last) {
        return {};
    }
    std::ostringstream oss;
    oss << *first++;
    while (first != last) {
        oss << delim << *first++;
    }
    return oss.str();
}

void TurboMind::Impl::WarmUp(int index)
{
    auto& ctx = *TM_CHECK_NOTNULL(contexts_[index]);

    auto& global = ctx.comm.h_global;
    auto& linear = *ctx.linear;

    if (auto str = std::getenv("TM_GEMM_IMPORT")) {
        std::ifstream ifs(str);
        const int     n_imported = linear.Import(ifs);
        if (index == 0) {
            TM_LOG_INFO("[GEMM] %d records imported", n_imported);
        }
        return;
    }

    global->Sync();

    *ctx.is_warm_up = 1;
    linear.set_measure(true);

    if (index == 0) {
        gateway_->set_threshold(engine_param_.attn_dp_size);
    }

    global->Sync();

    if (ctx.comm.h_tp_group->rank() == 0) {

        std::vector<int> bss = linear.GetTuningSeq();
        if (bss.empty()) {
            bss = gemm::GenerateTuningSequence(gemm::GetDefaultTuningGenerators());
        }

        const int max_fwd_token_num = engine_param_.max_forward_token_num;

        // remove bs that is too large
        bss.erase(std::remove_if(bss.begin(), bss.end(), [&](auto x) { return x > max_fwd_token_num; }), bss.end());

        if (bss.empty() || bss.back() < max_fwd_token_num) {
            bss.push_back(max_fwd_token_num);
        }

        auto str = Join(bss.begin(), bss.end(), ", ");
        TM_LOG_INFO("[Engine] Warm-up lengths: %s", str.c_str());

        if (!bss.empty()) {
            const auto                         max_bs = *std::max_element(bss.begin(), bss.end());
            Buffer_<int>                       input_ids(max_bs, kCPU);
            std::mt19937                       g{};
            std::uniform_int_distribution<int> d{0, (int)model_param_.vocab_size - 1};
            for (auto& x : input_ids) {
                x = d(g);
            }

            auto tick = std::chrono::steady_clock::now();

            for (auto token_num : bss) {

                TM_LOG_INFO("[WarmUp] %d", token_num);

                auto r = CreateRequest();

                TensorMap inputs{{"input_ids", input_ids.slice(0, token_num)}};

                ModelRequest::InputParam param{};
                param.session.start_flag     = true;
                param.session.end_flag       = true;
                param.gen_cfg.max_new_tokens = 1;
                param.tensors                = std::make_shared<TensorMap>(inputs);

                struct Channel {
                    int                flag = 1;
                    std::promise<void> promise;
                };
                auto c = std::make_shared<Channel>();

                ModelRequest::OutputParam out = r->Forward(std::move(param), [c] {
                    /// NOTE: It's risky to set `out.state` here, `out` may not be initialized at this point
                    if (std::exchange(c->flag, 0)) {
                        c->promise.set_value();
                    }
                });

                c->promise.get_future().get();

                int status = -1;
                if (auto state = out.state->exchange(nullptr)) {
                    status = state->status;
                }

                if (status != Request::kFinish) {
                    TM_LOG_ERROR("[Engine] Warm-up for %d tokens failed with status %d", (int)token_num, (int)status);
                }
            }

            auto tock = std::chrono::steady_clock::now();

            TM_LOG_INFO("[WarmUp] Warm-up finished in %.2f seconds.",
                        std::chrono::duration<float, std::ratio<1, 1>>(tock - tick).count());
        }
    }

    global->Sync();

    linear.set_measure(false);
    *ctx.is_warm_up = 0;

    if (index == 0) {
        if (auto path = std::getenv("TM_GEMM_EXPORT")) {
            std::ofstream ofs(path);
            const auto    n_records = linear.Export(ofs);
            TM_LOG_INFO("[GEMM] %d records exported.", n_records);
        }

        gateway_->set_threshold(1);
        need_warm_up_ = 0;
    }

    global->Sync();
}

TurboMind::~TurboMind() = default;

TurboMind::TurboMind(string model_dir, string config, FFICtxFactory ffi_ctx_factory):
    impl_{std::make_unique<Impl>(model_dir, config, ffi_ctx_factory)}
{
}

void TurboMind::CreateWeights(int index)
{
    return impl_->CreateWeights(index);
}

TensorMap TurboMind::GetWeights(int index)
{
    return impl_->GetWeights(index);
}

void TurboMind::ProcessWeights(int index)
{
    return impl_->ProcessWeights(index);
}

void TurboMind::CreateEngine(int index)
{
    return impl_->CreateEngine(index);
}

void TurboMind::Sleep(int index, int level)
{
    return impl_->Sleep(index, level);
}

void TurboMind::WakeUp(int index, const vector<string>& tags)
{
    return impl_->WakeUp(index, tags);
}

shared_ptr<ScheduleMetrics> TurboMind::GetScheduleMetrics(int index)
{
    return impl_->engines_[index].GetScheduleMetrics();
}

unique_ptr<ModelRequest> TurboMind::CreateRequest()
{
    return impl_->CreateRequest();
}

bool TurboMind::is_dummy_node() const noexcept
{
    return impl_->n_queues_ == 0;
}

}  // namespace turbomind
