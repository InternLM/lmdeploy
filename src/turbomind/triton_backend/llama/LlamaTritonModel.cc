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

#include "src/turbomind/triton_backend/llama/LlamaTritonModel.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaInstanceComm.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/triton_backend/llama/LlamaTritonModelInstance.h"
#include "src/turbomind/triton_backend/transformer_triton_backend.hpp"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cuda_utils.h"
#include <cuda_runtime.h>
#include <mutex>
#include <yaml-cpp/yaml.h>

namespace ft = turbomind;

std::shared_ptr<AbstractTransformerModel> AbstractTransformerModel::createLlamaModel(std::string config_file)
{
    YAML::Node reader;
    try {
        reader = YAML::Load(config_file);
    }
    catch (const YAML::Exception& e) {
        std::cerr << "Error reading YAML config: " << e.what() << std::endl;
        ft::FT_CHECK(false);
    }

    const auto        ft_instance_hyperparameter = reader["ft_instance_hyperparameter"];
    const std::string data_type                  = ft_instance_hyperparameter["data_type"].as<std::string>();
    int               tensor_para_size           = ft_instance_hyperparameter["tensor_para_size"].as<int>();
    std::string       model_dir                  = ft_instance_hyperparameter["model_dir"].as<std::string>();

    if (data_type == "half" || data_type == "fp16" || data_type == "float16") {
        return std::make_shared<LlamaTritonModel<half>>(
            ft_instance_hyperparameter["tensor_para_size"].as<int>(),
            ft_instance_hyperparameter["pipeline_para_size"].as<int>(),
            ft_instance_hyperparameter["enable_custom_all_reduce"].as<int>(0),
            model_dir);
    }
    else if (data_type == "bf16" || data_type == "bfloat16") {
#ifdef ENABLE_BF16
        return std::make_shared<LlamaTritonModel<__nv_bfloat16>>(
            ft_instance_hyperparameter["tensor_para_size"].as<int>(),
            ft_instance_hyperparameter["pipeline_para_size"].as<int>(),
            ft_instance_hyperparameter["enable_custom_all_reduce"].as<int>(0),
            model_dir);
#else
        TM_LOG_ERROR("[ERROR] Turbomind is not built with ENABLE_BF16");
        ft::FT_CHECK(false);
#endif
    }
    else {
#ifdef ENABLE_FP32
        return std::make_shared<LlamaTritonModel<float>>(
            ft_instance_hyperparameter["tensor_para_size"].as<int>(),
            ft_instance_hyperparameter["pipeline_para_size"].as<int>(),
            ft_instance_hyperparameter["enable_custom_all_reduce"].as<int>(0),
            model_dir);
#else
        TM_LOG_ERROR("[ERROR] Turbomind is not built with ENABLE_BF32");
        ft::FT_CHECK(false);
#endif
    }
    return nullptr;
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

template<typename T>
void LlamaTritonModel<T>::handleMissingParams()
{
    if (model_param_.kv_head_num == 0) {
        model_param_.kv_head_num = model_param_.head_num;
        TM_LOG_WARNING("[LlamaTritonModel] `kv_head_num` is not set, default to `head_num` (%d).",
                       (int)model_param_.kv_head_num);
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

    if (!engine_param_.max_prefill_token_num) {
        engine_param_.max_prefill_token_num = 8192;
        TM_LOG_WARNING("[LlamaTritonModel] `max_prefill_token_num` is not set, default to %d.",
                       (int)engine_param_.max_prefill_token_num);
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

template<typename T>
LlamaTritonModel<T>::~LlamaTritonModel()
{
    ft::FT_CHECK(weights_.size() == engines_.size());
    for (int device_id = 0; device_id < (int)engines_.size(); ++device_id) {
        // Set device id before destructing CUDA resources
        ft::check_cuda_error(cudaSetDevice(device_id));
        engines_[device_id].reset();
        weights_[device_id].reset();
    }
}

template<typename T>
LlamaTritonModel<T>::LlamaTritonModel(size_t      tensor_para_size,
                                      size_t      pipeline_para_size,
                                      int         enable_custom_all_reduce,
                                      std::string model_dir,
                                      std::string config):
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    weights_(ft::getDeviceCount()),
    enable_custom_all_reduce_(enable_custom_all_reduce)
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
        ft::FT_CHECK(false);
    }

    const auto model_reader     = reader["model_config"];
    const auto attention_reader = reader["attention_config"];
    const auto lora_reader      = reader["lora_config"];
    const auto engine_reader    = reader["engine_config"];

    model_name_                     = model_reader["model_name"].as<std::string>();
    model_param_.head_num           = model_reader["head_num"].as<int>();
    model_param_.head_dim           = model_reader["size_per_head"].as<int>();
    model_param_.kv_head_num        = model_reader["kv_head_num"].as<int>(0);
    model_param_.hidden_units       = model_reader["hidden_units"].as<int>();
    model_param_.layer_num          = model_reader["num_layer"].as<int>();
    model_param_.inter_size         = model_reader["inter_size"].as<int>();
    model_param_.vocab_size         = model_reader["vocab_size"].as<int>();
    model_param_.norm_eps           = model_reader["norm_eps"].as<float>();
    model_param_.start_id           = model_reader["start_id"].as<int>();
    model_param_.end_id             = model_reader["end_id"].as<int>();
    attn_param_.cache_block_seq_len = attention_reader["cache_block_seq_len"].as<int>(0);
    model_param_.quant_policy       = engine_reader["quant_policy"].as<int>(0);

    // Only weight classes need these
    attn_bias_  = model_reader["attn_bias"].as<int>(0);
    group_size_ = model_reader["group_size"].as<int>(0);

    // rotary embedding parameters
    attn_param_.rotary_embedding_dim    = attention_reader["rotary_embedding"].as<int>();
    attn_param_.rotary_embedding_base   = attention_reader["rope_theta"].as<float>(10000.0f);
    attn_param_.rope_scaling_type       = attention_reader["rope_scaling_type"].as<std::string>("");
    attn_param_.rope_scaling_factor     = attention_reader["rope_scaling_factor"].as<float>(0.f);
    attn_param_.low_freq_factor         = attention_reader["low_freq_factor"].as<float>(1.0);
    attn_param_.high_freq_factor        = attention_reader["high_freq_factor"].as<float>(1.0);
    attn_param_.max_position_embeddings = attention_reader["max_position_embeddings"].as<int>(0);
    attn_param_.use_dynamic_ntk         = attention_reader["use_dynamic_ntk"].as<int>(0);
    attn_param_.use_logn_attn           = attention_reader["use_logn_attn"].as<int>(0);

    attn_param_.original_max_position_embeddings = attention_reader["original_max_position_embeddings"].as<int>(0);

    engine_param_.max_batch_size        = engine_reader["max_batch_size"].as<int>(0);
    engine_param_.max_prefill_token_num = engine_reader["max_prefill_token_num"].as<int>(0);
    engine_param_.max_context_token_num = engine_reader["max_context_token_num"].as<int>(0);
    engine_param_.session_len           = model_reader["session_len"].as<int>(0);

    engine_param_.cache_max_block_count = engine_reader["cache_max_entry_count"].as<float>(0);
    engine_param_.cache_chunk_size      = engine_reader["cache_chunk_size"].as<int>(0);
    engine_param_.enable_prefix_caching = engine_reader["enable_prefix_caching"].as<bool>(false);

    engine_param_.num_tokens_per_iter = engine_reader["num_tokens_per_iter"].as<int>(0);
    engine_param_.max_prefill_iters   = engine_reader["max_prefill_iters"].as<int>(1);

    lora_param_.policy        = ft::getLoraPolicy(reader["lora_config"]["lora_policy"].as<std::string>(""));
    lora_param_.r             = lora_reader["lora_r"].as<int>(0);
    lora_param_.scale         = lora_reader["lora_scale"].as<float>(0);
    lora_param_.max_wo_r      = lora_reader["lora_max_wo_r"].as<int>(0);
    lora_param_.rank_pattern  = getLoraPattern<int>(lora_reader["lora_rank_pattern"].as<std::string>(""),
                                                   [](const std::string& s) { return std::stoi(s); });
    lora_param_.scale_pattern = getLoraPattern<float>(lora_reader["lora_scale_pattern"].as<std::string>(""),
                                                      [](const std::string& s) { return std::stof(s); });
    handleMissingParams();

    shared_state_          = std::make_shared<ft::SharedState>();
    shared_state_->barrier = std::make_shared<ft::Barrier>(tensor_para_size);

    const auto device_count = ft::getDeviceCount();
    engines_.resize(device_count);

    const std::string weight_type_str = model_reader["weight_type"].as<std::string>();
    if (weight_type_str == "fp16" || weight_type_str == "float16") {
        weight_type_ = ft::WeightType::kFP16;
    }
    else if (weight_type_str == "bf16" || weight_type_str == "bfloat16") {
        weight_type_ = ft::WeightType::kBF16;
    }
    else if (weight_type_str == "fp32") {
        weight_type_ = ft::WeightType::kFP32;
    }
    else if (weight_type_str == "int8") {
        weight_type_ = ft::WeightType::kINT8;
    }
    else if (weight_type_str == "int4") {
        weight_type_ = ft::WeightType::kINT4;
    }
    else {
        std::cout << "[ERROR] Unsupported weight type: '" << weight_type_str << "'\n";
        ft::FT_CHECK(0);
    }

    TM_LOG_INFO("%s", toString().c_str());
}

template<typename T>
std::unique_ptr<ft::Engine<T>> LlamaTritonModel<T>::createSharedModelInstance(
    int                                                               device_id,
    int                                                               rank,
    std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
    std::shared_ptr<ft::AbstractCustomComm>                           custom_all_reduce_comm)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    const int comms_rank = device_id % (tensor_para_size_ * pipeline_para_size_);

    auto ctx = std::make_unique<ft::Context<T>>();

    ft::check_cuda_error(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking));

    ctx->allocator = std::make_unique<ft::Allocator<ft::AllocatorType::CUDA>>(device_id, false);
    ctx->allocator->setStream(ctx->stream);

    ctx->peer_allocator = std::make_unique<ft::Allocator<ft::AllocatorType::CUDA>>(device_id, true);
    ctx->peer_allocator->setStream(ctx->stream);

    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;

    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, ctx->stream);

    ctx->cublas_algo_map      = std::make_unique<ft::cublasAlgoMap>("gemm_config.in");
    ctx->cublas_wrapper_mutex = std::make_unique<std::mutex>();
    ctx->cublas_wrapper       = std::make_unique<ft::cublasMMWrapper>(cublas_handle,
                                                                cublaslt_handle,
                                                                ctx->stream,
                                                                ctx->cublas_algo_map.get(),
                                                                ctx->cublas_wrapper_mutex.get(),
                                                                ctx->allocator.get());
    ctx->linear               = std::make_unique<ft::LlamaLinear<T>>(ctx->cublas_wrapper.get(), ctx->stream);

    ft::check_cuda_error(cudaGetDeviceProperties(&ctx->cuda_device_prop, device_id));

    if (std::is_same<T, half>::value) {
        ctx->cublas_wrapper->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
#ifdef ENABLE_FP32
    else if (std::is_same<T, float>::value) {
        ctx.cublas_wrapper->setFP32GemmConfig();
    }
#endif
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        ctx->cublas_wrapper->setBF16GemmConfig();
    }
#endif

    ft::NcclParam tensor_para   = nccl_params.first[comms_rank];
    ft::NcclParam pipeline_para = nccl_params.second[comms_rank];

    ft::FT_CHECK(tensor_para.world_size_ == tensor_para_size_);
    ft::FT_CHECK(pipeline_para.world_size_ == pipeline_para_size_);

    auto model = std::make_unique<ft::LlamaV2<T>>(model_param_,  //
                                                  attn_param_,
                                                  lora_param_,
                                                  tensor_para,
                                                  *ctx,
                                                  engine_param_.max_batch_size,
                                                  weights_[device_id]);

    auto engine = std::make_unique<ft::Engine<T>>(engine_param_,  //
                                                  std::move(model),
                                                  std::move(ctx),
                                                  shared_state_,
                                                  device_id);

    engine->Start();

    return engine;
}

template<typename T>
std::unique_ptr<AbstractTransformerModelInstance>
LlamaTritonModel<T>::createModelInstance(int          device_id,
                                         int          rank,
                                         cudaStream_t stream,
                                         std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>>,
                                         std::shared_ptr<ft::AbstractCustomComm>)
{
    ft::check_cuda_error(cudaSetDevice(device_id));

    ft::FT_CHECK(engines_[device_id] != nullptr);

    auto allocator = std::make_unique<ft::Allocator<ft::AllocatorType::CUDA>>(device_id, false);

    allocator->setStream(stream);

    return std::make_unique<LlamaTritonModelInstance<T>>(*engines_[device_id], std::move(allocator), device_id);
}

template<typename T>
void LlamaTritonModel<T>::createSharedWeights(int device_id, int rank)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    const int tensor_para_rank   = rank % tensor_para_size_;
    const int pipeline_para_rank = rank / tensor_para_size_;
    ft::FT_CHECK(pipeline_para_size_ == 1 && pipeline_para_rank == 0);
    weights_[device_id] = std::make_shared<ft::LlamaWeight<T>>(model_param_.head_num,
                                                               model_param_.kv_head_num,
                                                               model_param_.head_dim,
                                                               model_param_.hidden_units,
                                                               model_param_.inter_size,
                                                               model_param_.vocab_size,
                                                               model_param_.layer_num,
                                                               attn_bias_,
                                                               weight_type_,
                                                               group_size_,
                                                               lora_param_,
                                                               tensor_para_size_,
                                                               tensor_para_rank);
    // model inited with model_dir
    if (model_dir_ != "") {
        weights_[device_id]->loadModel(model_dir_);
    }
    return;
}

template<typename T>
TensorMap LlamaTritonModel<T>::getParams(int deviceId, int rank)
{
    ft::check_cuda_error(cudaSetDevice(deviceId));
    // shared_weight should be created before getParams
    ft::FT_CHECK(weights_[deviceId] != nullptr);
    ft::TensorMap output = weights_[deviceId]->getParams();
    TensorMap     result;
    for (auto [name, tensor] : output) {
        result.emplace(name, triton::Tensor{tensor.where, tensor.type, tensor.shape, tensor.data});
    }
    return result;
}

template<typename T>
void LlamaTritonModel<T>::processWeights(int device_id, int rank)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    ft::FT_CHECK(weights_[device_id] != nullptr);

    cudaDeviceProp props{};
    ft::check_cuda_error(cudaGetDeviceProperties(&props, device_id));

    weights_[device_id]->prepare(props);
    ft::sync_check_cuda_error();
}

template<typename T>
void LlamaTritonModel<T>::createEngine(int                                                               device_id,
                                       int                                                               rank,
                                       std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
                                       std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm)
{

    auto engine = createSharedModelInstance(device_id, rank, nccl_params, custom_all_reduce_comm);
    engine->set_ffi_lock(ffi_lock_);

    if (weight_type_ == ft::WeightType::kINT4) {
        engine->model().tune();
    }

    engines_[device_id] = std::move(engine);
}

template<typename T>
std::string LlamaTritonModel<T>::toString()
{
    std::stringstream ss;
    ss << "Model: "  //
       << "\nhead_num: " << model_param_.head_num << "\nkv_head_num: " << model_param_.kv_head_num
       << "\nsize_per_head: " << model_param_.head_dim << "\ninter_size: " << model_param_.inter_size
       << "\nnum_layer: " << model_param_.layer_num << "\nvocab_size: " << model_param_.vocab_size
       << "\nattn_bias: " << attn_bias_ << "\nmax_batch_size: " << engine_param_.max_batch_size
       << "\nmax_prefill_token_num: " << engine_param_.max_prefill_token_num
       << "\nmax_context_token_num: " << engine_param_.max_context_token_num
       << "\nnum_tokens_per_iter: " << engine_param_.num_tokens_per_iter
       << "\nmax_prefill_iters: " << engine_param_.max_prefill_iters << "\nsession_len: " << engine_param_.session_len
       << "\ncache_max_entry_count: " << engine_param_.cache_max_block_count
       << "\ncache_block_seq_len: " << attn_param_.cache_block_seq_len
       << "\ncache_chunk_size: " << engine_param_.cache_chunk_size
       << "\nenable_prefix_caching: " << engine_param_.enable_prefix_caching << "\nstart_id: " << model_param_.start_id
       << "\ntensor_para_size: " << tensor_para_size_ << "\npipeline_para_size: " << pipeline_para_size_
       << "\nenable_custom_all_reduce: " << enable_custom_all_reduce_ << "\nmodel_name: " << model_name_
       << "\nmodel_dir: " << model_dir_ << "\nquant_policy: " << model_param_.quant_policy
       << "\ngroup_size: " << group_size_ << std::endl;

    return ss.str();
}

template<typename T>
void LlamaTritonModel<T>::createCustomComms(
    std::vector<std::shared_ptr<ft::AbstractCustomComm>>* custom_all_reduce_comms, int world_size)
{
    using commDataType = typename ft::CustomARCommTypeConverter<T>::Type;
    ft::initCustomAllReduceComm<commDataType>(custom_all_reduce_comms, enable_custom_all_reduce_, world_size);
}

template<typename T>
std::unique_ptr<ft::AbstractInstanceComm> LlamaTritonModel<T>::createInstanceComm(int size)
{
    return std::make_unique<ft::LlamaInstanceComm>(size);
}

template<typename T>
int LlamaTritonModel<T>::getTensorParaSize()
{
    return tensor_para_size_;
}

template<typename T>
int LlamaTritonModel<T>::getPipelineParaSize()
{
    return pipeline_para_size_;
}

#ifdef ENABLE_FP32
template struct LlamaTritonModel<float>;
#endif
template struct LlamaTritonModel<half>;
#ifdef ENABLE_BF16
template struct LlamaTritonModel<__nv_bfloat16>;
#endif
