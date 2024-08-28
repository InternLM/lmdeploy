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
#include "3rdparty/INIReader.h"
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

namespace ft = turbomind;

std::shared_ptr<AbstractTransformerModel> AbstractTransformerModel::createLlamaModel(std::string inifile)
{
    INIReader reader = INIReader(inifile);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << inifile << "'\n";
        return nullptr;
    }

    const std::string data_type        = reader.Get("ft_instance_hyperparameter", "data_type");
    int               tensor_para_size = reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size");
    std::string       model_dir        = reader.Get("ft_instance_hyperparameter", "model_dir");

    if (data_type == "half" || data_type == "fp16") {
        return std::make_shared<LlamaTritonModel<half>>(
            reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "enable_custom_all_reduce", 0),
            model_dir);
    }
    else if (data_type == "bf16") {
#ifdef ENABLE_BF16
        return std::make_shared<LlamaTritonModel<__nv_bfloat16>>(
            reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "enable_custom_all_reduce", 0),
            model_dir);
#else
        TM_LOG_ERROR("[ERROR] Turbomind is not built with ENABLE_BF16");
        ft::FT_CHECK(false);
#endif
    }
    else {
#ifdef ENABLE_FP32
        return std::make_shared<LlamaTritonModel<float>>(
            reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "enable_custom_all_reduce", 0),
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
    INIReader reader;
    FT_CHECK_WITH_INFO(!(config.empty() && model_dir.empty()), "invalid init options");

    if (!model_dir.empty()) {
        model_dir_ = model_dir;
        const std::string inifile{model_dir + "/config.ini"};
        reader = INIReader(inifile);
        if (reader.ParseError() < 0) {
            TM_LOG_ERROR("[ERROR] Can't load %s", inifile.c_str());
            ft::FT_CHECK(false);
        }
    }

    if (!config.empty()) {
        std::FILE* tmpf = std::tmpfile();
        std::fputs(config.c_str(), tmpf);
        std::rewind(tmpf);
        reader = INIReader(tmpf);
        if (reader.ParseError() < 0) {
            TM_LOG_ERROR("[ERROR] Can't init with config %s", config.c_str());
            ft::FT_CHECK(false);
        }
    }

    model_name_                     = reader.Get("llama", "model_name");
    model_param_.head_num           = reader.GetInteger("llama", "head_num");
    model_param_.head_dim           = reader.GetInteger("llama", "size_per_head");
    model_param_.kv_head_num        = reader.GetInteger("llama", "kv_head_num", 0);
    model_param_.hidden_units       = reader.GetInteger("llama", "hidden_units");
    model_param_.layer_num          = reader.GetInteger("llama", "num_layer");
    model_param_.inter_size         = reader.GetInteger("llama", "inter_size");
    model_param_.vocab_size         = reader.GetInteger("llama", "vocab_size");
    model_param_.norm_eps           = reader.GetFloat("llama", "norm_eps");
    model_param_.start_id           = reader.GetInteger("llama", "start_id");
    model_param_.end_id             = reader.GetInteger("llama", "end_id");
    attn_param_.cache_block_seq_len = reader.GetInteger("llama", "cache_block_seq_len", 0);
    model_param_.quant_policy       = reader.GetInteger("llama", "quant_policy", 0);

    // Only weight classes need these
    attn_bias_  = reader.GetInteger("llama", "attn_bias", 0);
    group_size_ = reader.GetInteger("llama", "group_size", 0);

    // rotary embedding parameters
    attn_param_.rotary_embedding_dim    = reader.GetInteger("llama", "rotary_embedding");
    attn_param_.rotary_embedding_base   = reader.GetFloat("llama", "rope_theta", 10000.0f);
    attn_param_.rope_scaling_type       = reader.Get("llama", "rope_scaling_type", "");
    attn_param_.rope_scaling_factor     = reader.GetFloat("llama", "rope_scaling_factor", 0.f);
    attn_param_.low_freq_factor         = reader.GetFloat("llama", "low_freq_factor", 1.0);
    attn_param_.high_freq_factor        = reader.GetFloat("llama", "high_freq_factor", 1.0);
    attn_param_.max_position_embeddings = reader.GetInteger("llama", "max_position_embeddings", 0);
    attn_param_.use_dynamic_ntk         = reader.GetInteger("llama", "use_dynamic_ntk", 0);
    attn_param_.use_logn_attn           = reader.GetInteger("llama", "use_logn_attn", 0);

    attn_param_.original_max_position_embeddings = reader.GetInteger("llama", "original_max_position_embeddings", 0);

    engine_param_.max_batch_size        = reader.GetInteger("llama", "max_batch_size", 0);
    engine_param_.max_prefill_token_num = reader.GetInteger("llama", "max_prefill_token_num", 0);
    engine_param_.max_context_token_num = reader.GetInteger("llama", "max_context_token_num", 0);
    engine_param_.session_len           = reader.GetInteger("llama", "session_len", 0);
    engine_param_.step_length           = reader.GetInteger("llama", "step_length", 0);

    engine_param_.cache_max_block_count = reader.GetFloat("llama", "cache_max_entry_count", 0);
    engine_param_.cache_chunk_size      = reader.GetInteger("llama", "cache_chunk_size", 0);
    engine_param_.enable_prefix_caching = reader.GetBoolean("llama", "enable_prefix_caching", false);

    engine_param_.num_tokens_per_iter = reader.GetInteger("llama", "num_tokens_per_iter", 0);
    engine_param_.max_prefill_iters   = reader.GetInteger("llama", "max_prefill_iters", 1);

    lora_param_.policy        = ft::getLoraPolicy(reader.Get("llama", "lora_policy", ""));
    lora_param_.r             = reader.GetInteger("llama", "lora_r", 0);
    lora_param_.scale         = reader.GetFloat("llama", "lora_scale", 0);
    lora_param_.max_wo_r      = reader.GetInteger("llama", "lora_max_wo_r", 0);
    lora_param_.rank_pattern  = getLoraPattern<int>(reader.Get("llama", "lora_rank_pattern", ""),
                                                   [](const std::string& s) { return std::stoi(s); });
    lora_param_.scale_pattern = getLoraPattern<float>(reader.Get("llama", "lora_scale_pattern", ""),
                                                      [](const std::string& s) { return std::stof(s); });
    handleMissingParams();

    shared_state_          = std::make_shared<ft::SharedState>();
    shared_state_->barrier = std::make_shared<ft::Barrier>(tensor_para_size);

    const auto device_count = ft::getDeviceCount();
    engines_.resize(device_count);

    const std::string weight_type_str = reader.Get("llama", "weight_type");
    if (weight_type_str == "fp16") {
        weight_type_ = ft::WeightType::kFP16;
    }
    else if (weight_type_str == "bf16") {
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
