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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/triton_backend/multi_gpu_gpt/ParallelGptTritonModel.cc

#include "src/turbomind/triton_backend/llama/LlamaTritonModel.h"
#include "3rdparty/INIReader.h"
#include "src/turbomind/models/llama/LlamaInstanceComm.h"
#include "src/turbomind/triton_backend/llama/LlamaTritonModelInstance.h"
#include "src/turbomind/triton_backend/transformer_triton_backend.hpp"
#include "src/turbomind/utils/allocator.h"
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
    }else if (data_type == "bf16") {
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
        return std::make_shared<LlamaTritonModel<float>>(
            reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size"),
            reader.GetInteger("ft_instance_hyperparameter", "enable_custom_all_reduce", 0),
            model_dir);
    }
}

template<typename T>
void LlamaTritonModel<T>::handleMissingParams()
{
    if (kv_head_num_ == 0) {
        kv_head_num_ = head_num_;
        TM_LOG_WARNING("[LlamaTritonModel] `kv_head_num` is not set, default to `head_num` (%d).", (int)kv_head_num_);
    }

    if (!attn_params_.max_position_embeddings) {
        attn_params_.max_position_embeddings = 2048;
        TM_LOG_WARNING("[LlamaTritonModel] `max_position_embeddings` is not set, default to %d.",
                       (int)attn_params_.max_position_embeddings);
    }

    if (!engine_params_.max_batch_size) {
        engine_params_.max_batch_size = 64;
        TM_LOG_WARNING("[LlamaTritonModel] `max_batch_size` is not set, default to %d.",
                       (int)engine_params_.max_batch_size);
    }

    if (!engine_params_.session_len) {
        engine_params_.session_len = attn_params_.max_position_embeddings;
        TM_LOG_WARNING("[LlamaTritonModel] `session_len` is not set, default to %d.", (int)engine_params_.session_len);
    }

    if (!engine_params_.max_context_token_num) {
        engine_params_.max_context_token_num = engine_params_.session_len;
        TM_LOG_WARNING("[LlamaTritonModel] `max_context_token_num` is not set, default to %d.",
                       (int)engine_params_.max_context_token_num);
    }

    if (engine_params_.max_context_token_num <= engine_params_.max_batch_size) {
        engine_params_.max_context_token_num *= engine_params_.session_len;
        TM_LOG_WARNING("[LlamaTritonModel] `max_context_token_num` = %d.", (int)engine_params_.max_context_token_num);
    }

    if (!engine_params_.step_length) {
        engine_params_.step_length = 1;
    }

    if (!engine_params_.cache_max_block_count) {
        engine_params_.cache_max_block_count = .95f;
        TM_LOG_WARNING("[LlamaTritonModel] `cache_max_entry_count` is not set, default to %f.",
                       engine_params_.cache_max_block_count);
    }

    if (!cache_block_seq_len_) {
        cache_block_seq_len_ = 128;
        TM_LOG_WARNING("[LlamaTritonModel] `cache_block_seq_len` is not set, default to %d.", cache_block_seq_len_);
    }

    if (!engine_params_.cache_chunk_size) {
        engine_params_.cache_chunk_size = engine_params_.cache_max_block_count;
        TM_LOG_WARNING("[LlamaTritonModel] `cache_chunk_size` is not set, default to %d.",
                       (int)engine_params_.cache_chunk_size);
    }

    if (!engine_params_.num_tokens_per_iter) {
        engine_params_.num_tokens_per_iter = engine_params_.max_context_token_num;
        TM_LOG_WARNING("[LlamaTritonModel] `num_tokens_per_iter` is not set, default to `max_context_token_num` (%d).",
                       (int)engine_params_.num_tokens_per_iter);
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
    shared_weights_(std::vector<std::shared_ptr<ft::LlamaWeight<T>>>(ft::getDeviceCount())),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    INIReader reader;
    FT_CHECK_WITH_INFO((config.empty() ^ model_dir.empty()), "invalid init options");

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

    if (!model_dir.empty()) {
        model_dir_ = model_dir;
        const std::string inifile{model_dir + "/config.ini"};
        reader = INIReader(inifile);
        if (reader.ParseError() < 0) {
            TM_LOG_ERROR("[ERROR] Can't load %s", inifile.c_str());
            ft::FT_CHECK(false);
        }
    }

    model_name_          = reader.Get("llama", "model_name");
    head_num_            = reader.GetInteger("llama", "head_num");
    kv_head_num_         = reader.GetInteger("llama", "kv_head_num", 0);
    size_per_head_       = reader.GetInteger("llama", "size_per_head");
    inter_size_          = reader.GetInteger("llama", "inter_size");
    num_layer_           = reader.GetInteger("llama", "num_layer");
    vocab_size_          = reader.GetInteger("llama", "vocab_size");
    norm_eps_            = reader.GetFloat("llama", "norm_eps");
    start_id_            = reader.GetInteger("llama", "start_id");
    end_id_              = reader.GetInteger("llama", "end_id");
    use_context_fmha_    = reader.GetInteger("llama", "use_context_fmha", 1);
    cache_block_seq_len_ = reader.GetInteger("llama", "cache_block_seq_len", 0);

    attn_bias_    = reader.GetInteger("llama", "attn_bias", 0);
    quant_policy_ = reader.GetInteger("llama", "quant_policy", 0);
    group_size_   = reader.GetInteger("llama", "group_size", 0);

    // rotary embedding parameters
    attn_params_.rotary_embedding_dim    = reader.GetInteger("llama", "rotary_embedding");
    attn_params_.rotary_embedding_base   = reader.GetFloat("llama", "rope_theta", 10000.0f);
    attn_params_.rope_scaling_factor     = reader.GetFloat("llama", "rope_scaling_factor", 0.f);
    attn_params_.max_position_embeddings = reader.GetInteger("llama", "max_position_embeddings", 0);
    // attn_params_.use_dynamic_ntk         = reader.GetInteger("llama", "use_dynamic_ntk", 0);
    attn_params_.use_logn_attn = reader.GetInteger("llama", "use_logn_attn", 0);

    engine_params_.max_batch_size        = reader.GetInteger("llama", "max_batch_size", 0);
    engine_params_.max_context_token_num = reader.GetInteger("llama", "max_context_token_num", 0);
    engine_params_.session_len           = reader.GetInteger("llama", "session_len", 0);
    engine_params_.step_length           = reader.GetInteger("llama", "step_length", 0);

    engine_params_.cache_max_block_count = reader.GetFloat("llama", "cache_max_entry_count", 0);
    engine_params_.cache_chunk_size      = reader.GetInteger("llama", "cache_chunk_size", 0);

    engine_params_.num_tokens_per_iter   = reader.GetInteger("llama", "num_tokens_per_iter", 0);
    engine_params_.extra_tokens_per_iter = reader.GetInteger("llama", "extra_tokens_per_iter", 0);
    engine_params_.max_prefill_iters     = reader.GetInteger("llama", "max_prefill_iters", 1);

    handleMissingParams();

    shared_state_          = std::make_shared<typename ft::LlamaV2<T>::SharedState>();
    shared_state_->barrier = std::make_shared<ft::Barrier>(tensor_para_size);

    const auto device_count = ft::getDeviceCount();
    shared_instances_.resize(device_count);
    shared_mutexes_.resize(device_count);

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
}

template<typename T>
std::unique_ptr<LlamaTritonSharedModelInstance<T>> LlamaTritonModel<T>::createSharedModelInstance(
    int                                                               device_id,
    int                                                               rank,
    std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
    std::shared_ptr<ft::AbstractCustomComm>                           custom_all_reduce_comm)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    const int comms_rank = device_id % (tensor_para_size_ * pipeline_para_size_);

    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator(
        new ft::Allocator<ft::AllocatorType::CUDA>(device_id));

    /// TODO: this stream handle is leaked
    cudaStream_t stream{};
    ft::check_cuda_error(cudaStreamCreate(&stream));

    allocator->setStream(stream);

    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;

    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);

    std::unique_ptr<ft::cublasAlgoMap>   cublas_algo_map(new ft::cublasAlgoMap("gemm_config.in"));
    std::unique_ptr<std::mutex>          cublas_wrapper_mutex(new std::mutex());
    std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper(new ft::cublasMMWrapper(
        cublas_handle, cublaslt_handle, stream, cublas_algo_map.get(), cublas_wrapper_mutex.get(), allocator.get()));

    std::unique_ptr<cudaDeviceProp> cuda_device_prop_ptr(new cudaDeviceProp);
    ft::check_cuda_error(cudaGetDeviceProperties(cuda_device_prop_ptr.get(), device_id));

    if (std::is_same<T, half>::value) {
        cublas_wrapper->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper->setBF16GemmConfig();
    }
#endif

    ft::NcclParam tensor_para   = nccl_params.first[comms_rank];
    ft::NcclParam pipeline_para = nccl_params.second[comms_rank];

    ft::FT_CHECK(tensor_para.world_size_ == tensor_para_size_);
    ft::FT_CHECK(pipeline_para.world_size_ == pipeline_para_size_);

    auto llama = std::make_unique<ft::LlamaV2<T>>(head_num_,
                                                  kv_head_num_,
                                                  size_per_head_,
                                                  inter_size_,
                                                  num_layer_,
                                                  vocab_size_,
                                                  norm_eps_,
                                                  attn_params_,
                                                  start_id_,
                                                  end_id_,
                                                  cache_block_seq_len_,
                                                  quant_policy_,
                                                  use_context_fmha_,
                                                  engine_params_,
                                                  shared_state_,
                                                  shared_weights_[device_id].get(),
                                                  tensor_para,
                                                  stream,
                                                  cublas_wrapper.get(),
                                                  allocator.get(),
                                                  false,  // is_free_buffer_after_forward,
                                                  cuda_device_prop_ptr.get());

    return std::make_unique<LlamaTritonSharedModelInstance<T>>(
        LlamaTritonSharedModelInstance<T>{std::move(allocator),
                                          std::move(cublas_algo_map),
                                          std::move(cublas_wrapper_mutex),
                                          std::move(cublas_wrapper),
                                          std::move(cuda_device_prop_ptr),
                                          shared_weights_[device_id],
                                          std::move(llama),
                                          engine_params_.session_len});
}

template<typename T>
std::unique_ptr<AbstractTransformerModelInstance>
LlamaTritonModel<T>::createModelInstance(int                                                               device_id,
                                         int                                                               rank,
                                         cudaStream_t                                                      stream,
                                         std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
                                         std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    // const int comms_rank = device_id % (tensor_para_size_ * pipeline_para_size_);

    std::shared_ptr<LlamaTritonSharedModelInstance<T>> instance;
    {
        std::lock_guard<std::mutex> lock(shared_mutexes_[device_id]);
        instance = shared_instances_[device_id];
        if (!instance) {
            instance = createSharedModelInstance(device_id, rank, nccl_params, custom_all_reduce_comm);
            instance->llm->setFfiLock(ffi_lock_);
            shared_instances_[device_id] = instance;
        }
    }

    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator(
        new ft::Allocator<ft::AllocatorType::CUDA>(device_id));

    allocator->setStream(stream);

    return std::make_unique<LlamaTritonModelInstance<T>>(instance, std::move(allocator));
}

template<typename T>
void LlamaTritonModel<T>::createSharedWeights(int device_id, int rank)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    const int tensor_para_rank   = rank % tensor_para_size_;
    const int pipeline_para_rank = rank / tensor_para_size_;
    ft::FT_CHECK(pipeline_para_size_ == 1 && pipeline_para_rank == 0);
    shared_weights_[device_id] = std::make_shared<ft::LlamaWeight<T>>(head_num_,
                                                                      kv_head_num_,
                                                                      size_per_head_,
                                                                      inter_size_,
                                                                      vocab_size_,
                                                                      num_layer_,
                                                                      attn_bias_,
                                                                      weight_type_,
                                                                      group_size_,
                                                                      tensor_para_size_,
                                                                      tensor_para_rank);
    // model inited with model_dir
    if (model_dir_ != "") {
        shared_weights_[device_id]->loadModel(model_dir_);
    }
    return;
}

template<typename T>
TensorMap LlamaTritonModel<T>::getParams(int deviceId, int rank)
{
    ft::check_cuda_error(cudaSetDevice(deviceId));
    // shared_weight should be created before getParams
    ft::FT_CHECK(shared_weights_[deviceId] != nullptr);
    ft::TensorMap output = shared_weights_[deviceId]->getParams();
    TensorMap     result;
    for (auto [name, tensor] : output) {
        result.emplace(name, triton::Tensor{tensor.where, tensor.type, tensor.shape, tensor.data});
    }
    return result;
}

template<typename T>
std::string LlamaTritonModel<T>::toString()
{
    std::stringstream ss;
    ss << "Model: "
       << "\nhead_num: " << head_num_ << "\nkv_head_num: " << kv_head_num_ << "\nsize_per_head: " << size_per_head_
       << "\ninter_size: " << inter_size_ << "\nnum_layer: " << num_layer_ << "\nvocab_size: " << vocab_size_
       << "\nattn_bias: " << attn_bias_ << "\nmax_batch_size: " << engine_params_.max_batch_size
       << "\nmax_context_token_num: " << engine_params_.max_context_token_num
       << "\nsession_len: " << engine_params_.session_len << "\nstep_length: " << engine_params_.step_length
       << "\ncache_max_entry_count: " << engine_params_.cache_max_block_count
       << "\ncache_block_seq_len: " << cache_block_seq_len_ << "\ncache_chunk_size: " << engine_params_.cache_chunk_size
       << "\nuse_context_fmha: " << use_context_fmha_ << "\nstart_id: " << start_id_
       << "\ntensor_para_size: " << tensor_para_size_ << "\npipeline_para_size: " << pipeline_para_size_
       << "\nenable_custom_all_reduce: " << enable_custom_all_reduce_ << "\nmodel_name: " << model_name_
       << "\nmodel_dir: " << model_dir_ << "\nquant_policy: " << quant_policy_ << "\ngroup_size: " << group_size_
       << std::endl;

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
std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>>
LlamaTritonModel<T>::createNcclParams(const int node_id, const int device_id_start, const bool multi_node)
{
    const auto device_count     = ft::getDeviceCount();
    bool       need_nccl_params = false;
    // create nccl group when there are non-occupied devices
    for (int i = 0; i < device_count; ++i) {
        std::lock_guard<std::mutex> lock(shared_mutexes_[i]);
        if (shared_instances_[i] == nullptr) {
            need_nccl_params = true;
            break;
        }
    }
    if (need_nccl_params) {
        return AbstractTransformerModel::createNcclParams(node_id, device_id_start, multi_node);
    }
    else {
        TM_LOG_INFO("Skipping NCCL param creation.");

        const int tensor_para_size   = getTensorParaSize();
        const int pipeline_para_size = getPipelineParaSize();
        const int local_comm_size    = multi_node ? device_count : tensor_para_size * pipeline_para_size;

        std::vector<ft::NcclParam> tensor_para_params(local_comm_size);
        std::vector<ft::NcclParam> pipeline_para_params(local_comm_size);
        return {std::move(tensor_para_params), std::move(pipeline_para_params)};
    }
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

template struct LlamaTritonModel<float>;
template struct LlamaTritonModel<half>;
#ifdef ENABLE_BF16
template struct LlamaTritonModel<__nv_bfloat16>;
#endif
