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

    if (!max_batch_size_) {
        max_batch_size_ = 32;
        TM_LOG_WARNING("[LlamaTritonModel] `max_batch_size` is not set, default to %d.", (int)max_batch_size_);
    }

    if (!session_len_) {
        session_len_ = 2160;
        TM_LOG_WARNING("[LlamaTritonModel] `session_len` is not set, default to %d.", (int)session_len_);
    }

    if (!max_context_token_num_) {
        max_context_token_num_ = (int)std::sqrt(max_batch_size_);
        TM_LOG_WARNING("[LlamaTritonModel] `max_context_token_num` is not set, default to %d.",
                       (int)max_context_token_num_);
    }

    if (!step_length_) {
        step_length_ = 1;
        TM_LOG_WARNING("[LlamaTritonModel] `step_length` is not set, default to %d.", (int)step_length_);
    }

    if (!cache_max_entry_count_) {
        cache_max_entry_count_ = 32;
        TM_LOG_WARNING("[LlamaTritonModel] `cache_max_entry_count` is not set, default to %d.",
                       (int)cache_max_entry_count_);
    }

    if (!cache_chunk_size_) {
        cache_chunk_size_ = cache_max_entry_count_;
        TM_LOG_WARNING("[LlamaTritonModel] `cache_chunk_size` is not set, default to %d.", (int)cache_chunk_size_);
    }
}

template<typename T>
LlamaTritonModel<T>::LlamaTritonModel(size_t      tensor_para_size,
                                      size_t      pipeline_para_size,
                                      int         enable_custom_all_reduce,
                                      std::string model_dir):
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    shared_weights_(std::vector<std::shared_ptr<ft::LlamaWeight<T>>>(ft::getDeviceCount())),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    model_dir_ = model_dir;
    const std::string inifile{model_dir + "/config.ini"};
    INIReader         reader = INIReader(inifile);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << inifile << "'\n";
        ft::FT_CHECK(false);
    }

    model_name_            = reader.Get("llama", "model_name");
    head_num_              = reader.GetInteger("llama", "head_num");
    kv_head_num_           = reader.GetInteger("llama", "kv_head_num", 0);
    size_per_head_         = reader.GetInteger("llama", "size_per_head");
    inter_size_            = reader.GetInteger("llama", "inter_size");
    num_layer_             = reader.GetInteger("llama", "num_layer");
    vocab_size_            = reader.GetInteger("llama", "vocab_size");
    norm_eps_              = reader.GetFloat("llama", "norm_eps");
    start_id_              = reader.GetInteger("llama", "start_id");
    end_id_                = reader.GetInteger("llama", "end_id");
    max_batch_size_        = reader.GetInteger("llama", "max_batch_size", 0);
    max_context_token_num_ = reader.GetInteger("llama", "max_context_token_num", 0);
    session_len_           = reader.GetInteger("llama", "session_len", 0);
    step_length_           = reader.GetInteger("llama", "step_length", 0);
    cache_max_entry_count_ = reader.GetInteger("llama", "cache_max_entry_count", 0);
    use_context_fmha_      = reader.GetInteger("llama", "use_context_fmha", 1);
    cache_chunk_size_      = reader.GetInteger("llama", "cache_chunk_size", 0);
    attn_bias_             = reader.GetInteger("llama", "attn_bias", 0);
    quant_policy_          = reader.GetInteger("llama", "quant_policy", 0);
    group_size_            = reader.GetInteger("llama", "group_size", 0);

    attn_params_.rotray_embedding_dim    = reader.GetInteger("llama", "rotary_embedding");
    attn_params_.rotary_embedding_base   = reader.GetFloat("llama", "rope_theta", 10000.0f);
    attn_params_.max_position_embeddings = reader.GetInteger("llama", "max_position_embeddings", 0);
    attn_params_.use_dynamic_ntk         = reader.GetInteger("llama", "use_dynamic_ntk", 0);
    attn_params_.use_logn_attn           = reader.GetInteger("llama", "use_logn_attn", 0);

    handleMissingParams();

    if (max_context_token_num_ <= max_batch_size_) {
        max_context_token_num_ *= session_len_;
    }

    shared_state_          = std::make_shared<typename ft::LlamaV2<T>::SharedState>();
    shared_state_->barrier = std::make_shared<ft::Barrier>(tensor_para_size);

    const auto device_count = ft::getDeviceCount();
    shared_instances_.resize(device_count);
    shared_mutexes_.resize(device_count);

    const std::string weight_type_str = reader.Get("llama", "weight_type");
    if (weight_type_str == "fp16") {
        weight_type_ = ft::WeightType::kFP16;
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

    ft::NcclParam tensor_para   = nccl_params.first[comms_rank];
    ft::NcclParam pipeline_para = nccl_params.second[comms_rank];

    ft::FT_CHECK(tensor_para.world_size_ == tensor_para_size_);
    ft::FT_CHECK(pipeline_para.world_size_ = pipeline_para_size_);

    auto llama = std::make_unique<ft::LlamaV2<T>>(head_num_,
                                                  kv_head_num_,
                                                  size_per_head_,
                                                  inter_size_,
                                                  num_layer_,
                                                  vocab_size_,
                                                  attn_params_,
                                                  norm_eps_,
                                                  max_batch_size_,
                                                  max_context_token_num_,
                                                  session_len_,
                                                  step_length_,
                                                  start_id_,
                                                  end_id_,
                                                  cache_max_entry_count_,
                                                  cache_chunk_size_,
                                                  quant_policy_,
                                                  use_context_fmha_,
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
                                          session_len_});
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
    shared_weights_[device_id]->loadModel(model_dir_);
    return;
}

template<typename T>
std::string LlamaTritonModel<T>::toString()
{
    std::stringstream ss;
    ss << "Model: "
       << "\nhead_num: " << head_num_ << "\nkv_head_num: " << kv_head_num_ << "\nsize_per_head: " << size_per_head_
       << "\ninter_size: " << inter_size_ << "\nnum_layer: " << num_layer_ << "\nvocab_size: " << vocab_size_
       << "\nattn_bias: " << attn_bias_ << "\nmax_batch_size: " << max_batch_size_
       << "\nmax_context_token_num: " << max_context_token_num_ << "\nsession_len: " << session_len_
       << "\nstep_length: " << step_length_ << "\ncache_max_entry_count: " << cache_max_entry_count_
       << "\ncache_chunk_size: " << cache_chunk_size_ << "\nuse_context_fmha: " << use_context_fmha_
       << "\nstart_id: " << start_id_ << "\ntensor_para_size: " << tensor_para_size_
       << "\npipeline_para_size: " << pipeline_para_size_ << "\nenable_custom_all_reduce: " << enable_custom_all_reduce_
       << "\nmodel_name: " << model_name_ << "\nmodel_dir: " << model_dir_ << "\nquant_policy: " << quant_policy_
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
