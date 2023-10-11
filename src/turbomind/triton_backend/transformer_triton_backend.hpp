/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/triton_backend/transformer_triton_backend.hpp

#pragma once

#include <functional>
#include <memory>
#include <sstream>
#ifdef __linux__
#include <sys/time.h>
#endif
#include <vector>

#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/custom_ar_comm.h"
#include "src/turbomind/utils/instance_comm.h"
#include "src/turbomind/utils/nccl_utils.h"

namespace ft = turbomind;

namespace triton {
#ifdef USE_TRITONSERVER_DATATYPE

#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

#ifndef TRITONSERVER_API_VERSION_MAJOR
#error TRITONSERVER_API_VERSION_MAJOR Undefined!
#endif

#ifndef TRITONSERVER_API_VERSION_MINOR
#error TRITONSERVER_API_VERSION_MINOR Undefined!
#endif

#if (TRITONSERVER_API_VERSION_MAJOR == 1 && TRITONSERVER_API_VERSION_MINOR >= 17)                                      \
    || (TRITONSERVER_API_VERSION_MAJOR > 1)
#define ENABLE_TRITON_BF16 1
#endif

typedef TRITONSERVER_DataType   DataType;
typedef TRITONSERVER_MemoryType MemoryType;

constexpr TRITONSERVER_DataType TYPE_INVALID = TRITONSERVER_TYPE_INVALID;
constexpr TRITONSERVER_DataType TYPE_BOOL    = TRITONSERVER_TYPE_BOOL;
constexpr TRITONSERVER_DataType TYPE_UINT8   = TRITONSERVER_TYPE_UINT8;
constexpr TRITONSERVER_DataType TYPE_UINT16  = TRITONSERVER_TYPE_UINT16;
constexpr TRITONSERVER_DataType TYPE_UINT32  = TRITONSERVER_TYPE_UINT32;
constexpr TRITONSERVER_DataType TYPE_UINT64  = TRITONSERVER_TYPE_UINT64;
constexpr TRITONSERVER_DataType TYPE_INT8    = TRITONSERVER_TYPE_INT8;
constexpr TRITONSERVER_DataType TYPE_INT16   = TRITONSERVER_TYPE_INT16;
constexpr TRITONSERVER_DataType TYPE_INT32   = TRITONSERVER_TYPE_INT32;
constexpr TRITONSERVER_DataType TYPE_INT64   = TRITONSERVER_TYPE_INT64;
constexpr TRITONSERVER_DataType TYPE_FP16    = TRITONSERVER_TYPE_FP16;
constexpr TRITONSERVER_DataType TYPE_FP32    = TRITONSERVER_TYPE_FP32;
constexpr TRITONSERVER_DataType TYPE_FP64    = TRITONSERVER_TYPE_FP64;
constexpr TRITONSERVER_DataType TYPE_BYTES   = TRITONSERVER_TYPE_BYTES;

#ifdef ENABLE_TRITON_BF16
constexpr TRITONSERVER_DataType TYPE_BF16 = TRITONSERVER_TYPE_BF16;
#endif
constexpr TRITONSERVER_MemoryType MEMORY_CPU        = TRITONSERVER_MEMORY_CPU;
constexpr TRITONSERVER_MemoryType MEMORY_CPU_PINNED = TRITONSERVER_MEMORY_CPU_PINNED;
constexpr TRITONSERVER_MemoryType MEMORY_GPU        = TRITONSERVER_MEMORY_GPU;

#else

typedef ft::DataType   DataType;
typedef ft::MemoryType MemoryType;

constexpr DataType   TYPE_INVALID      = ft::TYPE_INVALID;
constexpr DataType   TYPE_BOOL         = ft::TYPE_BOOL;
constexpr DataType   TYPE_UINT8        = ft::TYPE_UINT8;
constexpr DataType   TYPE_UINT16       = ft::TYPE_UINT16;
constexpr DataType   TYPE_UINT32       = ft::TYPE_UINT32;
constexpr DataType   TYPE_UINT64       = ft::TYPE_UINT64;
constexpr DataType   TYPE_INT8         = ft::TYPE_INT8;
constexpr DataType   TYPE_INT16        = ft::TYPE_INT16;
constexpr DataType   TYPE_INT32        = ft::TYPE_INT32;
constexpr DataType   TYPE_INT64        = ft::TYPE_INT64;
constexpr DataType   TYPE_FP16         = ft::TYPE_FP16;
constexpr DataType   TYPE_FP32         = ft::TYPE_FP32;
constexpr DataType   TYPE_FP64         = ft::TYPE_FP64;
constexpr DataType   TYPE_BYTES        = ft::TYPE_BYTES;
constexpr DataType   TYPE_BF16         = ft::TYPE_BF16;
constexpr MemoryType MEMORY_CPU        = ft::MEMORY_CPU;
constexpr MemoryType MEMORY_CPU_PINNED = ft::MEMORY_CPU_PINNED;
constexpr MemoryType MEMORY_GPU        = ft::MEMORY_GPU;

#endif

struct Tensor {
    const MemoryType          where;
    const DataType            type;
    const std::vector<size_t> shape;
    const void*               data;

    Tensor(const MemoryType _where, const DataType _type, const std::vector<size_t> _shape, const void* _data):
        where(_where), type(_type), shape(_shape), data(_data)
    {
    }

    static ft::DataType convertTritonTypeToFt(DataType tmp_type)
    {
        ft::DataType ft_data_type;
        switch (tmp_type) {
            case TYPE_INVALID:
                ft_data_type = ft::DataType::TYPE_INVALID;
                break;
            case TYPE_BOOL:
                ft_data_type = ft::DataType::TYPE_BOOL;
                break;
            case TYPE_UINT8:
                ft_data_type = ft::DataType::TYPE_UINT8;
                break;
            case TYPE_UINT16:
                ft_data_type = ft::DataType::TYPE_UINT16;
                break;
            case TYPE_UINT32:
                ft_data_type = ft::DataType::TYPE_UINT32;
                break;
            case TYPE_UINT64:
                ft_data_type = ft::DataType::TYPE_UINT64;
                break;
            case TYPE_INT8:
                ft_data_type = ft::DataType::TYPE_INT8;
                break;
            case TYPE_INT16:
                ft_data_type = ft::DataType::TYPE_INT16;
                break;
            case TYPE_INT32:
                ft_data_type = ft::DataType::TYPE_INT32;
                break;
            case TYPE_INT64:
                ft_data_type = ft::DataType::TYPE_INT64;
                break;
            case TYPE_FP16:
                ft_data_type = ft::DataType::TYPE_FP16;
                break;
            case TYPE_FP32:
                ft_data_type = ft::DataType::TYPE_FP32;
                break;
            case TYPE_FP64:
                ft_data_type = ft::DataType::TYPE_FP64;
                break;
#ifdef ENABLE_TRITON_BF16
            case TYPE_BF16:
                ft_data_type = ft::DataType::TYPE_BF16;
                break;
#endif
            case TYPE_BYTES:
                ft_data_type = ft::DataType::TYPE_BYTES;
                break;
            default:
                FT_CHECK_WITH_INFO(false, "Unknown data type with type id: " + std::to_string(tmp_type));
                break;
        }
        return ft_data_type;
    }

    ft::Tensor convertTritonTensorToFt()
    {
        ft::DataType   ft_data_type = convertTritonTypeToFt(type);
        ft::MemoryType ft_memory_type;
        switch (where) {
            case MEMORY_CPU:
                ft_memory_type = ft::MemoryType::MEMORY_CPU;
                break;
            case MEMORY_CPU_PINNED:
                ft_memory_type = ft::MemoryType::MEMORY_CPU_PINNED;
                break;
            case MEMORY_GPU:
                ft_memory_type = ft::MemoryType::MEMORY_GPU;
                break;
        }
        return ft::Tensor{ft_memory_type, ft_data_type, shape, data};
    }

    static Tensor convertFtTensorToTriton(ft::Tensor ft_tensor)
    {
        DataType triton_data_type;
        switch (ft_tensor.type) {
            case TYPE_INVALID:
                triton_data_type = TYPE_INVALID;
                break;
            case TYPE_BOOL:
                triton_data_type = TYPE_BOOL;
                break;
            case TYPE_UINT8:
                triton_data_type = TYPE_UINT8;
                break;
            case TYPE_UINT16:
                triton_data_type = TYPE_UINT16;
                break;
            case TYPE_UINT32:
                triton_data_type = TYPE_UINT32;
                break;
            case TYPE_UINT64:
                triton_data_type = TYPE_UINT64;
                break;
            case TYPE_INT8:
                triton_data_type = TYPE_INT8;
                break;
            case TYPE_INT16:
                triton_data_type = TYPE_INT16;
                break;
            case TYPE_INT32:
                triton_data_type = TYPE_INT32;
                break;
            case TYPE_INT64:
                triton_data_type = TYPE_INT64;
                break;
            case TYPE_FP16:
                triton_data_type = TYPE_FP16;
                break;
            case TYPE_FP32:
                triton_data_type = TYPE_FP32;
                break;
            case TYPE_FP64:
                triton_data_type = TYPE_FP64;
                break;
#ifdef ENABLE_TRITON_BF16
            case TYPE_BF16:
                triton_data_type = TYPE_BF16;
                break;
#endif
            case TYPE_BYTES:
                triton_data_type = TYPE_BYTES;
                break;
            default:
                FT_CHECK_WITH_INFO(false, "Unknown data type with type id: " + std::to_string(ft_tensor.type));
                break;
        }
        MemoryType triton_memory_type;
        switch (ft_tensor.where) {
            case MEMORY_CPU:
                triton_memory_type = MEMORY_CPU;
                break;
            case MEMORY_CPU_PINNED:
                triton_memory_type = MEMORY_CPU_PINNED;
                break;
            case MEMORY_GPU:
                triton_memory_type = MEMORY_GPU;
                break;
        }
        return Tensor{triton_memory_type, triton_data_type, ft_tensor.shape, ft_tensor.data};
    }
};

}  // namespace triton

using triton_stream_cb_t = std::function<void(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>, void*)>;

struct AbstractTransformerModel;
struct AbstractTransformerModelInstance;

struct AbstractTransformerModelInstance {
    virtual ~AbstractTransformerModelInstance() = default;
    virtual std::shared_ptr<std::vector<triton::Tensor>>
    forward(std::shared_ptr<std::vector<triton::Tensor>> input_tensors) = 0;

    virtual std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
    forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors) = 0;

    virtual std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
    forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors, ft::AbstractInstanceComm*)
    {
        return forward(input_tensors);
    }

    void registerCallback(triton_stream_cb_t cb, void* ctx)
    {
        stream_cb_  = cb;
        stream_ctx_ = ctx;
    }

    void unRegisterCallback()
    {
        stream_cb_  = nullptr;
        stream_ctx_ = nullptr;
    }

    triton_stream_cb_t stream_cb_  = nullptr;
    void*              stream_ctx_ = nullptr;
};

struct AbstractTransformerModel {
    static std::shared_ptr<AbstractTransformerModel> createLlamaModel(std::string model_dir);

    virtual std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>>
    createNcclParams(const int node_id, const int device_id_start = 0, const bool multi_node = false);

    virtual void createCustomComms(std::vector<std::shared_ptr<ft::AbstractCustomComm>>* custom_all_reduce_comms,
                                   int                                                   world_size) = 0;

    virtual std::unique_ptr<ft::AbstractInstanceComm> createInstanceComm(int size)
    {
        return nullptr;
    }

    virtual std::unique_ptr<AbstractTransformerModelInstance>
    createModelInstance(int                                                               deviceId,
                        int                                                               rank,
                        cudaStream_t                                                      stream,
                        std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
                        std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm = nullptr) = 0;

    virtual void createSharedWeights(int deviceId, int rank) = 0;

    virtual std::string toString()            = 0;
    virtual int         getTensorParaSize()   = 0;
    virtual int         getPipelineParaSize() = 0;
};
