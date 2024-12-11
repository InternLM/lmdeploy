// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "src/turbomind/python/dlpack.h"
#include "src/turbomind/triton_backend/llama/LlamaTritonModel.h"
#include "src/turbomind/triton_backend/transformer_triton_backend.hpp"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/nccl_utils.h"

namespace py = pybind11;
namespace ft = turbomind;
using namespace pybind11::literals;

// prepare to bind container
using TensorVector = std::vector<ft::Tensor>;
PYBIND11_MAKE_OPAQUE(TensorVector);
using TensorMap = std::unordered_map<std::string, ft::Tensor>;
PYBIND11_MAKE_OPAQUE(TensorMap);
static const char kDlTensorCapsuleName[] = "dltensor";

DLDevice getDLDevice(ft::Tensor& tensor)
{
    int device_id = 0;
    if (tensor.where == ft::MEMORY_GPU) {
        cudaPointerAttributes ptr_attr;
        cudaPointerGetAttributes(&ptr_attr, tensor.data);
        device_id = ptr_attr.device;
    }

    DLDevice device{kDLCPU, device_id};

    switch (tensor.where) {
        case ft::MEMORY_CPU:
            device.device_type = DLDeviceType::kDLCPU;
            break;
        case ft::MEMORY_CPU_PINNED:
            device.device_type = DLDeviceType::kDLCUDAHost;
            break;
        case ft::MEMORY_GPU:
            device.device_type = DLDeviceType::kDLCUDA;
            break;
        default:
            break;
    }

    return device;
}

DLManagedTensor* TritonTensorToDLManagedTensor(ft::Tensor& tensor)
{
    DLDevice device = getDLDevice(tensor);

    DLDataType data_type{0, 0, 1};
    switch (tensor.type) {
        case ft::TYPE_BOOL:
            data_type.code = DLDataTypeCode::kDLBool;
            data_type.bits = 8;
            break;
        case ft::TYPE_UINT8:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 8;
            break;
        case ft::TYPE_UINT16:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 16;
            break;
        case ft::TYPE_UINT32:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 32;
            break;
        case ft::TYPE_UINT64:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 64;
            break;
        case ft::TYPE_INT8:
        case ft::TYPE_BYTES:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 8;
            break;
        case ft::TYPE_INT16:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 16;
            break;
        case ft::TYPE_INT32:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 32;
            break;
        case ft::TYPE_INT64:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 64;
            break;
        case ft::TYPE_FP16:
            data_type.code = DLDataTypeCode::kDLFloat;
            data_type.bits = 16;
            break;
        case ft::TYPE_FP32:
            data_type.code = DLDataTypeCode::kDLFloat;
            data_type.bits = 32;
            break;
        case ft::TYPE_FP64:
            data_type.code = DLDataTypeCode::kDLFloat;
            data_type.bits = 64;
            break;
        case ft::TYPE_BF16:
            data_type.code = DLDataTypeCode::kDLBfloat;
            data_type.bits = 16;
            break;
        default:
            break;
    }
    DLTensor dl_tensor{const_cast<void*>(tensor.data),
                       device,
                       (int32_t)(tensor.shape.size()),
                       data_type,
                       reinterpret_cast<int64_t*>(const_cast<size_t*>(tensor.shape.data())),
                       (int64_t*)(nullptr),
                       0};
    return new DLManagedTensor{dl_tensor, nullptr, [](DLManagedTensor* dlmt) { delete dlmt; }};
}

ft::MemoryType getMemoryType(DLDevice device)
{
    switch (device.device_type) {
        case DLDeviceType::kDLCUDAHost:
            return ft::MemoryType::MEMORY_CPU_PINNED;
        case DLDeviceType::kDLCUDA:
            return ft::MemoryType::MEMORY_GPU;
        case DLDeviceType::kDLCPU:
        default:
            return ft::MemoryType::MEMORY_CPU;
    }
}

ft::DataType getDataType(DLDataType data_type)
{
    switch (data_type.code) {
        case DLDataTypeCode::kDLUInt:
            switch (data_type.bits) {
                case 8:
                    return ft::TYPE_UINT8;
                case 16:
                    return ft::TYPE_UINT16;
                case 32:
                    return ft::TYPE_UINT32;
                case 64:
                    return ft::TYPE_UINT64;
                default:
                    return ft::TYPE_INVALID;
            }
            break;
        case DLDataTypeCode::kDLInt:
            switch (data_type.bits) {
                case 8:
                    return ft::TYPE_INT8;
                case 16:
                    return ft::TYPE_INT16;
                case 32:
                    return ft::TYPE_INT32;
                case 64:
                    return ft::TYPE_INT64;
                default:
                    return ft::TYPE_INVALID;
            }
            break;
        case DLDataTypeCode::kDLFloat:
            switch (data_type.bits) {
                case 16:
                    return ft::TYPE_FP16;
                case 32:
                    return ft::TYPE_FP32;
                case 64:
                    return ft::TYPE_FP64;
                default:
                    return ft::TYPE_INVALID;
            }
            break;
        case DLDataTypeCode::kDLBfloat:
            switch (data_type.bits) {
                case 16:
                    return ft::TYPE_BF16;
                default:
                    return ft::TYPE_INVALID;
            }
            break;
        case DLDataTypeCode::kDLBool:
            return ft::TYPE_BOOL;
        default:
            return ft::TYPE_INVALID;
    }
}

std::shared_ptr<ft::Tensor> DLManagedTensorToTritonTensor(DLManagedTensor* tensor)
{
    auto& dl_tensor = tensor->dl_tensor;
    auto  where     = getMemoryType(dl_tensor.device);
    auto  dtype     = getDataType(dl_tensor.dtype);
    assert(dl_tensor.ndim > 0);
    std::vector<size_t> shape(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim);
    auto                data = dl_tensor.data;

    return std::make_shared<ft::Tensor>(where, dtype, shape, data);
}

DLTensor GetDLTensor(py::object obj)
{
    py::capsule      cap  = obj.attr("__dlpack__")();
    DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap.ptr(), kDlTensorCapsuleName));
    return dlmt->dl_tensor;
}

static void safe_memcpy(void* dst, const void* src, size_t size)
{
    cudaPointerAttributes dat{};
    cudaPointerAttributes sat{};
    ft::check_cuda_error(cudaPointerGetAttributes(&dat, dst));
    ft::check_cuda_error(cudaPointerGetAttributes(&sat, src));
    try {
        if (dat.devicePointer && sat.devicePointer) {
            // Both can be accessed from current context
            ft::check_cuda_error(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
        }
        else if (dat.type == cudaMemoryTypeDevice && sat.type == cudaMemoryTypeDevice) {
            if (dat.device != sat.device) {
                // On different devices, try peer memcpy
                ft::check_cuda_error(cudaMemcpyPeer(dst, dat.device, src, sat.device, size));
            }
            else {
                // Same device, switch to the device first (this is unlikely)
                ft::CudaDeviceGuard guard(dat.device);
                ft::check_cuda_error(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
            }
        }
        else {
            // Unknown case, give it a try anyway
            ft::check_cuda_error(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
        }
    }
    catch (...) {
        int device_id{-1};
        cudaGetDevice(&device_id);
        TM_LOG_ERROR("cudaMemcpy failed: dst=(%d, %d, %p, %p), src=(%d, %d, %p, %p), size=%s, device=%d",
                     (int)dat.type,
                     dat.device,
                     dat.devicePointer,
                     dat.hostPointer,
                     (int)sat.type,
                     sat.device,
                     sat.devicePointer,
                     sat.hostPointer,
                     std::to_string(size).c_str(),
                     device_id);
        throw;
    }
}

PYBIND11_MODULE(_turbomind, m)
{
    // nccl param
    py::class_<ft::NcclParam>(m, "NcclParam")
        .def(py::init<int, int>(), "rank"_a = 0, "world_size"_a = 1)
        .def("__str__", &ft::NcclParam::toString);

    // custom comm
    py::class_<ft::AbstractCustomComm, std::shared_ptr<ft::AbstractCustomComm>>(m, "AbstractCustomComm");

    // data type
    py::enum_<ft::DataType>(m, "DataType")
        .value("TYPE_INVALID", ft::DataType::TYPE_INVALID)
        .value("TYPE_BOOL", ft::DataType::TYPE_BOOL)
        .value("TYPE_UINT8", ft::DataType::TYPE_UINT8)
        .value("TYPE_UINT16", ft::DataType::TYPE_UINT16)
        .value("TYPE_UINT32", ft::DataType::TYPE_UINT32)
        .value("TYPE_UINT64", ft::DataType::TYPE_UINT64)
        .value("TYPE_INT8", ft::DataType::TYPE_INT8)
        .value("TYPE_INT16", ft::DataType::TYPE_INT16)
        .value("TYPE_INT32", ft::DataType::TYPE_INT32)
        .value("TYPE_INT64", ft::DataType::TYPE_INT64)
        .value("TYPE_FP16", ft::DataType::TYPE_FP16)
        .value("TYPE_FP32", ft::DataType::TYPE_FP32)
        .value("TYPE_FP64", ft::DataType::TYPE_FP64)
        .value("TYPE_BYTES", ft::DataType::TYPE_BYTES)
        .value("TYPE_BF16", ft::DataType::TYPE_BF16);

    // memory type
    py::enum_<ft::MemoryType>(m, "MemoryType")
        .value("MEMORY_CPU", ft::MemoryType::MEMORY_CPU)
        .value("MEMORY_CPU_PINNED", ft::MemoryType::MEMORY_CPU_PINNED)
        .value("MEMORY_GPU", ft::MemoryType::MEMORY_GPU);

    // tensor
    py::class_<ft::Tensor, std::shared_ptr<ft::Tensor>>(m, "Tensor")
        .def_readonly("where", &ft::Tensor::where)
        .def_readonly("type", &ft::Tensor::type)
        .def_readonly("shape", &ft::Tensor::shape)
        .def_readonly("data", &ft::Tensor::data)
        .def(py::init(
            [](const ft::MemoryType where, const ft::DataType type, const std::vector<size_t>& shape, const long data) {
                auto data_ptr = reinterpret_cast<void*>(data);
                return new ft::Tensor(where, type, shape, data_ptr);
            }))
        .def(
            "view",
            [](ft::Tensor* self, ft::DataType new_type) {
                return new ft::Tensor(self->where, new_type, self->shape, self->data);
            },
            "new_type"_a)
        .def(
            "view",
            [](ft::Tensor* self, std::vector<size_t> new_shape) {
                return new ft::Tensor(self->where, self->type, new_shape, self->data);
            },
            "new_shape"_a)
        .def(
            "copy_from",
            [](ft::Tensor* self, py::object obj) {
                py::capsule      cap = obj.attr("__dlpack__")();
                DLManagedTensor* dlmt =
                    static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap.ptr(), kDlTensorCapsuleName));
                auto src = DLManagedTensorToTritonTensor(dlmt);
                switch (self->type) {
                    case ft::TYPE_FP16:
                    case ft::TYPE_FP32:
                    case ft::TYPE_INT32:
                    case ft::TYPE_BF16: {
                        auto num_element =
                            std::accumulate(src->shape.begin(), src->shape.end(), 1LL, std::multiplies<int64_t>());
                        auto num_bytes = num_element * dlmt->dl_tensor.dtype.bits / 8;
                        ft::FT_CHECK(self->shape.size() == 1 && num_bytes == self->shape[0]);
                        safe_memcpy(const_cast<void*>(self->data), src->data, num_bytes);
                        break;
                    }
                    default:
                        ft::FT_CHECK(0);
                }
            },
            "tensor"_a)
        .def(
            "__dlpack__",
            [](ft::Tensor* self, long stream) {
                DLManagedTensor* dlmt = TritonTensorToDLManagedTensor(*self);
                return py::capsule(dlmt, kDlTensorCapsuleName, [](PyObject* obj) {
                    DLManagedTensor* dlmt =
                        static_cast<DLManagedTensor*>(PyCapsule_GetPointer(obj, kDlTensorCapsuleName));
                    if (dlmt) {
                        dlmt->deleter(dlmt);
                    }
                    else {
                        // The tensor has been deleted. Clear any error from
                        // PyCapsule_GetPointer.
                        PyErr_Clear();
                    }
                });
            },
            "stream"_a = 0)
        .def("__dlpack_device__", [](ft::Tensor* self) {
            auto device = getDLDevice(*self);
            return std::tuple<int, int>(int(device.device_type), device.device_id);
        });
    m.def(
        "from_dlpack",
        [](py::object obj) {
            py::capsule      cap = obj.attr("__dlpack__")();
            DLManagedTensor* dlmt =
                static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap.ptr(), kDlTensorCapsuleName));
            auto ret = DLManagedTensorToTritonTensor(dlmt);
            return ret;
        },
        "dl_managed_tensor"_a);

    // transformer model instance
    using ft::AbstractTransformerModelInstance;
    py::bind_map<TensorMap, std::shared_ptr<TensorMap>>(m, "TensorMap");
    py::class_<AbstractTransformerModelInstance>(m, "AbstractTransformerModelInstance")
        .def(
            "forward",
            [](AbstractTransformerModelInstance* model, std::shared_ptr<TensorMap> input_tensors) {
                return model->forward(input_tensors);
            },
            py::call_guard<py::gil_scoped_release>(),
            "input_tensors"_a)
        .def(
            "register_callback",
            [](AbstractTransformerModelInstance* self, ft::triton_stream_cb_t cb, py::object ctx) {
                self->registerCallback(cb, ctx.ptr());
            },
            "callback"_a,
            "context"_a = nullptr)
        .def("unregister_callback", &AbstractTransformerModelInstance::unRegisterCallback);

    // transformer model
    using ft::AbstractTransformerModel;
    using ft::LlamaTritonModel;
    py::class_<AbstractTransformerModel, std::shared_ptr<AbstractTransformerModel>>(m, "AbstractTransformerModel")
        .def_static(
            "create_llama_model",
            [](std::string model_dir,
               std::string config,
               size_t      tensor_para_size,
               size_t      pipeline_para_size,
               int         enable_custom_all_reduce,
               std::string data_type) -> std::shared_ptr<AbstractTransformerModel> {
                auto gil_control = [state = PyGILState_STATE{}](int op) mutable {
                    if (op) {
                        state = PyGILState_Ensure();
                    }
                    else {
                        PyGILState_Release(state);
                    }
                };
                if (data_type == "half" || data_type == "fp16" || data_type == "float16" || data_type == "int4") {
                    auto model = std::make_shared<LlamaTritonModel<half>>(
                        tensor_para_size, pipeline_para_size, enable_custom_all_reduce, model_dir, config);
                    model->set_ffi_lock(gil_control);
                    return model;
                }
                else if (data_type == "bf16" || data_type == "bfloat16") {
#ifdef ENABLE_BF16
                    auto model = std::make_shared<LlamaTritonModel<__nv_bfloat16>>(
                        tensor_para_size, pipeline_para_size, enable_custom_all_reduce, model_dir, config);
                    model->set_ffi_lock(gil_control);
                    return model;
#else
                    throw std::runtime_error("Error: turbomind has not been built with bf16 support.");
#endif
                }
                else {
#ifdef ENABLE_FP32
                    auto model = std::make_shared<LlamaTritonModel<float>>(
                        tensor_para_size, pipeline_para_size, enable_custom_all_reduce, model_dir, config);
                    model->set_ffi_lock(gil_control);
                    return model;
#else
                    throw std::runtime_error("Error: turbomind has not been built with fp32 support.");
#endif
                }
            },
            "model_dir"_a,
            "config"_a                   = "",
            "tensor_para_size"_a         = 1,
            "pipeline_para_size"_a       = 1,
            "enable_custom_all_reduce"_a = 0,
            "data_type"_a                = "half")
        .def("create_nccl_params",
             &AbstractTransformerModel::createNcclParams,
             "node_id"_a,
             "device_id_start"_a = 0,
             "multi_node"_a      = false)
        .def(
            "create_custom_comms",
            [](AbstractTransformerModel* model, int world_size) {
                std::vector<std::shared_ptr<ft::AbstractCustomComm>> ret;
                model->createCustomComms(&ret, world_size);
                return ret;
            },
            "world_size"_a)
        .def(
            "create_model_instance",
            [](AbstractTransformerModel*                                         model,
               int                                                               deviceId,
               int                                                               rank,
               long                                                              stream_id,
               std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
               std::shared_ptr<ft::AbstractCustomComm>                           custom_all_reduce_comm = nullptr) {
                cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
                return model->createModelInstance(deviceId, rank, stream, nccl_params, custom_all_reduce_comm);
            },
            py::call_guard<py::gil_scoped_release>(),
            "device_id"_a,
            "rank"_a,
            "stream"_a,
            "nccl_params"_a,
            "custom_all_reduce_comm"_a = nullptr)
        .def("create_shared_weights",
             &AbstractTransformerModel::createSharedWeights,
             py::call_guard<py::gil_scoped_release>(),
             "device_id"_a,
             "rank"_a)
        .def(
            "get_params",
            [](AbstractTransformerModel* model, int deviceId, int rank) {
                TensorMap output = model->getParams(deviceId, rank);
                return output;
            },
            py::call_guard<py::gil_scoped_release>(),
            "device_id"_a,
            "rank"_a)
        .def(
            "process_weight",
            [](AbstractTransformerModel* model, int deviceId, int rank) { model->processWeights(deviceId, rank); },
            py::call_guard<py::gil_scoped_release>(),
            "device_id"_a,
            "rank"_a)
        .def(
            "create_engine",
            [](AbstractTransformerModel*                                         model,
               int                                                               deviceId,
               int                                                               rank,
               std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
               std::shared_ptr<ft::AbstractCustomComm>                           custom_all_reduce_comm = nullptr) {
                model->createEngine(deviceId, rank, nccl_params, custom_all_reduce_comm);
            },
            py::call_guard<py::gil_scoped_release>(),
            "device_id"_a,
            "rank"_a,
            "nccl_params"_a,
            "custom_all_reduce_comm"_a = nullptr)
        .def("__str__", &AbstractTransformerModel::toString)
        .def("__repr__", &AbstractTransformerModel::toString)
        .def("get_tensor_para_size", &AbstractTransformerModel::getTensorParaSize)
        .def("get_pipeline_para_size", &AbstractTransformerModel::getPipelineParaSize);
}
