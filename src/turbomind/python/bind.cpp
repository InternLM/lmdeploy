#include "src/turbomind/kernels/gemm_s_f16/format.h"
#include "src/turbomind/python/dlpack.h"
#include "src/turbomind/triton_backend/llama/LlamaTritonModel.h"
#include "src/turbomind/triton_backend/transformer_triton_backend.hpp"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/nccl_utils.h"
#include <cuda_runtime.h>
#include <memory>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
namespace ft = turbomind;
using namespace pybind11::literals;

// prepare to bind container
using TensorVector = std::vector<triton::Tensor>;
PYBIND11_MAKE_OPAQUE(TensorVector);
using TensorMap = std::unordered_map<std::string, triton::Tensor>;
PYBIND11_MAKE_OPAQUE(TensorMap);
static const char kDlTensorCapsuleName[] = "dltensor";

template<typename T>
std::shared_ptr<T> make_shared_nodel(T data)
{
    return std::shared_ptr<T>(&data, [](T*) {});
}

DLDevice getDLDevice(triton::Tensor& tensor)
{
    int device_id = 0;
    if (tensor.where == triton::MEMORY_GPU) {
        cudaPointerAttributes ptr_attr;
        cudaPointerGetAttributes(&ptr_attr, tensor.data);
        device_id = ptr_attr.device;
    }

    DLDevice device{kDLCPU, device_id};

    switch (tensor.where) {
        case triton::MEMORY_CPU:
            device.device_type = DLDeviceType::kDLCPU;
            break;
        case triton::MEMORY_CPU_PINNED:
            device.device_type = DLDeviceType::kDLCUDAHost;
        case triton::MEMORY_GPU:
            device.device_type = DLDeviceType::kDLCUDA;
            break;
        default:
            break;
    }

    return device;
}

std::unique_ptr<DLManagedTensor> TritonTensorToDLManagedTensor(triton::Tensor& tensor)
{
    DLDevice device = getDLDevice(tensor);

    DLDataType data_type{0, 0, 1};
    switch (tensor.type) {
        case triton::TYPE_BOOL:
            data_type.code = DLDataTypeCode::kDLBool;
            data_type.bits = 8;
            break;
        case triton::TYPE_UINT8:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 8;
            break;
        case triton::TYPE_UINT16:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 16;
            break;
        case triton::TYPE_UINT32:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 32;
            break;
        case triton::TYPE_UINT64:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 64;
            break;
        case triton::TYPE_INT8:
        case triton::TYPE_BYTES:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 8;
            break;
        case triton::TYPE_INT16:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 16;
            break;
        case triton::TYPE_INT32:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 32;
            break;
        case triton::TYPE_INT64:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 64;
            break;
        case triton::TYPE_FP16:
            data_type.code = DLDataTypeCode::kDLFloat;
            data_type.bits = 16;
            break;
        case triton::TYPE_FP32:
            data_type.code = DLDataTypeCode::kDLFloat;
            data_type.bits = 32;
            break;
        case triton::TYPE_FP64:
            data_type.code = DLDataTypeCode::kDLFloat;
            data_type.bits = 64;
            break;
        case triton::TYPE_BF16:
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

    return std::unique_ptr<DLManagedTensor>(new DLManagedTensor{dl_tensor, nullptr, [](DLManagedTensor*) {}});
}

triton::MemoryType getMemoryType(DLDevice device)
{
    switch (device.device_type) {
        case DLDeviceType::kDLCPU:
            return triton::MemoryType::MEMORY_CPU;
        case DLDeviceType::kDLCUDAHost:
            return triton::MemoryType::MEMORY_CPU_PINNED;
        case DLDeviceType::kDLCUDA:
            return triton::MemoryType::MEMORY_GPU;
        default:
            return triton::MemoryType::MEMORY_CPU;
    }
}

triton::DataType getDataType(DLDataType data_type)
{
    switch (data_type.code) {
        case DLDataTypeCode::kDLUInt:
            switch (data_type.bits) {
                case 8:
                    return triton::TYPE_UINT8;
                case 16:
                    return triton::TYPE_UINT16;
                case 32:
                    return triton::TYPE_UINT32;
                case 64:
                    return triton::TYPE_UINT64;
                default:
                    return triton::TYPE_INVALID;
            }
            break;
        case DLDataTypeCode::kDLInt:
            switch (data_type.bits) {
                case 8:
                    return triton::TYPE_INT8;
                case 16:
                    return triton::TYPE_INT16;
                case 32:
                    return triton::TYPE_INT32;
                case 64:
                    return triton::TYPE_INT64;
                default:
                    return triton::TYPE_INVALID;
            }
            break;
        case DLDataTypeCode::kDLFloat:
            switch (data_type.bits) {
                case 16:
                    return triton::TYPE_FP16;
                case 32:
                    return triton::TYPE_FP32;
                case 64:
                    return triton::TYPE_FP64;
                default:
                    return triton::TYPE_INVALID;
            }
            break;
        case DLDataTypeCode::kDLBfloat:
            switch (data_type.bits) {
                case 16:
                    return triton::TYPE_BF16;
                default:
                    return triton::TYPE_INVALID;
            }
            break;
        case DLDataTypeCode::kDLBool:
            return triton::TYPE_BOOL;
        default:
            return triton::TYPE_INVALID;
    }
}

std::shared_ptr<triton::Tensor> DLManagedTensorToTritonTensor(DLManagedTensor* tensor)
{
    auto& dl_tensor = tensor->dl_tensor;
    auto  where     = getMemoryType(dl_tensor.device);
    auto  dtype     = getDataType(dl_tensor.dtype);
    assert(dl_tensor.ndim > 0);
    std::vector<size_t> shape(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim);
    auto                data = dl_tensor.data;

    return std::make_shared<triton::Tensor>(where, dtype, shape, data);
}

DLTensor GetDLTensor(py::object obj)
{
    py::capsule      cap  = obj.attr("__dlpack__")();
    DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap.ptr(), kDlTensorCapsuleName));
    return dlmt->dl_tensor;
}

PYBIND11_MODULE(_turbomind, m)
{
    // nccl param
    py::class_<ft::NcclParam>(m, "NcclParam")
        .def(py::init<int, int>(), "rank"_a = 0, "world_size"_a = 1)
        .def("__str__", &ft::NcclParam::toString);

    // custom comm
    py::class_<ft::AbstractCustomComm, std::shared_ptr<ft::AbstractCustomComm>>(m, "AbstractCustomComm");

    // instance comm
    py::class_<ft::AbstractInstanceComm>(m, "AbstractInstanceComm");

    // data type
    py::enum_<triton::DataType>(m, "DataType")
        .value("TYPE_INVALID", triton::DataType::TYPE_INVALID)
        .value("TYPE_BOOL", triton::DataType::TYPE_BOOL)
        .value("TYPE_UINT8", triton::DataType::TYPE_UINT8)
        .value("TYPE_UINT16", triton::DataType::TYPE_UINT16)
        .value("TYPE_UINT32", triton::DataType::TYPE_UINT32)
        .value("TYPE_UINT64", triton::DataType::TYPE_UINT64)
        .value("TYPE_INT8", triton::DataType::TYPE_INT8)
        .value("TYPE_INT16", triton::DataType::TYPE_INT16)
        .value("TYPE_INT32", triton::DataType::TYPE_INT32)
        .value("TYPE_INT64", triton::DataType::TYPE_INT64)
        .value("TYPE_FP16", triton::DataType::TYPE_FP16)
        .value("TYPE_FP32", triton::DataType::TYPE_FP32)
        .value("TYPE_FP64", triton::DataType::TYPE_FP64)
        .value("TYPE_BYTES", triton::DataType::TYPE_BYTES)
        .value("TYPE_BF16", triton::DataType::TYPE_BF16);

    // memory type
    py::enum_<triton::MemoryType>(m, "MemoryType")
        .value("MEMORY_CPU", triton::MemoryType::MEMORY_CPU)
        .value("MEMORY_CPU_PINNED", triton::MemoryType::MEMORY_CPU_PINNED)
        .value("MEMORY_GPU", triton::MemoryType::MEMORY_GPU);

    // tensor
    py::class_<triton::Tensor, std::shared_ptr<triton::Tensor>>(m, "Tensor")
        .def_readonly("where", &triton::Tensor::where)
        .def_readonly("type", &triton::Tensor::type)
        .def_readonly("shape", &triton::Tensor::shape)
        .def_readonly("data", &triton::Tensor::data)
        .def(py::init([](const triton::MemoryType   where,
                         const triton::DataType     type,
                         const std::vector<size_t>& shape,
                         const long                 data) {
            auto data_ptr = reinterpret_cast<void*>(data);
            return new triton::Tensor(where, type, shape, data_ptr);
        }))
        .def(
            "view",
            [](triton::Tensor* self, triton::DataType new_type) {
                return new triton::Tensor(self->where, new_type, self->shape, self->data);
            },
            "new_type"_a)
        .def(
            "view",
            [](triton::Tensor* self, std::vector<size_t> new_shape) {
                return new triton::Tensor(self->where, self->type, new_shape, self->data);
            },
            "new_shape"_a)
        .def(
            "__dlpack__",
            [](triton::Tensor* self, long stream) {
                auto tensor_ptr = TritonTensorToDLManagedTensor(*self);
                return new py::capsule(tensor_ptr.release(), kDlTensorCapsuleName, [](PyObject* obj) {
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
        .def("__dlpack_device__", [](triton::Tensor* self) {
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
    py::bind_map<TensorMap, std::shared_ptr<TensorMap>>(m, "TensorMap");
    py::class_<AbstractTransformerModelInstance>(m, "AbstractTransformerModelInstance")
        .def(
            "forward",
            [](AbstractTransformerModelInstance* model,
               std::shared_ptr<TensorMap>        input_tensors,
               ft::AbstractInstanceComm*         inst_comm) { return model->forward(input_tensors, inst_comm); },
            py::call_guard<py::gil_scoped_release>(),
            "input_tensors"_a,
            "inst_comm"_a = nullptr)
        .def(
            "register_callback",
            [](AbstractTransformerModelInstance* self, triton_stream_cb_t cb, py::object ctx) {
                self->registerCallback(cb, ctx.ptr());
            },
            "callback"_a,
            "context"_a = nullptr)
        .def("unregister_callback", &AbstractTransformerModelInstance::unRegisterCallback);

    // transformer model
    py::class_<AbstractTransformerModel, std::shared_ptr<AbstractTransformerModel>>(m, "AbstractTransformerModel")
        .def_static(
            "create_llama_model",
            [](std::string model_dir,
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
                if (data_type == "half" || data_type == "fp16" || data_type == "int4") {
                    auto model = std::make_shared<LlamaTritonModel<half>>(
                        tensor_para_size, pipeline_para_size, enable_custom_all_reduce, model_dir);
                    model->setFfiLock(gil_control);
                    return model;
                }
                else {
                    auto model = std::make_shared<LlamaTritonModel<float>>(
                        tensor_para_size, pipeline_para_size, enable_custom_all_reduce, model_dir);
                    model->setFfiLock(gil_control);
                    return model;
                }
            },
            "model_dir"_a,
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
        .def("create_instance_comm", &AbstractTransformerModel::createInstanceComm, "size"_a)
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
        .def("__str__", &AbstractTransformerModel::toString)
        .def("__repr__", &AbstractTransformerModel::toString)
        .def("get_tensor_para_size", &AbstractTransformerModel::getTensorParaSize)
        .def("get_pipeline_para_size", &AbstractTransformerModel::getPipelineParaSize);

    m.def("transpose_qk_s4_k_m8", [](py::object src, py::object dst, int m, int k, int size_per_head) {
        auto src_tensor = GetDLTensor(src);
        auto dst_tensor = GetDLTensor(dst);

        turbomind::transpose_qk_s4_k_m8_hf(
            (uint32_t*)dst_tensor.data, (const uint32_t*)src_tensor.data, m, k, size_per_head, nullptr);
    });

    m.def("fuse_w1_w3_s4_k_m8", [](py::object src, py::object dst, int m, int k) {
        auto src_tensor = GetDLTensor(src);
        auto dst_tensor = GetDLTensor(dst);

        turbomind::fuse_w1_w3_s4_k_m8((uint32_t*)dst_tensor.data, (const uint32_t*)src_tensor.data, m, k, nullptr);
    });

    m.def("convert_s4_k_m8",
          [](py::object A_dst,
             py::object Q_dst,
             py::object ws,
             py::object A_src,
             py::object scales,
             py::object qzeros,
             int        m,
             int        k,
             int        group_size) {
              auto a_dst = GetDLTensor(A_dst);
              auto q_dst = GetDLTensor(Q_dst);
              auto w     = GetDLTensor(ws);
              auto a_src = GetDLTensor(A_src);
              auto s     = GetDLTensor(scales);
              auto qz    = GetDLTensor(qzeros);

              turbomind::convert_s4_k_m8((uint32_t*)a_dst.data,
                                         (half2*)q_dst.data,
                                         (half*)w.data,
                                         (const uint32_t*)a_src.data,
                                         (const half*)s.data,
                                         (const uint32_t*)qz.data,
                                         m,
                                         k,
                                         group_size,
                                         nullptr);
          });

    m.def("dequantize_s4", [](py::object src, py::object dst) {
        auto src_tensor = GetDLTensor(src);
        auto dst_tensor = GetDLTensor(dst);
        auto src_count  = std::accumulate(src_tensor.shape, src_tensor.shape + src_tensor.ndim, size_t{1});
        auto dst_count  = std::accumulate(dst_tensor.shape, dst_tensor.shape + dst_tensor.ndim, size_t{1});
        turbomind::FT_CHECK(src_count * 8 == dst_count);
        turbomind::dequantize_s4((uint4*)dst_tensor.data, (uint32_t*)src_tensor.data, src_count, nullptr);
    });
}
