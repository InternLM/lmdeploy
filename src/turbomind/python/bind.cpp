// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "src/turbomind/engine/model_request.h"
#include "src/turbomind/python/dlpack.h"
#include "src/turbomind/triton_backend/llama/LlamaTritonModel.h"
#include "src/turbomind/triton_backend/transformer_triton_backend.hpp"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/nccl_utils.h"

namespace py = pybind11;
namespace ft = turbomind;
using namespace pybind11::literals;

using ft::ManagedTensor;
using ft::Tensor;

// prepare to bind container
using TensorMap = std::unordered_map<std::string, ft::ManagedTensor>;
PYBIND11_MAKE_OPAQUE(TensorMap);
static const char kDlTensorCapsuleName[] = "dltensor";

DLDevice getDLDevice(const ft::Tensor& tensor)
{
    int device_id = 0;
    if (tensor.where == ft::MEMORY_GPU) {
        cudaPointerAttributes ptr_attr{};
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

DLManagedTensor* TritonTensorToDLManagedTensor(ManagedTensor& tensor)
{
    DLDevice device = getDLDevice(*tensor);

    DLDataType data_type{0, 0, 1};
    switch (tensor->type) {
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
    ManagedTensor* ctx = new ManagedTensor(tensor);
    DLTensor       dl_tensor{const_cast<void*>((*ctx)->data),
                       device,
                       (int32_t)((*ctx)->shape.size()),
                       data_type,
                       reinterpret_cast<int64_t*>(const_cast<size_t*>((*ctx)->shape.data())),
                       (int64_t*)(nullptr),
                       0};
    return new DLManagedTensor{dl_tensor, ctx, [](DLManagedTensor* dlmt) {  //
                                   //    auto&             x = *(ManagedTensor*)dlmt->manager_ctx;
                                   //    std::stringstream ss;
                                   //    ss << "(";
                                   //    for (const auto& d : x->shape) {
                                   //        ss << d << ",";
                                   //    }
                                   //    ss << ")";
                                   //    std::cerr << "turbomind tensor dtor " << ss.str() << " " << std::endl;
                                   delete (ManagedTensor*)dlmt->manager_ctx;
                                   delete dlmt;
                               }};
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

std::shared_ptr<ManagedTensor> DLManagedTensorToTritonTensor(DLManagedTensor* tensor)
{
    auto& dl_tensor = tensor->dl_tensor;
    auto  where     = getMemoryType(dl_tensor.device);
    auto  dtype     = getDataType(dl_tensor.dtype);
    assert(dl_tensor.ndim > 0);
    std::vector<size_t> shape(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim);
    auto                data = dl_tensor.data;

    auto ret    = std::make_shared<ManagedTensor>();
    ret->tensor = Tensor(where, dtype, std::move(shape), data);
    ret->data_holder.reset((void*)nullptr, [tensor](void*) {
        // std::cerr << "dlpack tensor dtor" << std::endl;
        if (tensor->deleter) {
            tensor->deleter(tensor);
        }
    });
    return ret;
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

namespace {

struct ScopedGIL {
    ScopedGIL(const ScopedGIL&) = delete;
    ScopedGIL& operator=(const ScopedGIL&) = delete;
    ScopedGIL(ScopedGIL&&)                 = delete;
    ScopedGIL& operator=(ScopedGIL&&) = delete;
    ScopedGIL()
    {
        state = PyGILState_Ensure();
    }
    ~ScopedGIL()
    {
        PyGILState_Release(state);
    }
    PyGILState_STATE state;
};

}  // namespace

PYBIND11_MODULE(_turbomind, m)
{
    // nccl param
    py::class_<ft::NcclParam>(m, "NcclParam")
        .def(py::init<int, int>(), "rank"_a = 0, "world_size"_a = 1)
        .def("__str__", &ft::NcclParam::toString);

    // custom comm
    (void)py::class_<ft::AbstractCustomComm, std::shared_ptr<ft::AbstractCustomComm>>(m, "AbstractCustomComm");

    py::class_<ft::SessionParam>(m, "SessionParam")
        .def(py::init([](uint64_t id, int step, bool start, bool end) {
                 if (!start && end) {
                     throw std::logic_error("unsupported arguments: start=false, end=true");
                 }
                 ft::SessionParam param{};
                 param.id         = id;
                 param.step       = step;
                 param.start_flag = start;
                 param.end_flag   = end;
                 return param;
             }),
             "id"_a,
             "step"_a,
             "start"_a,
             "end"_a)
        .def_readwrite("id", &ft::SessionParam::id)
        .def_readwrite("step", &ft::SessionParam::step)
        .def_readwrite("start", &ft::SessionParam::start_flag)
        .def_readwrite("end", &ft::SessionParam::end_flag);

    py::class_<ft::GenerationConfig>(m, "GenerationConfig")
        .def(py::init())
        .def_readwrite("max_new_tokens", &ft::GenerationConfig::max_new_tokens)
        .def_readwrite("min_new_tokens", &ft::GenerationConfig::min_new_tokens)
        .def_readwrite("eos_ids", &ft::GenerationConfig::eos_ids)
        .def_readwrite("stop_ids", &ft::GenerationConfig::stop_ids)
        .def_readwrite("bad_ids", &ft::GenerationConfig::bad_ids)
        .def_readwrite("top_p", &ft::GenerationConfig::top_p)
        .def_readwrite("top_k", &ft::GenerationConfig::top_k)
        .def_readwrite("min_p", &ft::GenerationConfig::min_p)
        .def_readwrite("temperature", &ft::GenerationConfig::temperature)
        .def_readwrite("repetition_penalty", &ft::GenerationConfig::repetition_penalty)
        .def_readwrite("random_seed", &ft::GenerationConfig::random_seed)
        .def_readwrite("output_logprobs", &ft::GenerationConfig::output_logprobs)
        .def_readwrite("output_last_hidden_state", &ft::GenerationConfig::output_last_hidden_state)
        .def_readwrite("output_logits", &ft::GenerationConfig::output_logits)
        .def("__repr__", [](const ft::GenerationConfig& c) {
            std::ostringstream oss;
            oss << c;
            return oss.str();
        });

    py::class_<ft::RequestState, std::unique_ptr<ft::RequestState>>(m, "RequestState")
        .def_readonly("status", &ft::RequestState::status)
        .def_readonly("seq_len", &ft::RequestState::seq_len);

    py::class_<ft::AtomicRequestState, std::shared_ptr<ft::AtomicRequestState>>(m, "AtomicRequestState")
        .def("consume", [](ft::AtomicRequestState& s) { return s.exchange(nullptr); });

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
    py::class_<ManagedTensor, std::shared_ptr<ManagedTensor>>(m, "Tensor")
        .def_property_readonly("where", [](const ManagedTensor& t) { return t->where; })
        .def_property_readonly("type", [](const ManagedTensor& t) { return t->type; })
        .def_property_readonly("shape", [](const ManagedTensor& t) { return t->shape; })
        .def_property_readonly("data", [](const ManagedTensor& t) { return t->data; })
        .def(
            "view",
            [](const ManagedTensor& self, ft::DataType new_type) {
                auto x  = self;
                x->type = new_type;
                return std::make_shared<ManagedTensor>(std::move(x));
            },
            "new_type"_a)
        .def(
            "view",
            [](const ManagedTensor& self, std::vector<size_t> new_shape) {
                auto x   = self;
                x->shape = new_shape;
                return std::make_shared<ManagedTensor>(std::move(x));
            },
            "new_shape"_a)
        .def(
            "copy_from",
            [](ManagedTensor& self, py::object obj) {
                py::capsule      cap = obj.attr("__dlpack__")();
                DLManagedTensor* dlmt =
                    static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap.ptr(), kDlTensorCapsuleName));
                auto src = DLManagedTensorToTritonTensor(dlmt);
                // take ownership of capsule's payload
                cap.set_name("used_dltensor");
                switch (self->type) {
                    case ft::TYPE_FP16:
                    case ft::TYPE_FP32:
                    case ft::TYPE_INT32:
                    case ft::TYPE_BF16: {
                        auto num_element = std::accumulate(
                            (*src)->shape.begin(), (*src)->shape.end(), 1LL, std::multiplies<int64_t>());
                        auto num_bytes = num_element * dlmt->dl_tensor.dtype.bits / 8;
                        ft::FT_CHECK(self->shape.size() == 1 && num_bytes == self->shape[0]);
                        safe_memcpy(const_cast<void*>(self->data), (*src)->data, num_bytes);
                        break;
                    }
                    default:
                        ft::FT_CHECK(0);
                }
            },
            "tensor"_a)
        .def(
            "__dlpack__",
            [](ManagedTensor& self, long stream) {
                DLManagedTensor* dlmt = TritonTensorToDLManagedTensor(self);
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
        .def("__dlpack_device__", [](const ManagedTensor& self) {
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
            // take ownership of capsule's payload
            cap.set_name("used_dltensor");
            return ret;
        },
        "dl_managed_tensor"_a);

    // transformer model instance
    using ft::ModelRequest;
    py::bind_map<TensorMap, std::shared_ptr<TensorMap>>(m, "TensorMap");
    py::class_<ModelRequest>(m, "ModelRequest")
        .def(
            "forward",
            [](ModelRequest*               model_request,
               std::shared_ptr<TensorMap>  input_tensors,
               const ft::SessionParam&     session,
               const ft::GenerationConfig& gen_cfg,
               bool                        stream_output,
               std::function<void()>       cb) {
                ModelRequest::InputParam param{};
                param.tensors       = std::move(input_tensors);
                param.session       = session;
                param.gen_cfg       = gen_cfg;
                param.stream_output = stream_output;
                auto ret            = model_request->Forward(std::move(param), [cb = std::move(cb)]() {
                    try {
                        cb();
                    }
                    catch (const py::error_already_set& e) {
                        std::cerr << e.what() << std::endl;
                    }
                });
                return std::make_tuple(std::move(ret.tensors), std::move(ret.state));
            },
            py::call_guard<py::gil_scoped_release>(),
            "input_tensors"_a,
            "session"_a,
            "gen_cfg"_a,
            "stream_output"_a,
            "cb"_a)
        .def(
            "cancel",
            [](ModelRequest* model_request) {
                model_request->Cancel();  //
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "end",
            [](ModelRequest* model_request, std::function<void(int)> cb, uint64_t session_id) {
                model_request->End(std::move(cb), session_id);  //
            },
            py::call_guard<py::gil_scoped_release>(),
            "cb"_a,
            "session_id"_a);

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
                auto gil_factory = [] {  //
                    // erase the type
                    return std::static_pointer_cast<void>(std::make_shared<ScopedGIL>());
                };
                auto no_gil_deleter = [](AbstractTransformerModel* ptr) {
                    pybind11::gil_scoped_release release;
                    delete ptr;
                };

                if (data_type == "half" || data_type == "fp16" || data_type == "float16" || data_type == "int4") {
                    std::shared_ptr<LlamaTritonModel<half>> model(new LlamaTritonModel<half>(tensor_para_size,
                                                                                             pipeline_para_size,
                                                                                             enable_custom_all_reduce,
                                                                                             model_dir,
                                                                                             config,
                                                                                             gil_factory),
                                                                  no_gil_deleter);
                    return model;
                }
                else if (data_type == "bf16" || data_type == "bfloat16") {
#ifdef ENABLE_BF16
                    std::shared_ptr<LlamaTritonModel<__nv_bfloat16>> model(
                        new LlamaTritonModel<__nv_bfloat16>(tensor_para_size,
                                                            pipeline_para_size,
                                                            enable_custom_all_reduce,
                                                            model_dir,
                                                            config,
                                                            gil_factory),
                        no_gil_deleter);
                    return model;
#else
                    throw std::runtime_error("Error: turbomind has not been built with bf16 support.");
#endif
                }
                else {
#ifdef ENABLE_FP32
                    auto model = std::make_shared<LlamaTritonModel<float>>(
                        tensor_para_size, pipeline_para_size, enable_custom_all_reduce, model_dir, config, gil_factory);
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
        .def("destroy_nccl_params", &AbstractTransformerModel::destroyNcclParams, "params"_a)
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
            [](AbstractTransformerModel* model, int deviceId) { return model->createModelInstance(deviceId); },
            py::call_guard<py::gil_scoped_release>(),
            "device_id"_a)
        .def("create_shared_weights",
             &AbstractTransformerModel::createSharedWeights,
             py::call_guard<py::gil_scoped_release>(),
             "device_id"_a,
             "rank"_a)
        .def(
            "get_params",
            [](AbstractTransformerModel* model, int deviceId, int rank) {
                auto      output = model->getParams(deviceId, rank);
                TensorMap ret;
                for (const auto& [k, v] : output) {
                    // export reference to weight data only (no ownership)
                    ret.emplace(k, ManagedTensor{v});
                }
                return ret;
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
