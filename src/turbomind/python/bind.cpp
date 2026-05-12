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

#include "xgrammar/compiler.h"

#include "src/turbomind/core/data_format.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/engine_config.h"
#include "src/turbomind/engine/model_request.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/moe_weight.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/python/dlpack.h"
#include "src/turbomind/turbomind.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/metrics.h"

namespace py = pybind11;
namespace ft = turbomind;
using namespace pybind11::literals;

using ft::core::Tensor;

// prepare to bind container
using TensorMap = ft::core::TensorMap;
PYBIND11_MAKE_OPAQUE(TensorMap);
static const char kDlTensorCapsuleName[] = "dltensor";

DLDevice getDLDevice(const Tensor& tensor)
{
    int device_id = 0;
    if (tensor.device().type == ft::kDEVICE) {
        cudaPointerAttributes ptr_attr{};
        cudaPointerGetAttributes(&ptr_attr, tensor.raw_data());
        device_id = ptr_attr.device;
    }

    DLDevice device{kDLCPU, device_id};

    switch (tensor.device().type) {
        case ft::kCPU:
            device.device_type = DLDeviceType::kDLCPU;
            break;
        case ft::kCPUpinned:
            device.device_type = DLDeviceType::kDLCUDAHost;
            break;
        case ft::kDEVICE:
            device.device_type = DLDeviceType::kDLCUDA;
            break;
        default:
            break;
    }

    return device;
}

DLManagedTensor* TritonTensorToDLManagedTensor(Tensor& tensor)
{
    DLDevice   device = getDLDevice(tensor);
    DLDataType data_type{0, 0, 1};
    using ft::data_type_v;
    switch (tensor.dtype()) {
        case data_type_v<bool>:
            data_type.code = DLDataTypeCode::kDLBool;
            data_type.bits = 8;
            break;
        case data_type_v<uint8_t>:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 8;
            break;
        case data_type_v<uint16_t>:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 16;
            break;
        case data_type_v<uint32_t>:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 32;
            break;
        case data_type_v<uint64_t>:
            data_type.code = DLDataTypeCode::kDLUInt;
            data_type.bits = 64;
            break;
        case data_type_v<int8_t>:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 8;
            break;
        case data_type_v<int16_t>:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 16;
            break;
        case data_type_v<int32_t>:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 32;
            break;
        case data_type_v<int64_t>:
            data_type.code = DLDataTypeCode::kDLInt;
            data_type.bits = 64;
            break;
        case data_type_v<turbomind::half_t>:
            data_type.code = DLDataTypeCode::kDLFloat;
            data_type.bits = 16;
            break;
        case data_type_v<float>:
            data_type.code = DLDataTypeCode::kDLFloat;
            data_type.bits = 32;
            break;
        case data_type_v<double>:
            data_type.code = DLDataTypeCode::kDLFloat;
            data_type.bits = 64;
            break;
        case data_type_v<turbomind::bfloat16_t>:
            data_type.code = DLDataTypeCode::kDLBfloat;
            data_type.bits = 16;
            break;
        default:
            break;
    }

    static_assert(sizeof(int64_t) == sizeof(tensor.shape(0)));

    Tensor*  ctx = new Tensor(tensor);
    DLTensor dl_tensor{const_cast<void*>(ctx->raw_data()),
                       device,
                       (int32_t)(ctx->ndim()),
                       data_type,
                       (int64_t*)ctx->shape().data(),
                       (int64_t*)(nullptr),
                       0};
    return new DLManagedTensor{dl_tensor, ctx, [](DLManagedTensor* dlmt) {  //
                                   delete (Tensor*)dlmt->manager_ctx;
                                   delete dlmt;
                               }};
}

ft::DeviceType getMemoryType(DLDevice device)
{
    switch (device.device_type) {
        case DLDeviceType::kDLCUDAHost:
            return ft::DeviceType::kCPUpinned;
        case DLDeviceType::kDLCUDA:
            return ft::DeviceType::kDEVICE;
        case DLDeviceType::kDLCPU:
        default:
            return ft::DeviceType::kCPU;
    }
}

ft::DataType getDataType(DLDataType data_type)
{
    using ft::data_type_v;
    switch (data_type.code) {
        case DLDataTypeCode::kDLUInt:
            switch (data_type.bits) {
                case 8:
                    return data_type_v<uint8_t>;
                case 16:
                    return data_type_v<uint16_t>;
                case 32:
                    return data_type_v<uint32_t>;
                case 64:
                    return data_type_v<uint64_t>;
                default:
                    return data_type_v<void>;
            }
            break;
        case DLDataTypeCode::kDLInt:
            switch (data_type.bits) {
                case 8:
                    return data_type_v<int8_t>;
                case 16:
                    return data_type_v<int16_t>;
                case 32:
                    return data_type_v<int32_t>;
                case 64:
                    return data_type_v<int64_t>;
                default:
                    return data_type_v<void>;
            }
            break;
        case DLDataTypeCode::kDLFloat:
            switch (data_type.bits) {
                case 16:
                    return data_type_v<turbomind::half_t>;
                case 32:
                    return data_type_v<float>;
                case 64:
                    return data_type_v<double>;
                default:
                    return data_type_v<void>;
            }
            break;
        case DLDataTypeCode::kDLBfloat:
            switch (data_type.bits) {
                case 16:
                    return data_type_v<turbomind::bfloat16_t>;
                default:
                    return data_type_v<void>;
            }
            break;
        case DLDataTypeCode::kDLBool:
            return data_type_v<bool>;
        default:
            return data_type_v<void>;
    }
}

std::shared_ptr<Tensor> DLManagedTensorToTritonTensor(DLManagedTensor* tensor)
{
    auto& dl_tensor = tensor->dl_tensor;
    auto  where     = getMemoryType(dl_tensor.device);
    auto  dtype     = getDataType(dl_tensor.dtype);
    assert(dl_tensor.ndim > 0);
    std::vector<ft::core::ssize_t> shape(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim);

    std::shared_ptr<void> ptr{dl_tensor.data, [tensor](void* p) {
                                  if (tensor->deleter) {
                                      tensor->deleter(tensor);
                                  }
                              }};
    return std::make_shared<Tensor>(ptr, std::move(shape), dtype, where);
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
        TM_LOG_ERROR("cudaMemcpy failed: dst=({}, {}, {}, {}), src=({}, {}, {}, {}), size={}, device={}",
                     (int)dat.type,
                     dat.device,
                     dat.devicePointer,
                     dat.hostPointer,
                     (int)sat.type,
                     sat.device,
                     sat.devicePointer,
                     sat.hostPointer,
                     size,
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

// --- Generic config binding helper ---

template<typename Config>
void bind_config(py::module_& m, const char* name)
{
    py::class_<Config, turbomind::core::ModuleConfig> cls(m, name);
    cls.def(py::init<>());
    Config::for_each([&](const char* fname, auto member_ptr) { cls.def_readwrite(fname, member_ptr); });
    cls.def("clone", [](const Config& c) { return Config(c); });
}

template<typename T>
void bind_struct(py::module_& m, const char* name)
{
    py::class_<T> cls(m, name);
    cls.def(py::init<>());
    T::for_each([&](const char* fname, auto member_ptr) { cls.def_readwrite(fname, member_ptr); });
}

PYBIND11_MODULE(_turbomind, m)
{
    py::class_<ft::RequestMetrics, std::shared_ptr<ft::RequestMetrics>>(m, "RequestMetrics")
        .def(py::init())
        .def_property_readonly("enqueue_time",
                               [](ft::RequestMetrics& m) { return m.enqueue_time.load(std::memory_order_relaxed); })
        .def_property_readonly("scheduled_time",
                               [](ft::RequestMetrics& m) { return m.scheduled_time.load(std::memory_order_relaxed); });

    py::class_<ft::ScheduleMetrics, std::shared_ptr<ft::ScheduleMetrics>>(m, "ScheduleMetrics")
        .def(py::init())
        .def_readonly("total_seqs", &ft::ScheduleMetrics::total_seqs)
        .def_readonly("active_seqs", &ft::ScheduleMetrics::active_seqs)
        .def_readonly("waiting_seqs", &ft::ScheduleMetrics::waiting_seqs)
        .def_readonly("total_blocks", &ft::ScheduleMetrics::total_blocks)
        .def_readonly("active_blocks", &ft::ScheduleMetrics::active_blocks)
        .def_readonly("cached_blocks", &ft::ScheduleMetrics::cached_blocks)
        .def_readonly("free_blocks", &ft::ScheduleMetrics::free_blocks);

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
    {
        using namespace turbomind;
        py::enum_<ft::DataType>(m, "DataType")
            .value("TYPE_INVALID", kNull)
            .value("TYPE_BOOL", kBool)
            .value("TYPE_UINT8", kUint8)
            .value("TYPE_UINT16", kUint16)
            .value("TYPE_UINT32", kUint32)
            .value("TYPE_UINT64", kUint64)
            .value("TYPE_INT8", kInt8)
            .value("TYPE_INT16", kInt16)
            .value("TYPE_INT32", kInt32)
            .value("TYPE_INT64", kInt64)
            .value("TYPE_FP16", kFloat16)
            .value("TYPE_FP32", kFloat32)
            .value("TYPE_FP64", kFloat64)
            .value("TYPE_BF16", kBfloat16)
            .value("TYPE_FP8_E4M3", kFloat8_e4m3)
            .value("TYPE_FP4_E2M1", kFloat4_e2m1)
            .value("TYPE_UINT4", kUint4);

        // memory type
        py::enum_<ft::DeviceType>(m, "MemoryType")
            .value("MEMORY_CPU", ft::DeviceType::kCPU)
            .value("MEMORY_CPU_PINNED", ft::DeviceType::kCPUpinned)
            .value("MEMORY_GPU", ft::DeviceType::kDEVICE);
    }

    // DataFormat descriptors
    py::class_<turbomind::QuantParamDesc>(m, "QuantParamDesc")
        .def_readonly("dtype", &turbomind::QuantParamDesc::dtype)
        .def_readonly("transposed", &turbomind::QuantParamDesc::transposed)
        .def("present", &turbomind::QuantParamDesc::present);

    py::class_<turbomind::DataFormat>(m, "DataFormat")
        .def_readonly("dtype", &turbomind::DataFormat::dtype)
        .def_readonly("block_sizes", &turbomind::DataFormat::block_sizes)
        .def_readonly("scales", &turbomind::DataFormat::scales)
        .def_readonly("zeros", &turbomind::DataFormat::zeros)
        .def("is_quantized", &turbomind::DataFormat::is_quantized)
        .def("rank", &turbomind::DataFormat::rank);

    m.def("ResolveLinearWeightFormat",
          &turbomind::ResolveLinearWeightFormat,
          py::arg("data_type"),
          py::arg("weight_dtype"),
          py::arg("block_in"),
          py::arg("block_out"));

    // --- Config struct bindings ---
    py::class_<turbomind::core::ModuleConfig>(m, "ModuleConfig")
        .def_property_readonly("module_type", [](const turbomind::core::ModuleConfig& c) -> std::string {
            return std::string(c.module_type);
        });

    bind_config<turbomind::core::LinearConfig>(m, "LinearConfig");
    bind_struct<turbomind::core::RopeConfig>(m, "RopeConfig");
    bind_struct<turbomind::EngineConfig>(m, "EngineConfig");
    bind_config<turbomind::core::AttentionConfig>(m, "AttentionConfig");
    bind_config<turbomind::core::FfnConfig>(m, "FfnConfig");
    bind_config<turbomind::core::MoeConfig>(m, "MoeConfig");
    bind_config<turbomind::core::DeltaNetConfig>(m, "DeltaNetConfig");
    bind_config<turbomind::core::ModuleListConfig>(m, "ModuleListConfig");
    bind_config<turbomind::core::NormConfig>(m, "NormConfig");
    bind_config<turbomind::core::DecoderLayerConfig>(m, "DecoderLayerConfig");
    bind_config<turbomind::core::ModelWeightConfig>(m, "ModelWeightConfig");

    // tensor
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def_property_readonly("where", [](const Tensor& t) { return t.device().type; })
        .def_property_readonly("type", [](const Tensor& t) { return t.dtype(); })
        .def_property_readonly("shape", [](const Tensor& t) { return t.shape(); })
        .def_property_readonly("data", [](const Tensor& t) { return t.raw_data(); })
        .def_property_readonly("byte_size", [](const Tensor& t) { return t.byte_size(); })
        .def("__bool__", [](const Tensor& t) { return t.byte_size() > 0; })
        .def(
            "copy_from",
            [](Tensor& self, py::object obj) {
                py::capsule      cap = obj.attr("__dlpack__")();
                DLManagedTensor* dlmt =
                    static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap.ptr(), kDlTensorCapsuleName));
                auto src = DLManagedTensorToTritonTensor(dlmt);
                // take ownership of capsule's payload
                cap.set_name("used_dltensor");

                TM_CHECK_EQ(self.byte_size(), src->byte_size()) << self << " " << *src;
                safe_memcpy(self.raw_data(), src->raw_data(), self.byte_size());
            },
            "tensor"_a)
        .def(
            "__dlpack__",
            [](Tensor& self, long stream) {
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
        .def("__dlpack_device__", [](const Tensor& self) {
            auto device = getDLDevice(self);
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

    py::bind_map<TensorMap, std::shared_ptr<TensorMap>>(m, "TensorMap");

    using ft::ModelRequest;
    py::class_<ModelRequest>(m, "ModelRequest")
        .def(
            "forward",
            [](ModelRequest*               model_request,
               std::shared_ptr<TensorMap>  input_tensors,
               const ft::SessionParam&     session,
               const ft::GenerationConfig& gen_cfg,
               bool                        stream_output,
               bool                        enable_metrics,
               std::function<void()>       cb) {
                ModelRequest::InputParam param{};
                param.tensors        = std::move(input_tensors);
                param.session        = session;
                param.gen_cfg        = gen_cfg;
                param.stream_output  = stream_output;
                param.enable_metrics = enable_metrics;

                auto ret = model_request->Forward(std::move(param), [cb = std::move(cb)]() {
                    try {
                        cb();
                    }
                    catch (const py::error_already_set& e) {
                        std::cerr << e.what() << std::endl;
                    }
                });
                return std::make_tuple(std::move(ret.tensors), std::move(ret.state), std::move(ret.metrics));
            },
            py::call_guard<py::gil_scoped_release>(),
            "input_tensors"_a,
            "session"_a,
            "gen_cfg"_a,
            "stream_output"_a,
            "enable_metrics"_a,
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
            "session_id"_a)
        .def(
            "set_grammar",
            [](ModelRequest* model_request, const xgrammar::CompiledGrammar& grammar) {
                TM_LOG_INFO("Set grammar for model_request");
                model_request->setGrammar(grammar);
            },
            py::call_guard<py::gil_scoped_release>(),
            "grammar"_a);

    // Python context manager wrapper for ContextGuard.
    // Stores copies of Stream + Allocator; constructs the real guard
    // in-place on __enter__ and destroys it on __exit__.
    struct PyContextGuard {
        ft::core::Stream                        stream;
        ft::core::Allocator                     alloc;
        std::unique_ptr<ft::core::ContextGuard> guard;

        PyContextGuard(ft::core::Stream s, ft::core::Allocator a): stream(std::move(s)), alloc(std::move(a)) {}

        void enter()
        {
            guard = std::make_unique<ft::core::ContextGuard>(stream, alloc);
        }
        void exit()
        {
            guard.reset();
        }
    };

    py::class_<PyContextGuard>(m, "ContextGuard")
        .def("__enter__",
             [](PyContextGuard& g) -> PyContextGuard& {
                 g.enter();
                 return g;
             })
        .def("__exit__", [](PyContextGuard& g, py::object, py::object, py::object) { g.exit(); });

    // Param — lightweight handle to a Module parameter slot
    py::class_<ft::core::Param>(m, "Param")
        .def(
            "alloc",
            [](ft::core::Param& p, std::vector<size_t> shape, ft::DataType dtype) {
                return std::make_shared<Tensor>(p.alloc(shape, dtype));
            },
            "shape"_a,
            "dtype"_a)
        .def("get", [](ft::core::Param& p) { return std::make_shared<Tensor>(p.get()); })
        .def("__bool__", [](ft::core::Param& p) { return static_cast<bool>(p); });

    // Module class — navigation and allocation interface
    py::class_<ft::core::Module>(m, "Module")
        .def(
            "get",
            [](ft::core::Module& m, const std::string& segment) -> ft::core::Module* { return m.get(segment); },
            py::return_value_policy::reference,
            "segment"_a)
        .def(
            "param",
            [](ft::core::Module& m, const std::string& name) -> ft::core::Param { return m.param(name); },
            "name"_a)
        .def("prepare", [](ft::core::Module& m) { m.prepare(); })
        .def(
            "child",
            [](ft::core::Module& m, const std::string& name) -> ft::core::Module* { return m.child(name); },
            py::return_value_policy::reference,
            "name"_a)
        // Config-based create_child: accepts any ModuleConfig subclass
        .def(
            "create_child",
            [](ft::core::Module&              m,
               const std::string&             name,
               turbomind::core::ModuleConfig& config) -> ft::core::Module* { return m.create_child(name, config); },
            py::return_value_policy::reference,
            "name"_a,
            "config"_a)
        .def("type", [](ft::core::Module& m) -> const char* { return m.type(); })
        .def("full_path", [](ft::core::Module& m) -> std::string { return m.full_path(); })
        .def(
            "__getitem__",
            [](ft::core::Module& m, const std::string& key) -> ft::core::Module* { return m.get(key); },
            py::return_value_policy::reference)
        .def(
            "__getitem__",
            [](ft::core::Module& m, int idx) -> ft::core::Module* { return m.get(std::to_string(idx)); },
            py::return_value_policy::reference)
        // Deferred parent binding — transfer ownership of a previously created module
        .def(
            "add_child_raw",
            [](ft::core::Module& parent, const std::string& name, ft::core::Module* child) -> ft::core::Module* {
                auto owned = std::unique_ptr<ft::core::Module>(child);
                return parent.add_child(name, std::move(owned));
            },
            py::return_value_policy::reference,
            "name"_a,
            "child"_a);

    // Standalone module creation (no parent needed)
    m.def(
        "create_module",
        [](turbomind::core::ModuleConfig& config) -> ft::core::Module* {
            auto mod = ft::core::Module::create(config);
            return mod.release();
        },
        py::return_value_policy::reference,
        "config"_a);

    // LinearWeight — specific interface for weight loading
    py::class_<turbomind::LinearWeight, ft::core::Module>(m, "LinearWeight");

    // transformer model
    using ft::TurboMind;
    py::class_<TurboMind, std::shared_ptr<TurboMind>>(m, "TurboMind")
        .def_static(
            "create",
            [](std::string model_dir, turbomind::EngineConfig config) -> std::shared_ptr<TurboMind> {
                auto gil_factory = [] {  //
                    // erase the type
                    return std::static_pointer_cast<void>(std::make_shared<ScopedGIL>());
                };
                auto no_gil_deleter = [](TurboMind* ptr) {
                    pybind11::gil_scoped_release release;
                    delete ptr;
                };

                std::shared_ptr<TurboMind> model(new TurboMind(model_dir, std::move(config), gil_factory),
                                                 no_gil_deleter);
                return model;
            },
            "model_dir"_a,
            "engine_config"_a)
        .def(
            "create_request",
            [](TurboMind* model) { return model->CreateRequest(); },
            py::call_guard<py::gil_scoped_release>())
        .def("create_context", &TurboMind::CreateContext, py::call_guard<py::gil_scoped_release>(), "index"_a)
        .def(
            "create_root",
            [](TurboMind* model, int index) -> ft::core::Module* { return model->CreateRoot(index); },
            py::return_value_policy::reference,
            py::call_guard<py::gil_scoped_release>(),
            "index"_a)
        .def(
            "root",
            [](TurboMind* model, int index) -> ft::core::Module* { return model->root(index); },
            py::return_value_policy::reference,
            "index"_a)
        .def(
            "context",
            [](ft::TurboMind* model, int index) -> std::unique_ptr<PyContextGuard> {
                auto [stream, alloc] = model->weight_context(index);
                return std::make_unique<PyContextGuard>(std::move(stream), std::move(alloc));
            },
            "index"_a)
        .def(
            "process_weight",
            [](TurboMind* model, int index) { model->ProcessWeights(index); },
            py::call_guard<py::gil_scoped_release>(),
            "index"_a)
        .def(
            "create_engine",
            [](TurboMind* model, int index) { model->CreateEngine(index); },
            py::call_guard<py::gil_scoped_release>(),
            "index"_a)
        .def(
            "get_schedule_metrics",
            [](TurboMind* model, int index) { return model->GetScheduleMetrics(index); },
            py::call_guard<py::gil_scoped_release>(),
            "index"_a)
        .def("is_dummy_node", [](TurboMind* model) { return model->is_dummy_node(); })
        .def("attn_tp_rank", &TurboMind::GetAttnTpRank, "index"_a)
        .def("mlp_tp_rank", &TurboMind::GetMlpTpRank, "index"_a)
        .def("model_tp_rank", &TurboMind::GetModelTpRank, "index"_a);
}
