// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <xgrammar/xgrammar.h>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/model_request.h"
#include "src/turbomind/python/dlpack.h"
#include "src/turbomind/triton_backend/llama/LlamaTritonModel.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/metrics.h"

namespace nb  = nanobind;
namespace tm_ = turbomind;

using tm_::core::Tensor;
using namespace nanobind::literals;

// prepare to bind container
using TensorMap = turbomind::core::TensorMap;
NB_MAKE_OPAQUE(TensorMap)

static const char kDlTensorCapsuleName[] = "dltensor";

DLDevice getDLDevice(const Tensor& tensor)
{
    int device_id = 0;
    if (tensor.device().type == tm_::kDEVICE) {
        cudaPointerAttributes ptr_attr{};
        cudaPointerGetAttributes(&ptr_attr, tensor.raw_data());
        device_id = ptr_attr.device;
    }

    DLDevice device{kDLCPU, device_id};

    switch (tensor.device().type) {
        case tm_::kCPU:
            device.device_type = DLDeviceType::kDLCPU;
            break;
        case tm_::kCPUpinned:
            device.device_type = DLDeviceType::kDLCUDAHost;
            break;
        case tm_::kDEVICE:
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
    using tm_::data_type_v;
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

tm_::DeviceType getMemoryType(DLDevice device)
{
    switch (device.device_type) {
        case DLDeviceType::kDLCUDAHost:
            return tm_::DeviceType::kCPUpinned;
        case DLDeviceType::kDLCUDA:
            return tm_::DeviceType::kDEVICE;
        case DLDeviceType::kDLCPU:
        default:
            return tm_::DeviceType::kCPU;
    }
}

tm_::DataType getDataType(DLDataType data_type)
{
    using tm_::data_type_v;
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
    std::vector<tm_::core::ssize_t> shape(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim);

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
    tm_::check_cuda_error(cudaPointerGetAttributes(&dat, dst));
    tm_::check_cuda_error(cudaPointerGetAttributes(&sat, src));
    try {
        if (dat.devicePointer && sat.devicePointer) {
            // Both can be accessed from current context
            tm_::check_cuda_error(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
        }
        else if (dat.type == cudaMemoryTypeDevice && sat.type == cudaMemoryTypeDevice) {
            if (dat.device != sat.device) {
                // On different devices, try peer memcpy
                tm_::check_cuda_error(cudaMemcpyPeer(dst, dat.device, src, sat.device, size));
            }
            else {
                // Same device, switch to the device first (this is unlikely)
                tm_::CudaDeviceGuard guard(dat.device);
                tm_::check_cuda_error(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
            }
        }
        else {
            // Unknown case, give it a try anyway
            tm_::check_cuda_error(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
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

NB_MODULE(_turbomind, m)
{
    nb::class_<tm_::RequestMetrics>(m, "RequestMetrics")
        .def(nb::init())
        .def_ro("enque_time", &tm_::RequestMetrics::enque_time)
        .def_ro("scheduled_time", &tm_::RequestMetrics::scheduled_time);

    nb::class_<tm_::ScheduleMetrics>(m, "ScheduleMetrics")
        .def(nb::init())
        .def_ro("total_seqs", &tm_::ScheduleMetrics::total_seqs)
        .def_ro("active_seqs", &tm_::ScheduleMetrics::active_seqs)
        .def_ro("waiting_seqs", &tm_::ScheduleMetrics::waiting_seqs)
        .def_ro("total_blocks", &tm_::ScheduleMetrics::total_blocks)
        .def_ro("active_blocks", &tm_::ScheduleMetrics::active_blocks)
        .def_ro("cached_blocks", &tm_::ScheduleMetrics::cached_blocks)
        .def_ro("free_blocks", &tm_::ScheduleMetrics::free_blocks);

    nb::class_<tm_::SessionParam>(m, "SessionParam")
        .def(
            "__init__",
            [](tm_::SessionParam* param, uint64_t id, int step, bool start, bool end) {
                if (!start && end) {
                    throw std::logic_error("unsupported arguments: start=false, end=true");
                }
                new (param) tm_::SessionParam();
                param->id         = id;
                param->step       = step;
                param->start_flag = start;
                param->end_flag   = end;
            },
            "id"_a,
            "step"_a,
            "start"_a,
            "end"_a)
        .def_rw("id", &tm_::SessionParam::id)
        .def_rw("step", &tm_::SessionParam::step)
        .def_rw("start", &tm_::SessionParam::start_flag)
        .def_rw("end", &tm_::SessionParam::end_flag);

    nb::class_<tm_::GenerationConfig>(m, "GenerationConfig")
        .def(nb::init())
        .def_rw("max_new_tokens", &tm_::GenerationConfig::max_new_tokens)
        .def_rw("min_new_tokens", &tm_::GenerationConfig::min_new_tokens)
        .def_rw("eos_ids", &tm_::GenerationConfig::eos_ids)
        .def_rw("stop_ids", &tm_::GenerationConfig::stop_ids)
        .def_rw("bad_ids", &tm_::GenerationConfig::bad_ids)
        .def_rw("top_p", &tm_::GenerationConfig::top_p)
        .def_rw("top_k", &tm_::GenerationConfig::top_k)
        .def_rw("min_p", &tm_::GenerationConfig::min_p)
        .def_rw("temperature", &tm_::GenerationConfig::temperature)
        .def_rw("repetition_penalty", &tm_::GenerationConfig::repetition_penalty)
        .def_rw("random_seed", &tm_::GenerationConfig::random_seed)
        .def_rw("output_logprobs", &tm_::GenerationConfig::output_logprobs)
        .def_rw("output_last_hidden_state", &tm_::GenerationConfig::output_last_hidden_state)
        .def_rw("output_logits", &tm_::GenerationConfig::output_logits)
        .def("__repr__", [](const tm_::GenerationConfig& c) {
            std::ostringstream oss;
            oss << c;
            return oss.str();
        });

    nb::class_<tm_::RequestState>(m, "RequestState")
        .def_ro("status", &tm_::RequestState::status)
        .def_ro("seq_len", &tm_::RequestState::seq_len);

    nb::class_<tm_::AtomicRequestState>(m, "AtomicRequestState").def("consume", [](tm_::AtomicRequestState& s) {
        return s.exchange(nullptr);
    });

    // data type
    {
        using namespace turbomind;
        nb::enum_<tm_::DataType>(m, "DataType")
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
            .value("TYPE_BF16", kBfloat16);

        // memory type
        nb::enum_<tm_::DeviceType>(m, "MemoryType")
            .value("MEMORY_CPU", tm_::DeviceType::kCPU)
            .value("MEMORY_CPU_PINNED", tm_::DeviceType::kCPUpinned)
            .value("MEMORY_GPU", tm_::DeviceType::kDEVICE);
    }

    // tensor
    nb::class_<Tensor>(m, "Tensor")
        .def_prop_ro("where", [](const Tensor& t) { return t.device().type; })
        .def_prop_ro("type", [](const Tensor& t) { return t.dtype(); })
        .def_prop_ro("shape", [](const Tensor& t) { return t.shape(); })
        .def_prop_ro("data", [](Tensor& t) { return t.raw_data(); })
        .def(
            "copy_from",
            [](Tensor& self, nb::object obj) {
                nb::object cap     = obj.attr("__dlpack__")();
                PyObject*  cap_ptr = cap.ptr();

                if (!PyCapsule_CheckExact(cap_ptr)) {
                    throw nb::type_error("__dlpack__ did not return a capsule");
                }

                DLManagedTensor* dlmt =
                    static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap_ptr, kDlTensorCapsuleName));
                auto src = DLManagedTensorToTritonTensor(dlmt);
                // take ownership of capsule's payload
                PyCapsule_SetName(cap_ptr, "used_dltensor");

                TM_CHECK_EQ(self.byte_size(), src->byte_size()) << self << " " << *src;
                safe_memcpy(self.raw_data(), src->raw_data(), self.byte_size());
            },
            "tensor"_a)
        .def(
            "__dlpack__",
            [](Tensor& self, long stream) {
                DLManagedTensor* dlmt = TritonTensorToDLManagedTensor(self);
                return nb::capsule(dlmt, kDlTensorCapsuleName, [](void* obj_) noexcept {
                    PyObject*        obj = static_cast<PyObject*>(obj_);
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
        [](nb::object obj) {
            nb::object cap     = obj.attr("__dlpack__")();
            PyObject*  cap_ptr = cap.ptr();

            if (!PyCapsule_CheckExact(cap_ptr)) {
                throw nb::type_error("__dlpack__ did not return a capsule");
            }

            DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap_ptr, kDlTensorCapsuleName));
            auto             ret  = DLManagedTensorToTritonTensor(dlmt);
            // take ownership of capsule's payload
            PyCapsule_SetName(cap_ptr, "used_dltensor");
            return ret;
        },
        "dl_managed_tensor"_a);

    nb::bind_map<TensorMap>(m, "TensorMap");

    using tm_::ModelRequest;
    nb::class_<ModelRequest>(m, "ModelRequest")
        .def(
            "forward",
            [](ModelRequest*                model_request,
               std::shared_ptr<TensorMap>   input_tensors,
               const tm_::SessionParam&     session,
               const tm_::GenerationConfig& gen_cfg,
               bool                         stream_output,
               bool                         enable_metrics,
               std::function<void()>        cb) {
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
                    catch (const nb::python_error& e) {
                        std::cerr << e.what() << std::endl;
                    }
                });
                return std::make_tuple(std::move(ret.tensors), std::move(ret.state), std::move(ret.metrics));
            },
            nb::call_guard<nb::gil_scoped_release>(),
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
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "end",
            [](ModelRequest* model_request, std::function<void(int)> cb, uint64_t session_id) {
                model_request->End(std::move(cb), session_id);  //
            },
            nb::call_guard<nb::gil_scoped_release>(),
            "cb"_a,
            "session_id"_a)
        .def(
            "set_grammar",
            [](ModelRequest* model_request, const xgrammar::CompiledGrammar& grammar) {
                TM_LOG_INFO("Set grammar for model_request");
                model_request->setGrammar(grammar);
            },
            nb::call_guard<nb::gil_scoped_release>(),
            "grammar"_a);

    // transformer model
    using tm_::LlamaTritonModel;
    nb::class_<LlamaTritonModel>(m, "AbstractTransformerModel")
        .def_static(
            "create_llama_model",
            [](std::string model_dir,
               std::string config,
               std::string weight_type) -> std::shared_ptr<LlamaTritonModel> {
                auto gil_factory = [] {  //
                    // erase the type
                    return std::static_pointer_cast<void>(std::make_shared<ScopedGIL>());
                };
                auto no_gil_deleter = [](LlamaTritonModel* ptr) {
                    nb::gil_scoped_release release;
                    delete ptr;
                };

                std::shared_ptr<LlamaTritonModel> model(new LlamaTritonModel(model_dir, config, gil_factory),
                                                        no_gil_deleter);
                return model;
            },
            "model_dir"_a,
            "config"_a      = "",
            "weight_type"_a = "half")
        .def(
            "create_model_instance",
            [](LlamaTritonModel* model, int deviceId) { return model->createModelInstance(deviceId); },
            nb::call_guard<nb::gil_scoped_release>(),
            "device_id"_a)
        .def("create_shared_weights",
             &LlamaTritonModel::createSharedWeights,
             nb::call_guard<nb::gil_scoped_release>(),
             "device_id"_a,
             "rank"_a)
        .def(
            "get_params",
            [](LlamaTritonModel* model, int deviceId, int rank) { return model->getParams(deviceId, rank); },
            nb::call_guard<nb::gil_scoped_release>(),
            "device_id"_a,
            "rank"_a)
        .def(
            "process_weight",
            [](LlamaTritonModel* model, int deviceId, int rank) { model->processWeights(deviceId, rank); },
            nb::call_guard<nb::gil_scoped_release>(),
            "device_id"_a,
            "rank"_a)
        .def(
            "create_engine",
            [](LlamaTritonModel* model, int deviceId, int rank) { model->createEngine(deviceId, rank); },
            nb::call_guard<nb::gil_scoped_release>(),
            "device_id"_a,
            "rank"_a)
        .def(
            "get_schedule_metrics",
            [](LlamaTritonModel* model, int deviceId, int rank) { return model->getScheduleMetrics(deviceId, rank); },
            nb::call_guard<nb::gil_scoped_release>(),
            "device_id"_a,
            "rank"_a)
        .def(
            "sleep",
            [](LlamaTritonModel* model, int deviceId, int level) { model->sleep(deviceId, level); },
            nb::call_guard<nb::gil_scoped_release>(),
            "device_id"_a,
            "level"_a)
        .def(
            "wakeup",
            [](LlamaTritonModel* model, int deviceId, const std::vector<std::string>& tags, int rank) {
                model->wakeup(deviceId, tags, rank);
            },
            nb::call_guard<nb::gil_scoped_release>(),
            "device_id"_a,
            "tags"_a,
            "rank"_a)
        .def("__str__", &LlamaTritonModel::toString)
        .def("__repr__", &LlamaTritonModel::toString)
        .def("get_tensor_para_size", &LlamaTritonModel::getTensorParaSize)
        .def("get_pipeline_para_size", &LlamaTritonModel::getPipelineParaSize);
}
