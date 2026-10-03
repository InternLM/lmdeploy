#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/python/dlpack.h"

namespace py = pybind11;
namespace ft = turbomind;

namespace turbomind::python::detail {

using ft::core::Tensor;

inline constexpr char kDlTensorCapsuleName[] = "dltensor";

inline ft::DataType getDataType(DLDataType source)
{
    using ft::data_type_v;
    switch (source.code) {
        case DLDataTypeCode::kDLUInt:
            switch (source.bits) {
                case 8:
                    return data_type_v<uint8_t>;
                case 16:
                    return data_type_v<uint16_t>;
                case 32:
                    return data_type_v<uint32_t>;
                case 64:
                    return data_type_v<uint64_t>;
            }
            break;
        case DLDataTypeCode::kDLInt:
            switch (source.bits) {
                case 8:
                    return data_type_v<int8_t>;
                case 16:
                    return data_type_v<int16_t>;
                case 32:
                    return data_type_v<int32_t>;
                case 64:
                    return data_type_v<int64_t>;
            }
            break;
        case DLDataTypeCode::kDLFloat:
            switch (source.bits) {
                case 16:
                    return data_type_v<ft::half_t>;
                case 32:
                    return data_type_v<float>;
                case 64:
                    return data_type_v<double>;
            }
            break;
        case DLDataTypeCode::kDLBfloat:
            if (source.bits == 16) {
                return data_type_v<ft::bfloat16_t>;
            }
            break;
        case DLDataTypeCode::kDLBool:
            if (source.bits == 8) {
                return data_type_v<bool>;
            }
            break;
    }
    __builtin_unreachable();
}

inline ft::core::Device getDevice(DLDevice source)
{
    switch (source.device_type) {
        case DLDeviceType::kDLCUDA:
            return {ft::DeviceType::kDEVICE, source.device_id};
        case DLDeviceType::kDLCUDAHost:
            return {ft::DeviceType::kCPUpinned, -1};
        case DLDeviceType::kDLCPU:
            return {ft::DeviceType::kCPU, -1};
        default:
            __builtin_unreachable();
    }
}

inline Tensor ConsumeDLPackWithStrides(py::handle object, uintptr_t consumer_stream)
{
    const uintptr_t dlpack_stream = consumer_stream == 0 ? 1 : consumer_stream;

    py::capsule capsule = object.attr("__dlpack__")(py::arg("stream") = py::int_(dlpack_stream));
    auto*       managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), kDlTensorCapsuleName));
    auto&       source  = managed->dl_tensor;

    using index_t = ft::core::ssize_t;
    std::vector<index_t> shape(source.ndim);
    std::vector<index_t> stride(source.ndim);

    for (int i = 0; i < source.ndim; ++i) {
        shape[i] = static_cast<index_t>(source.shape[i]);
    }

    if (source.strides) {
        for (int i = 0; i < source.ndim; ++i) {
            stride[i] = static_cast<index_t>(source.strides[i]);
        }
    }
    else {
        stride.back() = 1;
        if (source.ndim == 2) {
            stride.front() = shape.back();
        }
    }

    index_t logical_size = shape[0];
    if (source.ndim == 2) {
        logical_size *= shape[1];
    }
    if (logical_size == 0) {
        stride.back() = 1;
        if (source.ndim == 2) {
            stride.front() = shape.back();
        }
    }

    void* data = source.data;
    if (source.byte_offset != 0) {
        data = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(data) + static_cast<uintptr_t>(source.byte_offset));
    }

    capsule.set_name("used_dltensor");
    std::shared_ptr<void> owner{data, [managed](void*) {
                                    if (managed->deleter) {
                                        managed->deleter(managed);
                                    }
                                }};

    return Tensor{std::move(owner),
                  ft::core::Layout{std::move(shape), std::move(stride)},
                  getDataType(source.dtype),
                  getDevice(source.device)};
}

}  // namespace turbomind::python::detail
