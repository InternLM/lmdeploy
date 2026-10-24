// Copyright (c) OpenMMLab. All rights reserved.

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>

#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/attention/verification/attention.h"
#include "src/turbomind/python/attention_component_bindings.h"
#include "src/turbomind/python/eagle3_dlpack_internal.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace py = pybind11;

namespace turbomind::python {
namespace {

int GetCudaOrdinal(py::handle tensor)
{
    return tensor.attr("__dlpack_device__")().cast<py::tuple>()[1].cast<int>();
}

template<class T>
int LaunchVerificationAttention(py::handle prefix_k_object,
                                py::handle prefix_v_object,
                                py::handle prefix_offsets_object,
                                int        max_history_length,
                                py::handle packed_qkv_object,
                                py::handle q_bias_object,
                                py::handle output_object,
                                py::handle cache_storage_object,
                                py::handle block_ptrs_object,
                                py::handle block_ptr_offsets_object,
                                py::handle q_offsets_object,
                                py::handle k_offsets_object,
                                py::handle finished_object,
                                py::handle partial_o_object,
                                py::handle partial_ml_object,
                                int        query_head_count,
                                int        kv_head_count,
                                int        head_dim,
                                int        block_len,
                                int        max_query_length,
                                int        max_key_length,
                                int        window_size,
                                int        requested_max_split_count,
                                int        rope_type,
                                int        rope_dim,
                                float      rope_base,
                                float      rope_factor,
                                int        mrope_mode,
                                int        mrope_section_t,
                                int        mrope_section_h,
                                int        mrope_section_w,
                                py::handle mrope_position_ids_object,
                                py::handle mrope_position_delta_object,
                                py::handle mrope_length_object,
                                uintptr_t  stream_ptr)
{
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    auto prefix_k = detail::ConsumeDLPackWithStrides(prefix_k_object, stream_ptr);
    auto prefix_v = detail::ConsumeDLPackWithStrides(prefix_v_object, stream_ptr);
    auto prefix_offsets = detail::ConsumeDLPackWithStrides(prefix_offsets_object, stream_ptr);
    auto packed_qkv = detail::ConsumeDLPackWithStrides(packed_qkv_object, stream_ptr);
    auto q_bias = detail::ConsumeDLPackWithStrides(q_bias_object, stream_ptr);
    auto output = detail::ConsumeDLPackWithStrides(output_object, stream_ptr);
    auto cache_storage =
        detail::ConsumeDLPackWithStrides(cache_storage_object, stream_ptr);
    (void)cache_storage;
    auto block_ptrs = detail::ConsumeDLPackWithStrides(block_ptrs_object, stream_ptr);
    auto block_ptr_offsets = detail::ConsumeDLPackWithStrides(block_ptr_offsets_object, stream_ptr);
    auto q_offsets = detail::ConsumeDLPackWithStrides(q_offsets_object, stream_ptr);
    auto k_offsets = detail::ConsumeDLPackWithStrides(k_offsets_object, stream_ptr);
    auto finished = detail::ConsumeDLPackWithStrides(finished_object, stream_ptr);
    auto partial_o = detail::ConsumeDLPackWithStrides(partial_o_object, stream_ptr);
    auto partial_ml = detail::ConsumeDLPackWithStrides(partial_ml_object, stream_ptr);
    auto mrope_position_ids = detail::ConsumeDLPackWithStrides(mrope_position_ids_object, stream_ptr);
    auto mrope_position_delta =
        detail::ConsumeDLPackWithStrides(mrope_position_delta_object, stream_ptr);
    auto mrope_length = detail::ConsumeDLPackWithStrides(mrope_length_object, stream_ptr);

    RopeKernelParam rope{};
    rope.type = static_cast<RopeType>(rope_type);
    rope.dim = rope_dim;
    rope.scale_factor =
        rope.type == RopeType::kNull ? 0.f : -std::log2(rope_base) / rope_dim;
    rope.inv_factor = rope_factor != 0.f ? 1.f / rope_factor : 1.f;
    rope.mrope_mode = static_cast<MropeMode>(mrope_mode);
    if (rope.mrope_mode != MropeMode::kNone) {
        rope.mrope.section = make_int3(mrope_section_t, mrope_section_h, mrope_section_w);
        rope.mrope.stride = static_cast<int>(mrope_position_ids.stride(0));
        rope.mrope.position_ids = mrope_position_ids.data<int>();
        rope.mrope.position_delta = mrope_position_delta.data<int>();
        rope.mrope.length = mrope_length.data<int>();
    }

    const int batch_size = static_cast<int>(q_offsets.shape(0) - 1);
    const int64_t prefix_head_stride = prefix_k.stride(0) / head_dim;
    invokeProcessKV_v2<T>(reinterpret_cast<char**>(block_ptrs.data<int64_t>()),
                          prefix_k.data<T>(),
                          prefix_v.data<T>(),
                          nullptr,
                          nullptr,
                          prefix_offsets.data<int>(),
                          prefix_offsets.data<int>(),
                          block_ptr_offsets.data<int>(),
                          nullptr,
                          nullptr,
                          rope,
                          0,
                          prefix_head_stride,
                          1,
                          prefix_head_stride,
                          block_len,
                          0,
                          0,
                          cutlass::FastDivmod(1),
                          max_history_length,
                          kv_head_count,
                          head_dim,
                          batch_size,
                          0,
                          stream);

    const int64_t qkv_stride = packed_qkv.stride(0);
    const int64_t qkv_head_stride = qkv_stride / head_dim;
    const T* packed = packed_qkv.data<T>();
    const T* submitted_k = packed + query_head_count * head_dim;
    const T* submitted_v = submitted_k + kv_head_count * head_dim;
    invokeProcessKV_v2<T>(reinterpret_cast<char**>(block_ptrs.data<int64_t>()),
                          submitted_k,
                          submitted_v,
                          nullptr,
                          nullptr,
                          q_offsets.data<int>(),
                          k_offsets.data<int>(),
                          block_ptr_offsets.data<int>(),
                          nullptr,
                          finished.data<bool>(),
                          rope,
                          0,
                          qkv_head_stride,
                          1,
                          qkv_head_stride,
                          block_len,
                          0,
                          0,
                          cutlass::FastDivmod(1),
                          max_query_length,
                          kv_head_count,
                          head_dim,
                          batch_size,
                          0,
                          stream);

    verification_attention::Arguments arguments{};
    arguments.out = output.data<T>();
    arguments.q = packed;
    arguments.q_bias = q_bias.shape(0) ? q_bias.data<T>() : nullptr;
    arguments.q_stride = qkv_stride;
    arguments.block_ptrs =
        reinterpret_cast<char* const*>(block_ptrs.data<int64_t>());
    arguments.block_ptr_offsets = block_ptr_offsets.data<int>();
    arguments.q_offsets = q_offsets.data<int>();
    arguments.k_offsets = k_offsets.data<int>();
    arguments.finished = finished.data<bool>();
    arguments.request_count = batch_size;
    arguments.query_count = static_cast<int>(packed_qkv.shape(0));
    arguments.query_offset = 0;
    arguments.max_query_length = max_query_length;
    arguments.max_key_length = max_key_length;
    arguments.query_head_count = query_head_count;
    arguments.kv_head_count = kv_head_count;
    arguments.query_group_size = query_head_count / kv_head_count;
    arguments.query_group_size_divmod = cutlass::FastDivmod(arguments.query_group_size);
    arguments.head_dim = head_dim;
    arguments.block_len = block_len;
    arguments.block_len_divmod = cutlass::FastDivmod(block_len);
    arguments.cache_block_offset = 0;
    arguments.window_size = window_size ? window_size : (256 << 20);
    arguments.qk_scale_log2 =
        std::log2(std::exp(1.f)) / std::sqrt(static_cast<float>(head_dim));
    arguments.rope = rope;
    arguments.partial_o = partial_o.data<float>();
    arguments.partial_ml = partial_ml.data<float>();
    arguments.data_type = packed_qkv.dtype();
    arguments.stream = stream;

    const int m_slices = (max_query_length * arguments.query_group_size
                          + verification_attention::CtaM(arguments) - 1)
                         / verification_attention::CtaM(arguments);
    const int base_cta_count = batch_size * kv_head_count * m_slices;
    arguments.split_count = verification_attention::choose_split_count(arguments.query_count,
                                                                        base_cta_count,
                                                                        arguments.max_key_length,
                                                                        verification_attention::KeyTile(arguments),
                                                                        static_cast<int>(partial_o.shape(0)),
                                                                        requested_max_split_count,
                                                                        getSMCount());
    verification_attention::run(arguments);
    return arguments.split_count;
}

}  // namespace

void BindVerificationAttention(py::module_& module)
{
#if TM_BUILD_VERIFICATION_ATTENTION_SM90
    module.def(
        "verification_attention",
        [](py::handle prefix_k,
           py::handle prefix_v,
           py::handle prefix_offsets,
           int max_history_length,
           py::handle packed_qkv,
           py::handle q_bias,
           py::handle output,
           py::handle cache_storage,
           py::handle block_ptrs,
           py::handle block_ptr_offsets,
           py::handle q_offsets,
           py::handle k_offsets,
           py::handle finished,
           py::handle partial_o,
           py::handle partial_ml,
           int query_head_count,
           int kv_head_count,
           int head_dim,
           int block_len,
           int max_query_length,
           int max_key_length,
           int window_size,
           int requested_max_split_count,
           int rope_type,
           int rope_dim,
           float rope_base,
           float rope_factor,
           int mrope_mode,
           int mrope_section_t,
           int mrope_section_h,
           int mrope_section_w,
           py::handle mrope_position_ids,
           py::handle mrope_position_delta,
           py::handle mrope_length,
           uintptr_t stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(packed_qkv)};
            const auto dtype = detail::ConsumeDLPackWithStrides(packed_qkv, stream_ptr).dtype();
            if (dtype == DataType::kHalf) {
                return LaunchVerificationAttention<half>(prefix_k,
                                                         prefix_v,
                                                         prefix_offsets,
                                                         max_history_length,
                                                         packed_qkv,
                                                         q_bias,
                                                         output,
                                                         cache_storage,
                                                         block_ptrs,
                                                         block_ptr_offsets,
                                                         q_offsets,
                                                         k_offsets,
                                                         finished,
                                                         partial_o,
                                                         partial_ml,
                                                         query_head_count,
                                                         kv_head_count,
                                                         head_dim,
                                                         block_len,
                                                         max_query_length,
                                                         max_key_length,
                                                         window_size,
                                                         requested_max_split_count,
                                                         rope_type,
                                                         rope_dim,
                                                         rope_base,
                                                         rope_factor,
                                                         mrope_mode,
                                                         mrope_section_t,
                                                         mrope_section_h,
                                                         mrope_section_w,
                                                         mrope_position_ids,
                                                         mrope_position_delta,
                                                         mrope_length,
                                                         stream_ptr);
            }
            return LaunchVerificationAttention<nv_bfloat16>(prefix_k,
                                                            prefix_v,
                                                            prefix_offsets,
                                                            max_history_length,
                                                            packed_qkv,
                                                            q_bias,
                                                            output,
                                                            cache_storage,
                                                            block_ptrs,
                                                            block_ptr_offsets,
                                                            q_offsets,
                                                            k_offsets,
                                                            finished,
                                                            partial_o,
                                                            partial_ml,
                                                            query_head_count,
                                                            kv_head_count,
                                                            head_dim,
                                                            block_len,
                                                            max_query_length,
                                                            max_key_length,
                                                            window_size,
                                                            requested_max_split_count,
                                                            rope_type,
                                                            rope_dim,
                                                            rope_base,
                                                            rope_factor,
                                                            mrope_mode,
                                                            mrope_section_t,
                                                            mrope_section_h,
                                                            mrope_section_w,
                                                            mrope_position_ids,
                                                            mrope_position_delta,
                                                            mrope_length,
                                                            stream_ptr);
        },
        py::arg("prefix_k"),
        py::arg("prefix_v"),
        py::arg("prefix_offsets"),
        py::arg("max_history_length"),
        py::arg("packed_qkv"),
        py::arg("q_bias"),
        py::arg("output"),
        py::arg("cache_storage"),
        py::arg("block_ptrs"),
        py::arg("block_ptr_offsets"),
        py::arg("q_offsets"),
        py::arg("k_offsets"),
        py::arg("finished"),
        py::arg("partial_o"),
        py::arg("partial_ml"),
        py::arg("query_head_count"),
        py::arg("kv_head_count"),
        py::arg("head_dim"),
        py::arg("block_len"),
        py::arg("max_query_length"),
        py::arg("max_key_length"),
        py::arg("window_size"),
        py::arg("requested_max_split_count"),
        py::arg("rope_type"),
        py::arg("rope_dim"),
        py::arg("rope_base"),
        py::arg("rope_factor"),
        py::arg("mrope_mode"),
        py::arg("mrope_section_t"),
        py::arg("mrope_section_h"),
        py::arg("mrope_section_w"),
        py::arg("mrope_position_ids"),
        py::arg("mrope_position_delta"),
        py::arg("mrope_length"),
        py::arg("stream_ptr"));
#else
    (void)module;
#endif
}

}  // namespace turbomind::python
