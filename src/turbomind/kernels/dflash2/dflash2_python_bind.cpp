// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>

#include "src/turbomind/kernels/dflash2/dflash2_kernels.h"
#include "src/turbomind/python/eagle3_component_bindings.h"
#include "src/turbomind/python/eagle3_dlpack_internal.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace py = pybind11;

namespace turbomind::python {
namespace {

int GetCudaOrdinal(py::handle tensor)
{
    return tensor.attr("__dlpack_device__")().cast<py::tuple>()[1].cast<int>();
}

int BitsOf(const detail::Tensor& t)
{
    return static_cast<int>(byte_size(t.dtype(), 8));
}

}  // namespace

void BindDflash2Kernels(py::module_& module)
{
    module.def(
        "grouped_conv",
        [](py::handle hidden_object,
           py::handle delta_object,
           py::handle base_object,
           py::handle out_object,
           int        block_size,
           int        num_groups,
           int        group_size,
           int        taps,
           uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(hidden_object)};
            auto hidden = detail::ConsumeDLPackWithStrides(hidden_object, stream_ptr);
            auto delta  = detail::ConsumeDLPackWithStrides(delta_object, stream_ptr);
            auto base   = detail::ConsumeDLPackWithStrides(base_object, stream_ptr);
            auto out    = detail::ConsumeDLPackWithStrides(out_object, stream_ptr);

            invokeGroupedConv(out.data_or(static_cast<void*>(nullptr)),
                              hidden.data_or(static_cast<const void*>(nullptr)),
                              delta.data_or(static_cast<const void*>(nullptr)),
                              base.data_or(static_cast<const void*>(nullptr)),
                              static_cast<int>(hidden.shape(0)),
                              static_cast<int>(hidden.shape(1)),
                              num_groups,
                              group_size,
                              taps,
                              block_size,
                              BitsOf(hidden),
                              reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("hidden"),
        py::arg("delta"),
        py::arg("base"),
        py::arg("out"),
        py::arg("block_size"),
        py::arg("num_groups"),
        py::arg("group_size"),
        py::arg("taps"),
        py::arg("stream_ptr"));

    module.def(
        "candidate_selector_scores",
        [](py::handle hidden_object,
           py::handle unary_object,
           py::handle pred_cb_object,
           py::handle succ_cb_object,
           py::handle cand_ids_object,
           py::handle anchor_object,
           py::handle out_object,
           int        top_k,
           uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(hidden_object)};
            auto hidden   = detail::ConsumeDLPackWithStrides(hidden_object, stream_ptr);
            auto unary    = detail::ConsumeDLPackWithStrides(unary_object, stream_ptr);
            auto pred_cb  = detail::ConsumeDLPackWithStrides(pred_cb_object, stream_ptr);
            auto succ_cb  = detail::ConsumeDLPackWithStrides(succ_cb_object, stream_ptr);
            auto cand_ids = detail::ConsumeDLPackWithStrides(cand_ids_object, stream_ptr);
            auto anchor   = detail::ConsumeDLPackWithStrides(anchor_object, stream_ptr);
            auto out      = detail::ConsumeDLPackWithStrides(out_object, stream_ptr);

            invokeCandidateSelector(reinterpret_cast<float*>(out.data_or(static_cast<void*>(nullptr))),
                                    hidden.data_or(static_cast<const void*>(nullptr)),
                                    unary.data_or(static_cast<const void*>(nullptr)),
                                    pred_cb.data_or(static_cast<const void*>(nullptr)),
                                    succ_cb.data_or(static_cast<const void*>(nullptr)),
                                    reinterpret_cast<const int*>(cand_ids.data_or(static_cast<int*>(nullptr))),
                                    reinterpret_cast<const int*>(anchor.data_or(static_cast<int*>(nullptr))),
                                    static_cast<int>(hidden.shape(0)),
                                    static_cast<int>(hidden.shape(1)),
                                    top_k,
                                    static_cast<int>(pred_cb.shape(0)),
                                    static_cast<int>(hidden.shape(2)),
                                    BitsOf(hidden),
                                    reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("hidden"),
        py::arg("unary_logits"),
        py::arg("predecessor_codebook"),
        py::arg("successor_codebook"),
        py::arg("candidate_ids"),
        py::arg("anchor_token_ids"),
        py::arg("out"),
        py::arg("top_k"),
        py::arg("stream_ptr"));

    module.def(
        "dflash2_attention",
        [](py::handle q_object,
           py::handle k_ctx_object,
           py::handle v_ctx_object,
           py::handle k_blk_object,
           py::handle v_blk_object,
           py::handle out_object,
           float      scale,
           int        window,
           uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(q_object)};
            auto q     = detail::ConsumeDLPackWithStrides(q_object, stream_ptr);
            auto k_ctx = detail::ConsumeDLPackWithStrides(k_ctx_object, stream_ptr);
            auto v_ctx = detail::ConsumeDLPackWithStrides(v_ctx_object, stream_ptr);
            auto k_blk = detail::ConsumeDLPackWithStrides(k_blk_object, stream_ptr);
            auto v_blk = detail::ConsumeDLPackWithStrides(v_blk_object, stream_ptr);
            auto out   = detail::ConsumeDLPackWithStrides(out_object, stream_ptr);

            invokeDflash2Attention(out.data_or(static_cast<void*>(nullptr)),
                                   q.data_or(static_cast<const void*>(nullptr)),
                                   k_ctx.data_or(static_cast<const void*>(nullptr)),
                                   v_ctx.data_or(static_cast<const void*>(nullptr)),
                                   k_blk.data_or(static_cast<const void*>(nullptr)),
                                   v_blk.data_or(static_cast<const void*>(nullptr)),
                                   static_cast<int>(q.shape(0)),
                                   static_cast<int>(k_ctx.shape(0)),
                                   static_cast<int>(q.shape(1)),
                                   static_cast<int>(k_ctx.shape(1)),
                                   static_cast<int>(q.shape(2)),
                                   scale,
                                   window,
                                   BitsOf(q),
                                   reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("q"),
        py::arg("k_ctx"),
        py::arg("v_ctx"),
        py::arg("k_blk"),
        py::arg("v_blk"),
        py::arg("out"),
        py::arg("scale"),
        py::arg("window"),
        py::arg("stream_ptr"));

    module.def(
        "dflash2_attention_int4",
        [](py::handle q_object,
           py::handle k_ctx_data_object,
           py::handle k_ctx_param_object,
           py::handle v_ctx_data_object,
           py::handle v_ctx_param_object,
           py::handle k_blk_object,
           py::handle v_blk_object,
           py::handle out_object,
           float      scale,
           int        window,
           uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(q_object)};
            auto q           = detail::ConsumeDLPackWithStrides(q_object, stream_ptr);
            auto k_ctx_data  = detail::ConsumeDLPackWithStrides(k_ctx_data_object, stream_ptr);
            auto k_ctx_param = detail::ConsumeDLPackWithStrides(k_ctx_param_object, stream_ptr);
            auto v_ctx_data  = detail::ConsumeDLPackWithStrides(v_ctx_data_object, stream_ptr);
            auto v_ctx_param = detail::ConsumeDLPackWithStrides(v_ctx_param_object, stream_ptr);
            auto k_blk       = detail::ConsumeDLPackWithStrides(k_blk_object, stream_ptr);
            auto v_blk       = detail::ConsumeDLPackWithStrides(v_blk_object, stream_ptr);
            auto out         = detail::ConsumeDLPackWithStrides(out_object, stream_ptr);

            invokeDflash2AttentionInt4(out.data_or(static_cast<void*>(nullptr)),
                                       q.data_or(static_cast<const void*>(nullptr)),
                                       k_ctx_data.data_or(static_cast<const void*>(nullptr)),
                                       k_ctx_param.data_or(static_cast<const void*>(nullptr)),
                                       v_ctx_data.data_or(static_cast<const void*>(nullptr)),
                                       v_ctx_param.data_or(static_cast<const void*>(nullptr)),
                                       k_blk.data_or(static_cast<const void*>(nullptr)),
                                       v_blk.data_or(static_cast<const void*>(nullptr)),
                                       static_cast<int>(q.shape(0)),
                                       static_cast<int>(k_ctx_data.shape(0)),
                                       static_cast<int>(q.shape(1)),
                                       static_cast<int>(k_ctx_data.shape(1)),
                                       static_cast<int>(q.shape(2)),
                                       scale,
                                       window,
                                       BitsOf(q),
                                       reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("q"),
        py::arg("k_ctx_data"),
        py::arg("k_ctx_param"),
        py::arg("v_ctx_data"),
        py::arg("v_ctx_param"),
        py::arg("k_blk"),
        py::arg("v_blk"),
        py::arg("out"),
        py::arg("scale"),
        py::arg("window"),
        py::arg("stream_ptr"));

    module.def(
        "dflash2_residual_rmsnorm",
        [](py::handle out_object, py::handle residual_object, py::handle x_object, py::handle w_object,
           float eps, float x_scale, uintptr_t stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(out_object)};
            auto out      = detail::ConsumeDLPackWithStrides(out_object, stream_ptr);
            auto residual = detail::ConsumeDLPackWithStrides(residual_object, stream_ptr);
            auto x        = detail::ConsumeDLPackWithStrides(x_object, stream_ptr);
            auto w        = detail::ConsumeDLPackWithStrides(w_object, stream_ptr);
            invokeDflash2ResidualRMSNorm(out.data_or(static_cast<void*>(nullptr)),
                                         reinterpret_cast<float*>(residual.data_or(static_cast<void*>(nullptr))),
                                         x.data_or(static_cast<const void*>(nullptr)),
                                         w.data_or(static_cast<const void*>(nullptr)),
                                         static_cast<int>(out.shape(0)),
                                         static_cast<int>(out.shape(1)),
                                         eps,
                                         1,
                                         x_scale,
                                         BitsOf(out),
                                         reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("out"), py::arg("residual"), py::arg("x"), py::arg("w"), py::arg("eps"), py::arg("x_scale"),
        py::arg("stream_ptr"));

    module.def(
        "dflash2_rope",
        [](py::handle x_object, py::handle positions_object, float base, uintptr_t stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(x_object)};
            auto x         = detail::ConsumeDLPackWithStrides(x_object, stream_ptr);
            auto positions = detail::ConsumeDLPackWithStrides(positions_object, stream_ptr);
            invokeDflash2Rope(x.data_or(static_cast<void*>(nullptr)),
                              reinterpret_cast<const int*>(positions.data_or(static_cast<int*>(nullptr))),
                              static_cast<int>(x.shape(0)),
                              static_cast<int>(x.shape(1)),
                              static_cast<int>(x.shape(2)),
                              base,
                              BitsOf(x),
                              reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("x"),
        py::arg("positions"),
        py::arg("base"),
        py::arg("stream_ptr"));
}
}  // namespace turbomind::python
