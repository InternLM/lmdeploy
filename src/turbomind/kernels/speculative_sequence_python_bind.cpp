// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>

#include "src/turbomind/kernels/speculative_sequence_kernels.h"
#include "src/turbomind/kernels/stop_criteria_kernels.h"
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

detail::Tensor ConsumeStopWords(py::handle object, int batch_size, uintptr_t stream_ptr, int& width)
{
    if (object.is_none()) {
        width = 0;
        return {};
    }

    width                = object.attr("shape").attr("__getitem__")(2).cast<int>();
    py::object flattened = object.attr("view")(batch_size, 2 * width);
    return detail::ConsumeDLPackWithStrides(flattened, stream_ptr);
}

}  // namespace

void BindSpeculativeSequence(py::module_& module)
{
    module.def(
        "initialize_target_verification",
        [](py::handle block_logits_active_object,
           py::handle effective_history_object,
           py::handle verification_draft_ids_object,
           py::handle request_token_ids_ptrs_object,
           py::handle entry_sequence_length_object,
           py::handle finished_on_entry_object,
           py::handle speculative_row_object,
           py::handle accepted_draft_count_object,
            py::handle request_to_generation_row_offsets_object,
            uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(entry_sequence_length_object)};
            auto block_logits_active = detail::ConsumeDLPackWithStrides(block_logits_active_object, stream_ptr);
            auto effective_history      = detail::ConsumeDLPackWithStrides(effective_history_object, stream_ptr);
            auto verification_draft_ids = detail::ConsumeDLPackWithStrides(verification_draft_ids_object, stream_ptr);
            auto request_token_ids_ptrs = detail::ConsumeDLPackWithStrides(request_token_ids_ptrs_object, stream_ptr);
            auto entry_sequence_length  = detail::ConsumeDLPackWithStrides(entry_sequence_length_object, stream_ptr);
            auto finished_on_entry      = detail::ConsumeDLPackWithStrides(finished_on_entry_object, stream_ptr);
            auto speculative_row        = detail::ConsumeDLPackWithStrides(speculative_row_object, stream_ptr);
            detail::Tensor accepted_draft_count;
            if (!accepted_draft_count_object.is_none()) {
                accepted_draft_count = detail::ConsumeDLPackWithStrides(accepted_draft_count_object, stream_ptr);
            }
            auto request_to_generation_row_offsets =
                detail::ConsumeDLPackWithStrides(request_to_generation_row_offsets_object, stream_ptr);

            invokeInitializeTargetVerification(
                block_logits_active.data_or(static_cast<bool*>(nullptr)),
                effective_history.data_or(static_cast<int*>(nullptr)),
                verification_draft_ids.data_or(static_cast<int*>(nullptr)),
                reinterpret_cast<const int* const*>(request_token_ids_ptrs.data_or(static_cast<int64_t*>(nullptr))),
                entry_sequence_length.data_or(static_cast<int*>(nullptr)),
                finished_on_entry.data_or(static_cast<bool*>(nullptr)),
                speculative_row.data_or(static_cast<bool*>(nullptr)),
                accepted_draft_count.data_or(static_cast<int*>(nullptr)),
                request_to_generation_row_offsets.data_or(static_cast<int*>(nullptr)),
                static_cast<int>(entry_sequence_length.shape(0)),
                static_cast<int>(block_logits_active.shape(1)),
                static_cast<int>(block_logits_active.shape(0)),
                reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("block_logits_active"),
        py::arg("effective_history"),
        py::arg("verification_draft_ids"),
        py::arg("request_token_ids_ptrs"),
        py::arg("entry_sequence_length"),
        py::arg("finished_on_entry"),
        py::arg("speculative_row"),
        py::arg("accepted_draft_count"),
        py::arg("request_to_generation_row_offsets"),
        py::arg("stream_ptr"));

    module.def(
        "build_draft_extension_key_offsets",
        [](py::handle k_offsets_object,
           py::handle q_offsets_object,
           py::handle entry_sequence_length_object,
           py::handle accept_len_object,
           int        extension_index,
           uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(q_offsets_object)};
            auto            k_offsets  = detail::ConsumeDLPackWithStrides(k_offsets_object, stream_ptr);
            auto            q_offsets  = detail::ConsumeDLPackWithStrides(q_offsets_object, stream_ptr);
            auto entry_sequence_length = detail::ConsumeDLPackWithStrides(entry_sequence_length_object, stream_ptr);
            auto accept_len            = detail::ConsumeDLPackWithStrides(accept_len_object, stream_ptr);

            invokeBuildDraftExtensionKeyOffsets(k_offsets.data_or(static_cast<int*>(nullptr)),
                                                q_offsets.data_or(static_cast<int*>(nullptr)),
                                                entry_sequence_length.data_or(static_cast<int*>(nullptr)),
                                                accept_len.data_or(static_cast<int*>(nullptr)),
                                                static_cast<int>(entry_sequence_length.shape(0)),
                                                extension_index,
                                                reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("k_offsets"),
        py::arg("q_offsets"),
        py::arg("entry_sequence_length"),
        py::arg("accept_len"),
        py::arg("extension_index"),
        py::arg("stream_ptr"));

    module.def(
        "build_draft_refresh_inputs",
        [](py::handle draft_input_ids_object,
           py::handle selected_token_pos_object,
           py::handle candidate_active_object,
           py::handle token_ids_ptrs_object,
           py::handle refresh_q_offsets_object,
           py::handle refresh_k_offsets_object,
           py::handle extension_q_offsets_object,
           py::handle accept_len_object,
           py::handle limit_to_accept_len_object,
           py::handle finished_object,
           uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(token_ids_ptrs_object)};
            auto            draft_input_ids = detail::ConsumeDLPackWithStrides(draft_input_ids_object, stream_ptr);
            auto selected_token_pos         = detail::ConsumeDLPackWithStrides(selected_token_pos_object, stream_ptr);
            auto candidate_active           = detail::ConsumeDLPackWithStrides(candidate_active_object, stream_ptr);
            auto token_ids_ptrs             = detail::ConsumeDLPackWithStrides(token_ids_ptrs_object, stream_ptr);
            auto refresh_q_offsets          = detail::ConsumeDLPackWithStrides(refresh_q_offsets_object, stream_ptr);
            auto refresh_k_offsets          = detail::ConsumeDLPackWithStrides(refresh_k_offsets_object, stream_ptr);
            auto extension_q_offsets        = detail::ConsumeDLPackWithStrides(extension_q_offsets_object, stream_ptr);
            auto accept_len                 = detail::ConsumeDLPackWithStrides(accept_len_object, stream_ptr);
            auto limit_to_accept_len        = detail::ConsumeDLPackWithStrides(limit_to_accept_len_object, stream_ptr);
            auto finished                   = detail::ConsumeDLPackWithStrides(finished_object, stream_ptr);

            invokeBuildDraftRefreshInputs(
                draft_input_ids.data_or(static_cast<int*>(nullptr)),
                selected_token_pos.data_or(static_cast<int*>(nullptr)),
                candidate_active.data_or(static_cast<bool*>(nullptr)),
                reinterpret_cast<const int* const*>(token_ids_ptrs.data_or(static_cast<int64_t*>(nullptr))),
                refresh_q_offsets.data_or(static_cast<int*>(nullptr)),
                refresh_k_offsets.data_or(static_cast<int*>(nullptr)),
                extension_q_offsets.data_or(static_cast<int*>(nullptr)),
                accept_len.data_or(static_cast<int*>(nullptr)),
                limit_to_accept_len.data_or(static_cast<bool*>(nullptr)),
                finished.data_or(static_cast<bool*>(nullptr)),
                static_cast<int>(draft_input_ids.shape(0)),
                static_cast<int>(accept_len.shape(0)),
                static_cast<int>(selected_token_pos.shape(0)),
                reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("draft_input_ids"),
        py::arg("selected_token_pos"),
        py::arg("candidate_active"),
        py::arg("token_ids_ptrs"),
        py::arg("refresh_q_offsets"),
        py::arg("refresh_k_offsets"),
        py::arg("extension_q_offsets"),
        py::arg("accept_len"),
        py::arg("limit_to_accept_len"),
        py::arg("finished"),
        py::arg("stream_ptr"));

    module.def(
        "draft_argmax_and_store_token",
        [](py::handle logits_object,
           py::handle proposal_ids_object,
           py::handle token_ids_ptrs_object,
           py::handle extension_q_offsets_object,
           py::handle candidate_active_object,
           py::handle entry_sequence_length_object,
           py::handle accept_len_object,
           int        proposal_index,
           int        vocab_size,
           uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(logits_object)};
            auto            logits         = detail::ConsumeDLPackWithStrides(logits_object, stream_ptr);
            auto            proposal_ids   = detail::ConsumeDLPackWithStrides(proposal_ids_object, stream_ptr);
            auto            token_ids_ptrs = detail::ConsumeDLPackWithStrides(token_ids_ptrs_object, stream_ptr);
            auto extension_q_offsets       = detail::ConsumeDLPackWithStrides(extension_q_offsets_object, stream_ptr);
            auto candidate_active          = detail::ConsumeDLPackWithStrides(candidate_active_object, stream_ptr);
            auto entry_sequence_length     = detail::ConsumeDLPackWithStrides(entry_sequence_length_object, stream_ptr);
            auto accept_len                = detail::ConsumeDLPackWithStrides(accept_len_object, stream_ptr);

            invokeDraftArgmaxAndStoreToken(
                logits,
                proposal_ids.data_or(static_cast<int*>(nullptr)),
                reinterpret_cast<int* const*>(token_ids_ptrs.data_or(static_cast<int64_t*>(nullptr))),
                extension_q_offsets.data_or(static_cast<int*>(nullptr)),
                candidate_active.data_or(static_cast<bool*>(nullptr)),
                entry_sequence_length.data_or(static_cast<int*>(nullptr)),
                accept_len.data_or(static_cast<int*>(nullptr)),
                static_cast<int>(token_ids_ptrs.shape(0)),
                static_cast<int>(logits.shape(0)),
                proposal_index,
                vocab_size,
                reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("logits"),
        py::arg("proposal_ids"),
        py::arg("token_ids_ptrs"),
        py::arg("extension_q_offsets"),
        py::arg("candidate_active"),
        py::arg("entry_sequence_length"),
        py::arg("accept_len"),
        py::arg("proposal_index"),
        py::arg("vocab_size"),
        py::arg("stream_ptr"));

    module.def(
        "stop_criteria",
        [](py::handle token_ids_ptrs_object,
           py::handle sequence_length_object,
           py::handle stop_words_object,
           py::handle sequence_length_limit_object,
           py::handle finished_object,
           uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(token_ids_ptrs_object)};
            auto            token_ids_ptrs   = detail::ConsumeDLPackWithStrides(token_ids_ptrs_object, stream_ptr);
            auto            sequence_length  = detail::ConsumeDLPackWithStrides(sequence_length_object, stream_ptr);
            int             stop_words_width = 0;
            auto            stop_words       = ConsumeStopWords(
                stop_words_object, static_cast<int>(sequence_length.shape(0)), stream_ptr, stop_words_width);
            auto sequence_length_limit = detail::ConsumeDLPackWithStrides(sequence_length_limit_object, stream_ptr);
            auto finished              = detail::ConsumeDLPackWithStrides(finished_object, stream_ptr);

            invokeStopCriteria(
                reinterpret_cast<const int* const*>(token_ids_ptrs.data_or(static_cast<int64_t*>(nullptr))),
                sequence_length.data_or(static_cast<int*>(nullptr)),
                stop_words.data_or(static_cast<int*>(nullptr)),
                stop_words_width,
                sequence_length_limit.data_or(static_cast<int*>(nullptr)),
                finished.data_or(static_cast<bool*>(nullptr)),
                static_cast<int>(sequence_length.shape(0)),
                reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("token_ids_ptrs"),
        py::arg("sequence_length"),
        py::arg("stop_words"),
        py::arg("sequence_length_limit"),
        py::arg("finished"),
        py::arg("stream_ptr"));

    module.def(
        "speculative_stop_criteria",
        [](py::handle token_ids_ptrs_object,
           py::handle entry_sequence_length_object,
           py::handle accept_len_object,
           py::handle stop_words_object,
           py::handle sequence_length_limit_object,
           py::handle finished_object,
           uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(token_ids_ptrs_object)};
            auto            token_ids_ptrs = detail::ConsumeDLPackWithStrides(token_ids_ptrs_object, stream_ptr);
            auto entry_sequence_length     = detail::ConsumeDLPackWithStrides(entry_sequence_length_object, stream_ptr);
            auto accept_len                = detail::ConsumeDLPackWithStrides(accept_len_object, stream_ptr);
            int  stop_words_width          = 0;
            auto stop_words                = ConsumeStopWords(
                stop_words_object, static_cast<int>(entry_sequence_length.shape(0)), stream_ptr, stop_words_width);
            auto sequence_length_limit = detail::ConsumeDLPackWithStrides(sequence_length_limit_object, stream_ptr);
            auto finished              = detail::ConsumeDLPackWithStrides(finished_object, stream_ptr);

            invokeStopCriteria(
                reinterpret_cast<const int* const*>(token_ids_ptrs.data_or(static_cast<int64_t*>(nullptr))),
                entry_sequence_length.data_or(static_cast<int*>(nullptr)),
                accept_len.data_or(static_cast<int*>(nullptr)),
                stop_words.data_or(static_cast<int*>(nullptr)),
                stop_words_width,
                sequence_length_limit.data_or(static_cast<int*>(nullptr)),
                finished.data_or(static_cast<bool*>(nullptr)),
                static_cast<int>(entry_sequence_length.shape(0)),
                reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("token_ids_ptrs"),
        py::arg("entry_sequence_length"),
        py::arg("accept_len"),
        py::arg("stop_words"),
        py::arg("sequence_length_limit"),
        py::arg("finished"),
        py::arg("stream_ptr"));
}

}  // namespace turbomind::python
