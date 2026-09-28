#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <pybind11/pybind11.h>

#include "src/turbomind/kernels/sampling_kernels.h"
#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/kernels/speculative_sampling_kernels.h"
#include "src/turbomind/models/llama/llama_kernels.h"
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

}  // namespace

void BindSpeculativeSampling(py::module_& module)
{
    module.def(
        "initialize_speculative_sampling_states",
        [](py::handle random_states_object,
           int        random_state_count,
           py::handle random_seeds_object,
           py::handle initialize_object,
           uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(random_states_object)};
            auto            random_states = detail::ConsumeDLPackWithStrides(random_states_object, stream_ptr);
            auto            random_seeds  = detail::ConsumeDLPackWithStrides(random_seeds_object, stream_ptr);
            auto            initialize    = detail::ConsumeDLPackWithStrides(initialize_object, stream_ptr);

            if (random_state_count == 0) {
                return;
            }

            InitializeRandomStates(reinterpret_cast<curandState_t*>(random_states.raw_data()),
                                   random_seeds.data<uint64_t>(),
                                   initialize.data<bool>(),
                                   static_cast<size_t>(random_state_count),
                                   reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("random_states"),
        py::arg("random_state_count"),
        py::arg("random_seeds"),
        py::arg("initialize"),
        py::arg("stream_ptr"));

    module.def(
        "verify_target_block",
        [](py::handle probabilities_object,
           py::handle probability_token_ids_object,
           py::handle kept_count_object,
           py::handle verification_draft_ids_object,
           py::handle greedy_object,
           py::handle logits_active_object,
           py::handle random_states_object,
           py::handle random_state_indices_object,
           py::handle request_token_ids_ptrs_object,
           py::handle entry_sequence_length_object,
           py::handle request_to_generation_offsets_object,
           py::handle speculative_row_object,
           py::handle selected_span_ids_object,
           py::handle accept_len_object,
           py::handle accepted_draft_count_object,
           int        position_count,
           uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(probabilities_object)};
            auto            probabilities = detail::ConsumeDLPackWithStrides(probabilities_object, stream_ptr);
            auto probability_token_ids    = detail::ConsumeDLPackWithStrides(probability_token_ids_object, stream_ptr);
            auto kept_count               = detail::ConsumeDLPackWithStrides(kept_count_object, stream_ptr);
            auto verification_draft_ids = detail::ConsumeDLPackWithStrides(verification_draft_ids_object, stream_ptr);
            auto greedy                 = detail::ConsumeDLPackWithStrides(greedy_object, stream_ptr);
            auto logits_active          = detail::ConsumeDLPackWithStrides(logits_active_object, stream_ptr);
            auto random_states          = detail::ConsumeDLPackWithStrides(random_states_object, stream_ptr);
            auto random_state_indices   = detail::ConsumeDLPackWithStrides(random_state_indices_object, stream_ptr);
            auto request_token_ids_ptrs =
                detail::ConsumeDLPackWithStrides(request_token_ids_ptrs_object, stream_ptr);
            auto entry_sequence_length =
                detail::ConsumeDLPackWithStrides(entry_sequence_length_object, stream_ptr);
            auto request_to_generation_offsets =
                detail::ConsumeDLPackWithStrides(request_to_generation_offsets_object, stream_ptr);
            auto speculative_row   = detail::ConsumeDLPackWithStrides(speculative_row_object, stream_ptr);
            auto selected_span_ids = detail::ConsumeDLPackWithStrides(selected_span_ids_object, stream_ptr);
            auto accept_len         = detail::ConsumeDLPackWithStrides(accept_len_object, stream_ptr);
            auto accepted_draft_count =
                accepted_draft_count_object.is_none() ?
                    core::Tensor{} :
                    detail::ConsumeDLPackWithStrides(accepted_draft_count_object, stream_ptr);

            VerifyTargetBlockParams params{};
            params.probabilities          = probabilities.data_or(static_cast<float*>(nullptr));
            params.probability_stride     = static_cast<int>(probabilities.stride(0));
            params.probability_token_ids  = probability_token_ids.data_or(static_cast<int*>(nullptr));
            params.token_id_stride        = static_cast<int>(probability_token_ids.stride(0));
            params.kept_count             = kept_count.data_or(static_cast<int*>(nullptr));
            params.verification_draft_ids = verification_draft_ids.data_or(static_cast<int*>(nullptr));
            params.draft_row_stride       = static_cast<int>(verification_draft_ids.stride(0));
            params.greedy                 = greedy.data_or(static_cast<bool*>(nullptr));
            params.logits_active          = logits_active.data_or(static_cast<bool*>(nullptr));
            params.random_states =
                reinterpret_cast<curandState_t*>(random_states.data_or(static_cast<uint8_t*>(nullptr)));
            params.random_state_indices = random_state_indices.data_or(static_cast<int*>(nullptr));
            params.request_token_ids_ptrs = reinterpret_cast<int* const*>(
                request_token_ids_ptrs.data_or(static_cast<int64_t*>(nullptr)));
            params.entry_sequence_length = entry_sequence_length.data_or(static_cast<int*>(nullptr));
            params.request_to_generation_offsets =
                request_to_generation_offsets.data_or(static_cast<int*>(nullptr));
            params.speculative_row      = speculative_row.data_or(static_cast<bool*>(nullptr));
            params.selected_span_ids    = selected_span_ids.data_or(static_cast<int*>(nullptr));
            params.selected_span_stride = static_cast<int>(selected_span_ids.stride(0));
            params.accept_len           = accept_len.data_or(static_cast<int*>(nullptr));
            params.accepted_draft_count = accepted_draft_count_object.is_none() ?
                                              nullptr :
                                              accepted_draft_count.data_or(static_cast<int*>(nullptr));
            params.request_count    = static_cast<int>(entry_sequence_length.shape(0));
            params.generation_count = static_cast<int>(random_state_indices.shape(0));
            params.position_count   = position_count;

            invokeVerifyTargetBlock(params, reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("probabilities"),
        py::arg("probability_token_ids"),
        py::arg("kept_count"),
        py::arg("verification_draft_ids"),
        py::arg("greedy"),
        py::arg("logits_active"),
        py::arg("random_states"),
        py::arg("random_state_indices"),
        py::arg("request_token_ids_ptrs"),
        py::arg("entry_sequence_length"),
        py::arg("request_to_generation_offsets"),
        py::arg("speculative_row"),
        py::arg("selected_span_ids"),
        py::arg("accept_len"),
        py::arg("accepted_draft_count"),
        py::arg("position_count"),
        py::arg("stream_ptr"));

    module.def(
        "sample_processed_probabilities",
        [](py::handle probabilities_object,
           py::handle indices_object,
           py::handle kept_object,
           py::handle curand_states_object,
           py::handle curand_state_indices_object,
           py::handle sample_mask_object,
           py::handle selected_tokens_object,
           py::handle sampled_logprobs_object,
           py::handle sampled_indexes_object,
           py::handle sampled_nums_object,
           uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(probabilities_object)};
            auto            probabilities = detail::ConsumeDLPackWithStrides(probabilities_object, stream_ptr);
            auto            indices       = detail::ConsumeDLPackWithStrides(indices_object, stream_ptr);
            auto            kept          = detail::ConsumeDLPackWithStrides(kept_object, stream_ptr);
            auto            curand_states = detail::ConsumeDLPackWithStrides(curand_states_object, stream_ptr);
            auto curand_state_indices     = detail::ConsumeDLPackWithStrides(curand_state_indices_object, stream_ptr);
            auto sample_mask              = sample_mask_object.is_none() ?
                                                core::Tensor{} :
                                                detail::ConsumeDLPackWithStrides(sample_mask_object, stream_ptr);
            auto selected_tokens          = detail::ConsumeDLPackWithStrides(selected_tokens_object, stream_ptr);
            auto sampled_logprobs         = sampled_logprobs_object.is_none() ?
                                                core::Tensor{} :
                                                detail::ConsumeDLPackWithStrides(sampled_logprobs_object, stream_ptr);
            auto sampled_indexes          = sampled_indexes_object.is_none() ?
                                                core::Tensor{} :
                                                detail::ConsumeDLPackWithStrides(sampled_indexes_object, stream_ptr);
            auto sampled_nums             = sampled_nums_object.is_none() ?
                                                core::Tensor{} :
                                                detail::ConsumeDLPackWithStrides(sampled_nums_object, stream_ptr);

            SamplingParams params{};
            params.probabilities = probabilities.data_or(static_cast<float*>(nullptr));
            params.stride        = static_cast<int>(probabilities.stride(0));
            params.indices       = indices.data_or(static_cast<int*>(nullptr));
            params.kept          = kept.data_or(static_cast<int*>(nullptr));
            params.curandstate =
                reinterpret_cast<curandState_t*>(curand_states.data_or(static_cast<uint8_t*>(nullptr)));
            params.curandstate_indices = curand_state_indices.data_or(static_cast<int*>(nullptr));
            params.sample_mask =
                sample_mask_object.is_none() ? nullptr : sample_mask.data_or(static_cast<bool*>(nullptr));
            params.batch_size      = static_cast<size_t>(probabilities.shape(0));
            params.selected_tokens = selected_tokens.data_or(static_cast<int*>(nullptr));
            params.sampled_logprobs =
                sampled_logprobs_object.is_none() ? nullptr : sampled_logprobs.data_or(static_cast<float*>(nullptr));
            params.sampled_indexes =
                sampled_indexes_object.is_none() ? nullptr : sampled_indexes.data_or(static_cast<int*>(nullptr));
            params.sampled_nums =
                sampled_nums_object.is_none() ? nullptr : sampled_nums.data_or(static_cast<int*>(nullptr));

            invokeSampling<float>(params, reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("probabilities"),
        py::arg("indices"),
        py::arg("kept"),
        py::arg("curand_states"),
        py::arg("curand_state_indices"),
        py::arg("sample_mask"),
        py::arg("selected_tokens"),
        py::arg("sampled_logprobs"),
        py::arg("sampled_indexes"),
        py::arg("sampled_nums"),
        py::arg("stream_ptr"));

    module.def(
        "append_one_token_and_advance_sequence",
        [](py::handle token_ids_ptrs_object,
           py::handle selected_tokens_object,
           py::handle sequence_length_object,
           uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(token_ids_ptrs_object)};
            auto            token_ids_ptrs  = detail::ConsumeDLPackWithStrides(token_ids_ptrs_object, stream_ptr);
            auto            selected_tokens = detail::ConsumeDLPackWithStrides(selected_tokens_object, stream_ptr);
            auto            sequence_length = detail::ConsumeDLPackWithStrides(sequence_length_object, stream_ptr);

            invokeAppendOneTokenAndAdvanceSequence(
                reinterpret_cast<int* const*>(token_ids_ptrs.data_or(static_cast<int64_t*>(nullptr))),
                selected_tokens.data_or(static_cast<int*>(nullptr)),
                sequence_length.data_or(static_cast<int*>(nullptr)),
                static_cast<int>(token_ids_ptrs.shape(0)),
                reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("token_ids_ptrs"),
        py::arg("selected_tokens"),
        py::arg("sequence_length"),
        py::arg("stream_ptr"));
}

}  // namespace turbomind::python
