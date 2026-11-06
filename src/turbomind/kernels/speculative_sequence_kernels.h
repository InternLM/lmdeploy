// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"

namespace turbomind {

void invokeBuildTargetInputs(int*              target_input_ids,
                             int*              target_key_lengths,
                             const int* const* request_token_ids_ptrs,
                             const int*        target_q_offsets,
                             const int*        sequence_length,
                             const bool*       target_ids_from_row,
                             const bool*       finished,
                             int               request_count,
                             cudaStream_t      stream);

void invokeBuildDraftExtensionKeyOffsets(int*         k_offsets,
                                         const int*   q_offsets,
                                         const int*   entry_sequence_length,
                                         const int*   accept_len,
                                         int          batch_size,
                                         int          extension_index,
                                         cudaStream_t stream);

void invokeInitializeTargetVerification(bool*             block_logits_active,
                                        int*              effective_history,
                                        int*              verification_draft_ids,
                                        const int* const* request_token_ids_ptrs,
                                        const int*        entry_sequence_length,
                                        const bool*       finished_on_entry,
                                        const bool*       speculative_row,
                                        int*              accepted_draft_count,
                                        const int*        request_to_generation_row_offsets,
                                        int               request_count,
                                        int               generation_count,
                                        int               position_count,
                                        cudaStream_t      stream);

void invokeBuildDraftRefreshInputs(int*              draft_input_ids,
                                   int*              selected_token_pos,
                                   bool*             candidate_active,
                                   const int* const* token_ids_ptrs,
                                   const int*        refresh_q_offsets,
                                   const int*        refresh_k_offsets,
                                   const int*        extension_q_offsets,
                                   const int*        accept_len,
                                   const bool*       limit_to_accept_len,
                                   const bool*       finished,
                                   int               draft_input_count,
                                   int               batch_size,
                                   int               candidate_count,
                                   cudaStream_t      stream);

void invokeDraftArgmaxAndStoreToken(const Tensor& logits,
                                    int*          proposal_ids,
                                    int* const*   token_ids_ptrs,
                                    const int*    extension_q_offsets,
                                    const bool*   candidate_active,
                                    const int*    entry_sequence_length,
                                    const int*    accept_len,
                                    int           batch_size,
                                    int           candidate_count,
                                    int           proposal_index,
                                    int           vocab_size,
                                    cudaStream_t  stream);

void invokeAdvanceSequenceByAcceptedSpan(
    int* sequence_length, const int* accept_len, int* accepted_draft_count, int batch_size, cudaStream_t stream);

}  // namespace turbomind
