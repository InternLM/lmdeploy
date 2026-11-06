# Copyright (c) OpenMMLab. All rights reserved.

import torch


def initialize_target_verification(
    block_logits_active: torch.Tensor,
    effective_history: torch.Tensor,
    verification_draft_ids: torch.Tensor,
    request_token_ids_ptrs: torch.Tensor,
    entry_sequence_length: torch.Tensor,
    finished_on_entry: torch.Tensor,
    speculative_row: torch.Tensor,
    accepted_draft_count: torch.Tensor | None,
    request_to_generation_row_offsets: torch.Tensor,
) -> None:
    from lmdeploy.turbomind import _turbomind

    stream = torch.cuda.current_stream(entry_sequence_length.device)

    _turbomind.initialize_target_verification(
        block_logits_active,
        effective_history,
        verification_draft_ids,
        request_token_ids_ptrs,
        entry_sequence_length,
        finished_on_entry,
        speculative_row,
        accepted_draft_count,
        request_to_generation_row_offsets,
        stream.cuda_stream,
    )


def build_draft_extension_key_offsets(
    k_offsets: torch.Tensor,
    q_offsets: torch.Tensor,
    entry_sequence_length: torch.Tensor,
    accept_len: torch.Tensor,
    extension_index: int,
) -> None:
    from lmdeploy.turbomind import _turbomind

    stream = torch.cuda.current_stream(q_offsets.device)

    _turbomind.build_draft_extension_key_offsets(
        k_offsets,
        q_offsets,
        entry_sequence_length,
        accept_len,
        extension_index,
        stream.cuda_stream,
    )


def stop_criteria(
    token_ids_ptrs: torch.Tensor,
    sequence_length: torch.Tensor,
    stop_words: torch.Tensor | None,
    sequence_length_limit: torch.Tensor,
    finished: torch.Tensor,
) -> None:
    from lmdeploy.turbomind import _turbomind

    stream = torch.cuda.current_stream(token_ids_ptrs.device)

    _turbomind.stop_criteria(
        token_ids_ptrs,
        sequence_length,
        stop_words,
        sequence_length_limit,
        finished,
        stream.cuda_stream,
    )


def speculative_stop_criteria(
    token_ids_ptrs: torch.Tensor,
    entry_sequence_length: torch.Tensor,
    accept_len: torch.Tensor,
    stop_words: torch.Tensor | None,
    sequence_length_limit: torch.Tensor,
    finished: torch.Tensor,
) -> None:
    from lmdeploy.turbomind import _turbomind

    stream = torch.cuda.current_stream(token_ids_ptrs.device)

    _turbomind.speculative_stop_criteria(
        token_ids_ptrs,
        entry_sequence_length,
        accept_len,
        stop_words,
        sequence_length_limit,
        finished,
        stream.cuda_stream,
    )


def build_draft_refresh_inputs(
    draft_input_ids: torch.Tensor,
    selected_token_pos: torch.Tensor,
    candidate_active: torch.Tensor,
    token_ids_ptrs: torch.Tensor,
    refresh_q_offsets: torch.Tensor,
    refresh_k_offsets: torch.Tensor,
    extension_q_offsets: torch.Tensor,
    accept_len: torch.Tensor,
    limit_to_accept_len: torch.Tensor,
    finished: torch.Tensor,
) -> None:
    from lmdeploy.turbomind import _turbomind

    stream = torch.cuda.current_stream(refresh_q_offsets.device)

    _turbomind.build_draft_refresh_inputs(
        draft_input_ids,
        selected_token_pos,
        candidate_active,
        token_ids_ptrs,
        refresh_q_offsets,
        refresh_k_offsets,
        extension_q_offsets,
        accept_len,
        limit_to_accept_len,
        finished,
        stream.cuda_stream,
    )


def draft_argmax_and_store_token(
    logits: torch.Tensor,
    proposal_ids: torch.Tensor,
    token_ids_ptrs: torch.Tensor,
    extension_q_offsets: torch.Tensor,
    candidate_active: torch.Tensor,
    entry_sequence_length: torch.Tensor,
    accept_len: torch.Tensor,
    proposal_index: int,
    vocab_size: int,
) -> None:
    from lmdeploy.turbomind import _turbomind

    stream = torch.cuda.current_stream(logits.device)

    _turbomind.draft_argmax_and_store_token(
        logits,
        proposal_ids,
        token_ids_ptrs,
        extension_q_offsets,
        candidate_active,
        entry_sequence_length,
        accept_len,
        proposal_index,
        vocab_size,
        stream.cuda_stream,
    )
