import torch

from lmdeploy.turbomind import _tm

STATE_CAPACITY_PER_LOGICAL_STATE = 4096


def allocate_random_states(random_state_count: int,
                           device: torch.device | str) -> torch.Tensor:
    return torch.full(
        (random_state_count * STATE_CAPACITY_PER_LOGICAL_STATE, ),
        0xA5,
        dtype=torch.uint8,
        device=device)


def initialize_random_states(random_states: torch.Tensor,
                             random_seeds: torch.Tensor,
                             initialize: torch.Tensor) -> None:
    stream_ptr = torch.cuda.current_stream(random_states.device).cuda_stream
    _tm.initialize_speculative_sampling_states(
        random_states,
        random_seeds.numel(),
        random_seeds,
        initialize,
        stream_ptr,
    )


def verify_target_block(
    probabilities: torch.Tensor,
    probability_token_ids: torch.Tensor,
    kept_count: torch.Tensor,
    verification_draft_ids: torch.Tensor,
    greedy: torch.Tensor,
    logits_active: torch.Tensor,
    random_states: torch.Tensor,
    random_state_indices: torch.Tensor,
    request_token_ids_ptrs: torch.Tensor,
    entry_sequence_length: torch.Tensor,
    request_to_generation_offsets: torch.Tensor,
    speculative_row: torch.Tensor,
    selected_span_ids: torch.Tensor,
    accept_len: torch.Tensor,
    accepted_draft_count: torch.Tensor | None,
    position_count: int,
) -> None:
    stream_ptr = torch.cuda.current_stream(probabilities.device).cuda_stream
    _tm.verify_target_block(
        probabilities,
        probability_token_ids,
        kept_count,
        verification_draft_ids,
        greedy,
        logits_active,
        random_states,
        random_state_indices,
        request_token_ids_ptrs,
        entry_sequence_length,
        request_to_generation_offsets,
        speculative_row,
        selected_span_ids,
        accept_len,
        accepted_draft_count,
        position_count,
        stream_ptr,
    )


def sample_processed_probabilities(
    probabilities: torch.Tensor,
    indices: torch.Tensor,
    kept: torch.Tensor,
    curand_states: torch.Tensor,
    curand_state_indices: torch.Tensor,
    sample_mask: torch.Tensor | None,
    selected_tokens: torch.Tensor,
    sampled_logprobs: torch.Tensor | None,
    sampled_indexes: torch.Tensor | None,
    sampled_nums: torch.Tensor | None,
) -> None:
    stream_ptr = torch.cuda.current_stream(probabilities.device).cuda_stream
    _tm.sample_processed_probabilities(
        probabilities,
        indices,
        kept,
        curand_states,
        curand_state_indices,
        sample_mask,
        selected_tokens,
        sampled_logprobs,
        sampled_indexes,
        sampled_nums,
        stream_ptr,
    )


def append_one_token_and_advance_sequence(
    token_ids_ptrs: torch.Tensor,
    selected_tokens: torch.Tensor,
    sequence_length: torch.Tensor,
) -> None:
    stream_ptr = torch.cuda.current_stream(token_ids_ptrs.device).cuda_stream
    _tm.append_one_token_and_advance_sequence(
        token_ids_ptrs,
        selected_tokens,
        sequence_length,
        stream_ptr,
    )
