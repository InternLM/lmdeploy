# Copyright (c) OpenMMLab. All rights reserved.

import torch


def initialize_target_verification_reference(
    token_rows: list[torch.Tensor | None],
    entry_sequence_length: torch.Tensor,
    finished_on_entry: torch.Tensor,
    speculative_row: torch.Tensor,
    request_to_generation_row_offsets: torch.Tensor,
    draft_count: int,
    metrics_enabled: bool,
):
    offsets = request_to_generation_row_offsets.cpu().tolist()
    entry_lengths = entry_sequence_length.cpu().tolist()
    inherited_finished = finished_on_entry.cpu().tolist()
    speculative = speculative_row.cpu().tolist()
    generation_count = offsets[-1]

    block_logits_active = torch.empty(
        (draft_count + 1, generation_count),
        dtype=torch.bool,
        device=entry_sequence_length.device,
    )
    effective_history = torch.empty(
        (draft_count + 1, generation_count),
        dtype=torch.int32,
        device=entry_sequence_length.device,
    )
    verification_draft_ids = torch.empty(
        (draft_count, generation_count),
        dtype=torch.int32,
        device=entry_sequence_length.device,
    )
    accepted_draft_count = (torch.empty(
        len(token_rows),
        dtype=torch.int32,
        device=entry_sequence_length.device,
    ) if metrics_enabled else None)

    for b, row in enumerate(token_rows):
        active = not inherited_finished[b]
        verify = active and speculative[b]
        if accepted_draft_count is not None:
            accepted_draft_count[b] = 0 if verify else -1

        if offsets[b + 1] == offsets[b]:
            continue

        g = offsets[b]
        S = entry_lengths[b]
        for i in range(draft_count + 1):
            block_logits_active[i, g] = active and (i == 0 or verify)
            effective_history[i, g] = S + i
        for i in range(draft_count):
            verification_draft_ids[i, g] = row[S + i] if verify else 0

    return (
        block_logits_active,
        effective_history,
        verification_draft_ids,
        accepted_draft_count,
    )


def build_draft_extension_key_offsets_reference(
    q_offsets: torch.Tensor,
    entry_sequence_length: torch.Tensor,
    accept_len: torch.Tensor,
    extension_index: int,
) -> torch.Tensor:
    q_width = q_offsets[1:] - q_offsets[:-1]
    key_len = torch.where(
        q_width == 1,
        entry_sequence_length + accept_len + extension_index,
        0,
    )
    return torch.cat([
        torch.zeros(1, dtype=torch.int32, device=key_len.device),
        torch.cumsum(key_len, dim=0, dtype=torch.int32),
    ])


def pack_stop_words(
    phrases_by_row: list[list[list[int]]],
    width: int,
    device: torch.device | str,
) -> torch.Tensor:
    packed = torch.full(
        (len(phrases_by_row), 2, width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for b, phrases in enumerate(phrases_by_row):
        flat_words = []
        ends = []
        for phrase in phrases:
            flat_words.extend(phrase)
            ends.append(len(flat_words))
        if flat_words:
            packed[b, 0, :len(flat_words)] = torch.tensor(flat_words,
                                                          dtype=torch.int32,
                                                          device=device)
        if ends:
            packed[b, 1, :len(ends)] = torch.tensor(ends,
                                                    dtype=torch.int32,
                                                    device=device)
    return packed


def stop_criteria_reference(
    token_rows: list[torch.Tensor],
    sequence_length: torch.Tensor,
    stop_words: torch.Tensor | None,
    sequence_length_limit: torch.Tensor,
    finished: torch.Tensor,
    accept_len: torch.Tensor | None = None,
):
    output_finished = finished.clone()
    output_accept_len = None if accept_len is None else accept_len.clone()
    lengths = sequence_length.cpu().tolist()
    limits = sequence_length_limit.cpu().tolist()

    packed = None if stop_words is None else stop_words.cpu().tolist()

    for b, token_row in enumerate(token_rows):
        if output_accept_len is not None:
            if bool(output_finished[b].item()):
                output_accept_len[b] = 0
                continue
            entry_len = lengths[b]
            span_len = int(output_accept_len[b].item())
        else:
            if bool(output_finished[b].item()):
                continue
            current_len = lengths[b]
            if current_len <= 0:
                continue
            entry_len = current_len - 1
            span_len = 1

        if span_len <= 0:
            continue

        tokens = token_row.cpu().tolist()
        phrases = []
        if packed is not None:
            phrase_begin = 0
            for phrase_end in packed[b][1]:
                if phrase_end < 0:
                    break
                phrases.append(packed[b][0][phrase_begin:phrase_end])
                phrase_begin = phrase_end

        for j in range(span_len):
            effective_len = entry_len + j + 1
            terminal = effective_len >= limits[b]

            if not terminal:
                for phrase in phrases:
                    phrase_size = len(phrase)
                    if phrase_size <= 0 or effective_len < phrase_size:
                        continue
                    if tokens[effective_len -
                              phrase_size:effective_len] == phrase:
                        terminal = True
                        break

            if terminal:
                if output_accept_len is not None:
                    output_accept_len[b] = j + 1
                output_finished[b] = True
                break

    return output_accept_len, output_finished


def build_draft_refresh_inputs_reference(
    token_rows: list[torch.Tensor | None],
    refresh_q_offsets: torch.Tensor,
    refresh_k_offsets: torch.Tensor,
    extension_q_offsets: torch.Tensor,
    accept_len: torch.Tensor,
    limit_to_accept_len: torch.Tensor,
    finished: torch.Tensor,
):
    q_offsets = refresh_q_offsets.cpu().tolist()
    k_offsets = refresh_k_offsets.cpu().tolist()
    extension_offsets = extension_q_offsets.cpu().tolist()
    limiting = limit_to_accept_len.cpu().tolist()
    finished_rows = finished.cpu().tolist()

    draft_input_ids = torch.zeros(
        q_offsets[-1],
        dtype=torch.int32,
        device=refresh_q_offsets.device,
    )
    selected_token_pos = torch.zeros(
        extension_offsets[-1],
        dtype=torch.int32,
        device=refresh_q_offsets.device,
    )
    candidate_active = torch.zeros(
        extension_offsets[-1],
        dtype=torch.bool,
        device=refresh_q_offsets.device,
    )

    for b, token_row in enumerate(token_rows):
        q_begin = q_offsets[b]
        q_end = q_offsets[b + 1]
        q_len = q_end - q_begin

        if extension_offsets[b + 1] != extension_offsets[b]:
            candidate = extension_offsets[b]
            if not finished_rows[b]:
                if limiting[b]:
                    committed = int(accept_len[b].item())
                    if committed > 0:
                        candidate_active[candidate] = True
                        selected_token_pos[candidate] = q_begin + committed - 1
                elif q_len > 0:
                    candidate_active[candidate] = True
                    selected_token_pos[candidate] = q_end - 1

        if q_len == 0:
            continue

        token_begin = k_offsets[b + 1] - k_offsets[b] - q_len + 1
        valid_len = int(accept_len[b].item()) if limiting[b] else q_len
        for j in range(valid_len):
            draft_input_ids[q_begin + j] = token_row[token_begin + j]

    return draft_input_ids, selected_token_pos, candidate_active


def draft_argmax_and_store_token_reference(
    logits: torch.Tensor,
    token_rows: list[torch.Tensor | None],
    extension_q_offsets: torch.Tensor,
    candidate_active: torch.Tensor,
    entry_sequence_length: torch.Tensor,
    accept_len: torch.Tensor,
    proposal_index: int,
    vocab_size: int,
):
    output_rows = [None if row is None else row.clone() for row in token_rows]
    proposal_ids = torch.zeros(
        logits.shape[0],
        dtype=torch.int32,
        device=logits.device,
    )
    extension_offsets = extension_q_offsets.cpu().tolist()

    for b, output_row in enumerate(output_rows):
        if extension_offsets[b + 1] == extension_offsets[b]:
            continue

        candidate = extension_offsets[b]
        if not bool(candidate_active[candidate].item()):
            continue

        row = logits[candidate, :vocab_size].float()
        row = torch.where(torch.isnan(row), -torch.inf, row)
        token_id = torch.argmax(row).to(torch.int32)
        proposal_ids[candidate] = token_id
        token_position = (int(entry_sequence_length[b].item()) +
                          int(accept_len[b].item()) + proposal_index)
        output_row[token_position] = token_id

    return output_rows, proposal_ids
