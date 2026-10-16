from collections.abc import Sequence

import torch


def greedy_reference(
        probabilities: torch.Tensor, token_ids: torch.Tensor,
        kept_count: torch.Tensor,
        draft_token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference the deterministic first-maximum greedy decision."""
    selected = []
    accepted = []
    probabilities_cpu = probabilities.cpu()
    token_ids_cpu = token_ids.cpu()
    kept_count_cpu = kept_count.cpu()
    draft_token_ids_cpu = draft_token_ids.cpu()

    for row in range(probabilities.shape[0]):
        kept = int(kept_count_cpu[row])
        target_index = int(torch.argmax(probabilities_cpu[row, :kept]))
        target_token = int(token_ids_cpu[row, target_index])
        draft_token = int(draft_token_ids_cpu[row])
        selected.append(target_token)
        accepted.append(target_token == draft_token)

    return (torch.tensor(selected, dtype=torch.int32),
            torch.tensor(accepted, dtype=torch.bool))


def recovery_token_reference(
        probabilities: torch.Tensor, token_ids: torch.Tensor,
        kept_count: torch.Tensor,
        draft_token_ids: torch.Tensor) -> Sequence[set[int]]:
    """Return the valid positive-mass non-draft corrections for each row."""
    valid_tokens = []
    probabilities_cpu = probabilities.cpu()
    token_ids_cpu = token_ids.cpu()
    kept_count_cpu = kept_count.cpu()
    draft_token_ids_cpu = draft_token_ids.cpu()

    for row in range(probabilities.shape[0]):
        kept = int(kept_count_cpu[row])
        draft_token = int(draft_token_ids_cpu[row])
        row_tokens = {
            int(token_ids_cpu[row, index])
            for index in range(kept)
            if int(token_ids_cpu[row, index]) != draft_token
            and float(probabilities_cpu[row, index]) > 0.0
        }
        valid_tokens.append(row_tokens)

    return valid_tokens
