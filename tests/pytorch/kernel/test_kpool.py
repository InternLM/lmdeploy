# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.kpool import (
    gather_kpool_token_tail,
    read_raw_tail,
    write_kpool_token_cache,
    write_raw_ring,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_aligned_store_resume_rebuilds_target_and_mtp_tails():
    # Store restores complete 64-token owners. Poison the omitted caches to
    # prove that the first forward reads no previous tail in either model.
    raw_keys = torch.full((2, 8, 8), float('nan'), dtype=torch.bfloat16, device='cuda')
    raw_scores = torch.full_like(raw_keys, float('nan'))
    token_cache = torch.full((3, 64, 2, 8), float('nan'), dtype=torch.bfloat16, device='cuda')
    state_ids = torch.tensor([1], device='cuda')
    history = torch.tensor([64], dtype=torch.int32, device='cuda')
    lengths = torch.tensor([3], dtype=torch.int32, device='cuda')
    starts = torch.tensor([0, 3], dtype=torch.int32, device='cuda')
    blocks = torch.tensor([[2, 0]], dtype=torch.int32, device='cuda')
    expected_keys = torch.zeros((1, 4, 8), dtype=torch.bfloat16, device='cuda')
    expected_scores = torch.zeros_like(expected_keys)

    target = read_raw_tail(raw_keys, raw_scores, state_ids, history)
    draft = gather_kpool_token_tail(token_cache, blocks, lengths, history + lengths, 4)
    for keys, scores in (target, draft[:2]):
        torch.testing.assert_close(keys, expected_keys)
        torch.testing.assert_close(scores, expected_scores)

    new_keys = torch.arange(24, dtype=torch.bfloat16, device='cuda').reshape(3, 8)
    new_scores = new_keys + 32
    write_raw_ring(raw_keys, raw_scores, new_keys, new_scores, state_ids, history, lengths, starts)
    write_kpool_token_cache(token_cache, new_keys, new_scores, blocks, lengths, history + lengths, starts)
    expected_keys[0, :3] = new_keys
    expected_scores[0, :3] = new_scores

    target = read_raw_tail(raw_keys, raw_scores, state_ids, history + lengths)
    draft = gather_kpool_token_tail(token_cache, blocks, lengths, history + 2 * lengths, 4)
    for keys, scores in (target, draft[:2]):
        torch.testing.assert_close(keys, expected_keys)
        torch.testing.assert_close(scores, expected_scores)
