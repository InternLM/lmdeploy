# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.kpool import (
    gather_kpool_token_tail,
    read_raw_tail,
    update_kpool,
    update_raw_kpool,
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

    # Each verification includes the root token and three draft tokens. Keep
    # only the accepted prefix, then propose again across pool/ring boundaries.
    committed_keys, committed_scores = new_keys, new_scores
    history += lengths
    lengths.fill_(4)
    starts[-1] = 4
    one = torch.ones_like(lengths)
    one_starts = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
    for round_id, accepted in enumerate((1, 2, 4, 1, 3)):
        trial_keys = torch.arange(32, dtype=torch.bfloat16, device='cuda').reshape(4, 8) + 64 * (round_id + 1)
        trial_scores = trial_keys / 8
        tail_len = committed_keys.size(0) % 4
        prefix_keys = committed_keys[-tail_len:] if tail_len else committed_keys[:0]
        prefix_scores = committed_scores[-tail_len:] if tail_len else committed_scores[:0]
        closed_keys = torch.cat((prefix_keys, trial_keys))[:4].unsqueeze(0)
        closed_scores = torch.cat((prefix_scores, trial_scores))[:4].unsqueeze(0)
        target_keys, target_scores, groups, valid = update_raw_kpool(
            trial_keys, trial_scores, raw_keys, raw_scores, state_ids, lengths, history + lengths, starts, 4)
        torch.testing.assert_close(target_keys[valid], closed_keys, rtol=0, atol=0)
        torch.testing.assert_close(target_scores[valid], closed_scores, rtol=0, atol=0)
        torch.testing.assert_close(groups[valid], history // 4)

        for step in range(4):
            end = history + step + 1
            tail_k, tail_s, draft_ids = gather_kpool_token_tail(token_cache, blocks, one, end, 4)
            keys, scores, draft_groups, closed = update_kpool(
                trial_keys[None, step:step + 1], trial_scores[None, step:step + 1],
                tail_k, tail_s, draft_ids, history + step, 4)
            torch.testing.assert_close(closed, valid[step:step + 1])
            torch.testing.assert_close(keys[closed], target_keys[step:step + 1][closed], rtol=0, atol=0)
            torch.testing.assert_close(scores[closed], target_scores[step:step + 1][closed], rtol=0, atol=0)
            torch.testing.assert_close(draft_groups, groups[step:step + 1])
            write_kpool_token_cache(token_cache, trial_keys[step:step + 1], trial_scores[step:step + 1],
                                     blocks, one, end, one_starts)

        history += accepted
        committed_keys = torch.cat((committed_keys, trial_keys[:accepted]))
        committed_scores = torch.cat((committed_scores, trial_scores[:accepted]))
        tail_len = committed_keys.size(0) % 4
        expected_keys.zero_()
        expected_scores.zero_()
        if tail_len:
            expected_keys[0, :tail_len] = committed_keys[-tail_len:]
            expected_scores[0, :tail_len] = committed_scores[-tail_len:]
        target = read_raw_tail(raw_keys, raw_scores, state_ids, history)
        draft = gather_kpool_token_tail(token_cache, blocks, one, history + one, 4)
        for keys, scores in (target, draft[:2]):
            torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
            torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('depth', [0, 3, 5])
def test_aligned_prefix_resume_ignores_stale_raw_ring(depth):
    from lmdeploy.pytorch.kernels.cuda.kpool import read_raw_tail, update_raw_kpool

    torch.manual_seed(42)
    steps, width = depth + 1, 16
    # Strided layer views, mixed query lengths and a previously used state row.
    keys = torch.randn(2 * steps, width, device='cuda', dtype=torch.bfloat16)
    scores = torch.randn_like(keys)
    ids = torch.tensor([2, 1], device='cuda')
    history = torch.tensor([64, 128], device='cuda')
    lengths = torch.tensor([steps, 1], device='cuda')
    starts = torch.tensor([0, steps], device='cuda')
    results = []
    for fill in (0., float('nan')):
        raw_k = torch.full((4, 2, depth + 5, width), fill, device='cuda', dtype=keys.dtype)[:, 1]
        raw_s = torch.full_like(raw_k, fill)
        for tail in read_raw_tail(raw_k, raw_s, ids, history):
            assert torch.count_nonzero(tail) == 0
        closed_k, closed_s, groups, valid = update_raw_kpool(
            keys, scores, raw_k, raw_s, ids, lengths, history + lengths, starts, 4)
        results.append((closed_k[valid], closed_s[valid], groups[valid]))
        # Rejection may resume at any accepted prefix, not at the trial end.
        for accepted in range(steps + 1):
            accepted_history = history + torch.tensor([accepted, min(accepted, 1)], device='cuda')
            tails = read_raw_tail(raw_k, raw_s, ids, accepted_history)
            for tail, projected in zip(tails, (keys, scores)):
                for row, (start, count) in enumerate(((0, accepted), (steps, min(accepted, 1)))):
                    n = count % 4
                    torch.testing.assert_close(tail[row, :n], projected[start + count - n:start + count],
                                               rtol=0, atol=0)
                    assert torch.count_nonzero(tail[row, n:]) == 0
    for zero, poisoned in zip(*results):
        torch.testing.assert_close(zero, poisoned, rtol=0, atol=0)
