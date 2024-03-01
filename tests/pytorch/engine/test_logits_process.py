import pytest
import torch
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
    TopKLogitsWarper, TopPLogitsWarper)


@pytest.mark.parametrize('inplace', [True, False])
def test_process_temperature(inplace):
    from lmdeploy.pytorch.engine.logits_process import _process_temperature

    batch_size = 4
    num_tokens = 16
    scores = torch.rand(batch_size, num_tokens)
    temperatures = torch.rand(batch_size)

    gt = []
    for score, temperature in zip(scores, temperatures):
        warper = TemperatureLogitsWarper(temperature.item())
        gt.append(warper(None, score[None]))
    gt = torch.cat(gt)

    out = _process_temperature(scores, temperatures, inplace=inplace)
    torch.testing.assert_close(out, gt)


@pytest.mark.parametrize('inplace', [True, False])
def test_process_bad_words(inplace):
    from lmdeploy.pytorch.engine.logits_process import _process_bad_words

    filter_value: float = -float('inf')
    batch_size = 4
    num_tokens = 16
    scores = torch.rand(batch_size, num_tokens)
    bad_words = torch.tensor([
        [0, 1],
        [3, -1],
        [4, 4],
        [-1, -1],
    ])

    out_scores = _process_bad_words(scores, bad_words, inplace=inplace)

    for score, bw in zip(out_scores, bad_words):
        bw = bw.tolist()

        for w in bw:
            if w >= 0:
                assert score[w] == filter_value


@pytest.mark.parametrize('inplace', [True, False])
def test_processrepetition_penalty(inplace):
    from lmdeploy.pytorch.engine.logits_process import \
        _process_repetition_penalty
    batch_size = 4
    num_tokens = 16
    scores = torch.rand(batch_size, num_tokens)
    input_ids = torch.tensor([
        [0, 1],
        [3, 6],
        [4, 4],
        [0, 0],
    ])
    penalties = 1 + torch.rand(batch_size)

    gt = []
    for score, ids, penalty in zip(scores, input_ids, penalties):
        warper = RepetitionPenaltyLogitsProcessor(penalty.item())
        gt.append(warper(ids[None], score[None].clone()))
    gt = torch.cat(gt)

    out = _process_repetition_penalty(scores,
                                      input_ids,
                                      penalties,
                                      inplace=inplace)
    torch.testing.assert_close(out, gt)


@pytest.mark.parametrize('inplace', [True, False])
def test_filter_topk_sorted(inplace):
    from lmdeploy.pytorch.engine.logits_process import _filter_topk_sorted

    batch_size = 4
    num_tokens = 16
    scores = torch.rand(batch_size, num_tokens).sort(1, descending=True)[0]
    top_k = torch.randint(4, num_tokens - 4, (batch_size, ))

    gt = []
    for score, k in zip(scores, top_k):
        warper = TopKLogitsWarper(k.item())
        gt.append(warper(None, score[None].clone()))
    gt = torch.cat(gt)

    out = _filter_topk_sorted(scores, top_k, inplace=inplace)
    torch.testing.assert_close(out, gt)


@pytest.mark.parametrize('inplace', [True, False])
def test_filter_topp_sorted(inplace):
    from lmdeploy.pytorch.engine.logits_process import _filter_topp_sorted

    batch_size = 4
    num_tokens = 16
    scores = torch.rand(batch_size, num_tokens).sort(1, descending=True)[0]
    top_p = torch.rand(batch_size)

    gt = []
    for score, p in zip(scores, top_p):
        warper = TopPLogitsWarper(p.item())
        gt.append(warper(None, score[None].clone()))
    gt = torch.cat(gt)

    out = _filter_topp_sorted(scores, top_p, inplace=inplace)
    torch.testing.assert_close(out, gt)
