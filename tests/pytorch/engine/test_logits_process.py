# yapf: disable
import torch
from transformers.generation.logits_process import (MinPLogitsWarper, RepetitionPenaltyLogitsProcessor,
                                                    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)

# yapf: enable


def test_process_temperature():
    from lmdeploy.pytorch.engine.logits_process import _process_temperature_

    batch_size = 4
    num_tokens = 16
    scores = torch.rand(batch_size, num_tokens)
    temperatures = torch.rand(batch_size)

    gt = []
    for score, temperature in zip(scores, temperatures):
        warper = TemperatureLogitsWarper(temperature.item())
        gt.append(warper(None, score[None]))
    gt = torch.cat(gt)

    out = _process_temperature_(scores, temperatures)
    torch.testing.assert_close(out, gt)


def test_process_bad_words():
    from lmdeploy.pytorch.engine.logits_process import _process_bad_words_

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
    mask = bad_words >= 0

    out_scores = _process_bad_words_(scores, bad_words.where(mask, 0), mask)

    for score, bw in zip(out_scores, bad_words):
        bw = bw.tolist()

        for w in bw:
            if w >= 0:
                assert score[w] == filter_value


def test_processrepetition_penalty():
    from lmdeploy.pytorch.engine.logits_process import _process_repetition_penalty_
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

    out = _process_repetition_penalty_(scores, input_ids, penalties)
    torch.testing.assert_close(out, gt)


def test_filter_topk_sorted():
    from lmdeploy.pytorch.engine.logits_process import _filter_topk_sorted_

    batch_size = 4
    num_tokens = 16
    scores = torch.rand(batch_size, num_tokens).sort(1, descending=True)[0]
    top_k = torch.randint(4, num_tokens - 4, (batch_size, ))

    gt = []
    for score, k in zip(scores, top_k):
        warper = TopKLogitsWarper(k.item())
        gt.append(warper(None, score[None].clone()))
    gt = torch.cat(gt)

    out = _filter_topk_sorted_(scores, top_k)
    torch.testing.assert_close(out, gt)


def test_filter_topp_sorted():
    from lmdeploy.pytorch.engine.logits_process import _filter_topp_sorted_

    batch_size = 4
    num_tokens = 16
    scores = torch.rand(batch_size, num_tokens).sort(1, descending=True)[0]
    top_p = torch.rand(batch_size)

    gt = []
    for score, p in zip(scores, top_p):
        warper = TopPLogitsWarper(p.item())
        gt.append(warper(None, score[None].clone()))
    gt = torch.cat(gt)

    out = _filter_topp_sorted_(scores, top_p)
    torch.testing.assert_close(out, gt)


def test_filter_minp_sorted():
    from lmdeploy.pytorch.engine.logits_process import _filter_minp_sorted_

    batch_size = 4
    num_tokens = 16
    scores = torch.rand(batch_size, num_tokens).sort(1, descending=True)[0]
    min_p = torch.rand(batch_size)

    gt = []
    for score, p in zip(scores, min_p):
        warper = MinPLogitsWarper(p.item())
        gt.append(warper(None, score[None].clone()))
    gt = torch.cat(gt)

    out = _filter_minp_sorted_(scores, min_p)
    torch.testing.assert_close(out, gt)


def test_filter_ngram():
    from lmdeploy.pytorch.engine.logits_process import _filter_repetition_ngram_
    vocab_size = 100

    def _get_emtas(n, window_size):
        batch_size = generated_ids.size(0)
        max_n = int(n.max().item())
        same_n = n.eq(max_n).all().item()
        max_window_size = int(window_size.max().item())
        if same_n:
            n = None
        return batch_size, max_n, max_window_size, n

    # base test
    generated_ids = torch.tensor([
        [2, 3, 4, 1, 2, 3, 4, 2, 3, 4],
        [9, 8, 7, 3, 8, 7, 5, 9, 8, 7],
        [9, 8, 7, 3, 8, 7, 5, 9, 8, 7],
    ],
                                 dtype=torch.int64)
    n = torch.tensor([3, 3, 2], dtype=torch.int64)
    threshold = torch.tensor([3, 3, 3], dtype=torch.int64)
    window_size = torch.tensor([10, 10, 10], dtype=torch.int64)

    batch_size, max_n, max_window_size, n = _get_emtas(n, window_size)
    scores = torch.rand(batch_size, vocab_size)
    stop_words = torch.randint(0, vocab_size, (batch_size, 3), dtype=torch.int64)
    _filter_repetition_ngram_(scores, stop_words, generated_ids, n, threshold, max_n, window_size, max_window_size)

    assert not scores[1].isinf().any().item()
    assert scores[0].isinf().sum().item() == vocab_size - 1
    assert scores[2].isinf().sum().item() == vocab_size - 1
    assert scores[0, stop_words[0, 0]] == 0
    assert scores[2, stop_words[2, 0]] == 0

    # test no ngram
    generated_ids = torch.tensor([
        [2, 3, 4, 1, 2, 3, 4, 2, 3, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    n = torch.tensor([3, 0], dtype=torch.int64)
    threshold = torch.tensor([3, 0], dtype=torch.int64)
    window_size = torch.tensor([10, 10], dtype=torch.int64)
    batch_size, max_n, max_window_size, n = _get_emtas(n, window_size)

    scores = torch.rand(batch_size, vocab_size)
    stop_words = torch.randint(0, vocab_size, (batch_size, 3), dtype=torch.int64)
    _filter_repetition_ngram_(scores, stop_words, generated_ids, n, threshold, max_n, window_size, max_window_size)
    assert not scores[1].isinf().any().item()
    assert scores[0].isinf().sum().item() == vocab_size - 1

    # test window
    generated_ids = torch.tensor([
        [2, 3, 4, 1, 2, 3, 4, 2, 3, 4],
        [2, 3, 4, 1, 2, 3, 4, 2, 3, 4],
    ])
    n = torch.tensor([2, 0], dtype=torch.int64)
    threshold = torch.tensor([3, 3], dtype=torch.int64)
    window_size = torch.tensor([10, 6], dtype=torch.int64)
    batch_size, max_n, max_window_size, n = _get_emtas(n, window_size)

    scores = torch.rand(batch_size, vocab_size)
    stop_words = torch.randint(0, vocab_size, (batch_size, 3), dtype=torch.int64)
    _filter_repetition_ngram_(scores, stop_words, generated_ids, n, threshold, max_n, window_size, max_window_size)
    assert not scores[1].isinf().any().item()
    assert scores[0].isinf().sum().item() == vocab_size - 1
