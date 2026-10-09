# yapf: disable
import torch
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

# yapf: enable


def test_sampling_inputs_select_sampling_rows():
    from lmdeploy.pytorch.engine.logits_process import SamplingInputs

    inputs = SamplingInputs(
        temperature=torch.arange(6),
        top_k=torch.arange(6),
        top_p=torch.arange(6) / 10,
        min_p=torch.arange(6) / 20,
        random_seeds=torch.arange(10, 16),
        random_offsets=torch.arange(20, 26),
        max_top_k=5,
        has_greedy=True,
        batch_size=6,
    )

    selected = inputs.select_sampling_rows(slice(2, None, 3))

    assert selected.batch_size == 2
    torch.testing.assert_close(selected.top_k, torch.tensor([2, 5]))
    torch.testing.assert_close(selected.top_p, torch.tensor([0.2, 0.5]))
    torch.testing.assert_close(selected.min_p, torch.tensor([0.1, 0.25]))
    torch.testing.assert_close(selected.random_seeds, torch.tensor([12, 15]))
    torch.testing.assert_close(selected.random_offsets, torch.tensor([22, 25]))
    assert selected.max_top_k == 5
    assert selected.has_greedy
    assert selected.temperature is None
    assert inputs.select_sampling_rows(slice(None)) is inputs


def test_sampling_inputs_record_stream_records_only_tensor_fields():
    from lmdeploy.pytorch.engine.logits_process import SamplingInputs

    recorded = []

    class _CudaTensor(torch.Tensor):

        @staticmethod
        def __new__(cls):
            return torch.Tensor._make_subclass(cls, torch.empty(1), False)

        @property
        def is_cuda(self):
            return True

        def record_stream(self, stream):
            recorded.append((id(self), stream))

    stream = object()
    temperature = _CudaTensor()
    nested_session_tensor = _CudaTensor()
    inputs = SamplingInputs(temperature=temperature,
                            session_ctx=[{'persistent': nested_session_tensor}])

    inputs.record_stream(stream)

    assert recorded == [(id(temperature), stream)]


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

    out_scores = _process_bad_words_(scores, bad_words, mask)

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


def test_filter_topp_bfloat16_boundary():
    from lmdeploy.pytorch.engine.logits_process import _filter_topp_sorted_

    scores = torch.tensor([[2.65625, 1.984375, 1.2578125, 1.171875,
                            0.08349609375, -1.6015625, -2.328125, -2.421875]],
                          dtype=torch.bfloat16)
    # The first four tokens already cover 0.9504036 of the probability mass.
    expected = TopPLogitsWarper(0.95)(None, scores.double()).to(scores.dtype)
    actual = _filter_topp_sorted_(scores.clone(), torch.tensor([0.95]))

    assert torch.isfinite(expected).sum() == 4
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_filter_minp_bfloat16_boundary():
    from lmdeploy.pytorch.engine.logits_process import _filter_minp_sorted_

    scores = torch.tensor([[1.5546875, 1.21875, -0.0966796875, -0.75,
                            -1.0390625, -1.8828125, -2.53125, -4.84375]],
                          dtype=torch.bfloat16)
    expected = MinPLogitsWarper(0.1)(None, scores.double()).to(scores.dtype)
    actual = _filter_minp_sorted_(scores.clone(), torch.tensor([0.1]))

    assert torch.isfinite(expected).sum() == 3
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sampling_probabilities_match_speculative_target(monkeypatch):
    from lmdeploy.pytorch.engine import logits_process

    logits = torch.tensor([[2.65625, 0.08349609375, 1.984375, -2.328125,
                            1.2578125, -1.6015625, 1.171875, -2.421875]],
                          dtype=torch.bfloat16)
    processor = logits_process.FusedLogitsProcessor(logits_process.SamplingInputs(
        batch_size=1, max_top_k=-1, top_p=torch.tensor([0.95])))
    observed = []

    def sample(probs, seeds, offsets, indices):
        observed.append(torch.zeros_like(probs).scatter(1, indices, probs))
        return indices[:, 0]

    monkeypatch.setattr(logits_process, '_multinomial_sampling', sample)
    processor.sampling(logits)
    # Rejection sampling normalizes filtered target logits in FP32.
    target_probs = processor.filter_logits(logits).softmax(-1, dtype=torch.float32)
    reference = TopPLogitsWarper(0.95)(None, logits.double()).softmax(-1).float()

    assert observed[0].dtype == torch.float32
    torch.testing.assert_close(observed[0], target_probs, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(observed[0], reference, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(observed[0].sum(-1), torch.ones(1), rtol=0, atol=1e-7)


def test_filter_ngram():
    from lmdeploy.pytorch.engine.logits_process import _filter_repetition_ngram_
    vocab_size = 100

    def _get_emtas(n, window_size):
        batch_size = generated_ids.size(0)
        max_n = int(n.max().item())
        same_n = n.eq(max_n).all().item()
        max_window_size = window_size
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

    batch_size, max_n, max_window_size, n = _get_emtas(n, 10)
    scores = torch.rand(batch_size, vocab_size)
    stop_words = torch.randint(0, vocab_size, (batch_size, 3), dtype=torch.int64)
    _filter_repetition_ngram_(scores, stop_words, generated_ids, n, threshold, max_n, max_window_size)

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
    batch_size, max_n, max_window_size, n = _get_emtas(n, 10)

    scores = torch.rand(batch_size, vocab_size)
    stop_words = torch.randint(0, vocab_size, (batch_size, 3), dtype=torch.int64)
    _filter_repetition_ngram_(scores, stop_words, generated_ids, n, threshold, max_n, max_window_size)
    assert not scores[1].isinf().any().item()
    assert scores[0].isinf().sum().item() == vocab_size - 1

    # test ids all 0
    generated_ids = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    n = torch.tensor([3], dtype=torch.int64)
    threshold = torch.tensor([3], dtype=torch.int64)
    batch_size, max_n, max_window_size, n = _get_emtas(n, 10)

    scores = torch.rand(batch_size, vocab_size)
    stop_words = torch.randint(0, vocab_size, (batch_size, 3), dtype=torch.int64)
    _filter_repetition_ngram_(scores, stop_words, generated_ids, n, threshold, max_n, max_window_size)
    assert scores[0].isinf().sum().item() == vocab_size - 1
