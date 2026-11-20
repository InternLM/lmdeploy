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


def test_sampling_delta_keeps_generated_ids_current():
    """The scheduler builds the next decode step's sampling inputs before the
    current step's tokens reach the sequences, so the generated ids it sends
    lag one token behind; the device-side delta must carry the latest one."""
    from types import SimpleNamespace

    from lmdeploy.pytorch.engine.logits_process import SamplingInputs
    from lmdeploy.pytorch.strategies.ar.step_inputs import (
        merge_sampling_delta,
        reindex_sampling_delta,
        step_sampling_delta,
    )

    pad = 0
    # step N: two sequences that have generated [5, 6] and [7]
    inputs = SamplingInputs(
        num_ignore_eos=torch.tensor([0, 0]),
        random_offsets=torch.tensor([0, 0]),
        repetition_penalty=torch.tensor([2.0, 2.0]),
        generated_ids=torch.tensor([[5, 6], [pad, 7]]),
        batch_size=2,
    )
    delta = step_sampling_delta(inputs.get_delta(), torch.tensor([8, 9]))

    # step N + 1: scheduler-side generated ids do not have tokens 8 and 9 yet
    stale = SamplingInputs(
        num_ignore_eos=torch.tensor([0, 0]),
        random_offsets=torch.tensor([0, 0]),
        repetition_penalty=torch.tensor([2.0, 2.0]),
        generated_ids=torch.tensor([[5, 6], [pad, 7]]),
        batch_size=2,
    )
    stale.update_delta(delta)
    assert stale.generated_ids.tolist() == [[5, 6, 8], [pad, 7, 9]]

    # the last generated token is penalized
    scores = torch.ones(2, 10)
    from lmdeploy.pytorch.engine.logits_process import _process_repetition_penalty_
    scores = _process_repetition_penalty_(scores, stale.generated_ids, stale.repetition_penalty)
    assert scores[0, 8] == 0.5 and scores[1, 9] == 0.5

    # merging keeps ids right-aligned; a batch without generated ids is padded
    from lmdeploy.pytorch.engine.logits_process import SamplingInputsDelta
    other = SamplingInputsDelta(num_ignore_eos=torch.tensor([0]),
                                random_offsets=torch.tensor([0]),
                                generated_ids=torch.tensor([[3]]))
    merged = merge_sampling_delta(delta, other, pad)
    assert merged.generated_ids.tolist() == [[5, 6, 8], [pad, 7, 9], [pad, pad, 3]]
    no_ids = SamplingInputsDelta(num_ignore_eos=torch.tensor([0]), random_offsets=torch.tensor([0]))
    merged = merge_sampling_delta(no_ids, other, pad)
    assert merged.generated_ids.tolist() == [[pad], [3]]

    reindexed = reindex_sampling_delta(delta, SimpleNamespace(indices=torch.tensor([1])))
    assert reindexed.generated_ids.tolist() == [[pad, 7, 9]]
