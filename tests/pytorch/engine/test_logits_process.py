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


def _rolling_requests(num_requests: int, lengths: tuple[int, ...]):
    """Deterministic token streams: request ``rid`` emits ``1 + rid * 100 +
    pos``; 0 is the pad id and never appears in a stream."""
    return [[1 + rid * 100 + pos for pos in range(lengths[rid % len(lengths)])] for rid in range(num_requests)]


def test_sampling_delta_generated_ids_bounded_under_rolling_batches():
    """Requests are admitted and finish continuously (max batch 2, alternating
    short/long outputs).

    The scheduler rebuilds its generated history every step, one token behind the device. The persisted history must
    always hold the latest token of every live row, and its width must stay bounded by the scheduler width + 1 instead
    of growing with every decode step.
    """
    from types import SimpleNamespace

    from lmdeploy.pytorch.engine.logits_process import (
        SamplingInputs,
        _process_repetition_penalty_,
    )
    from lmdeploy.pytorch.strategies.ar.step_inputs import (
        merge_sampling_delta,
        reindex_sampling_delta,
        step_sampling_delta,
    )

    pad, vocab, max_batch, penalty = 0, 2000, 2, 2.0
    streams = _rolling_requests(12, (5, 9))
    max_len = max(len(s) for s in streams)
    pending = list(range(len(streams)))
    live: list[int] = []  # request ids, in batch order
    emitted: dict[int, list[int]] = {}  # tokens produced so far per live request
    delta = None  # persisted device-side sampling delta (StepInputs.sampling_delta)

    def build_inputs(history: list[list[int]]) -> SamplingInputs:
        width = max((len(h) for h in history), default=0)
        ids = torch.full((len(history), width), pad, dtype=torch.int64)
        for i, h in enumerate(history):
            if h:
                ids[i, -len(h):] = torch.tensor(h)
        n = len(history)
        return SamplingInputs(
            num_ignore_eos=torch.zeros(n, dtype=torch.int64),
            random_offsets=torch.zeros(n, dtype=torch.int64),
            repetition_penalty=torch.full((n, ), penalty),
            generated_ids=ids,
            batch_size=n,
        )

    max_width_seen = 0
    num_steps = 0
    while pending or live:
        # finished rows leave the batch (reindex)
        keep = [i for i, rid in enumerate(live) if len(emitted[rid]) < len(streams[rid])]
        if len(keep) != len(live):
            if delta is not None:
                if keep:
                    delta = reindex_sampling_delta(delta, SimpleNamespace(indices=torch.tensor(keep)))
                else:
                    delta = None
            live = [live[i] for i in keep]

        # prefill of newly admitted requests, merged into the decode state
        new = []
        while pending and len(live) + len(new) < max_batch:
            new.append(pending.pop(0))
        if new:
            prefill = build_inputs([[] for _ in new]).get_delta()
            first = torch.tensor([streams[rid][0] for rid in new])
            prefill = step_sampling_delta(prefill, first)
            delta = prefill if delta is None else merge_sampling_delta(delta, prefill, pad)
            for rid in new:
                emitted[rid] = [streams[rid][0]]
            live += new
        if not live:
            continue

        # decode step: the scheduler history is one token behind the device
        sched = build_inputs([emitted[rid][:-1] for rid in live])
        sched_width = sched.generated_ids.size(1)
        sched.update_delta(delta)
        ids = sched.generated_ids

        # latest token of every row is present and is the penalized one
        assert ids.size(0) == len(live)
        assert ids[:, -1].tolist() == [emitted[rid][-1] for rid in live]
        # full history is intact, right-aligned
        for i, rid in enumerate(live):
            h = emitted[rid]
            assert ids[i, -len(h):].tolist() == h
            assert (ids[i, :ids.size(1) - len(h)] == pad).all()
        scores = _process_repetition_penalty_(torch.ones(len(live), vocab), ids, sched.repetition_penalty)
        for i, rid in enumerate(live):
            assert scores[i, emitted[rid][-1]] == 1 / penalty

        # width is bounded by what live requests need, not by past batches
        longest = max(len(emitted[rid]) for rid in live)
        assert ids.size(1) <= sched_width + 1
        assert ids.size(1) <= longest
        max_width_seen = max(max_width_seen, ids.size(1))

        # advance one decode step
        nxt = [streams[rid][len(emitted[rid])] for rid in live]
        delta = step_sampling_delta(sched.get_delta(), torch.tensor(nxt))
        for rid, token in zip(live, nxt):
            emitted[rid].append(token)
        # appended storage is compact, not a view into a larger allocation
        gen = delta.generated_ids
        assert gen.untyped_storage().nbytes() == gen.numel() * gen.element_size()
        num_steps += 1

    assert num_steps > 3 * max_len
    assert max_width_seen <= max_len


def test_update_delta_generated_ids_reconciliation():
    from lmdeploy.pytorch.engine.logits_process import SamplingInputs, SamplingInputsDelta

    def make(ids):
        return SamplingInputs(num_ignore_eos=torch.tensor([0, 0]),
                              random_offsets=torch.tensor([0, 0]),
                              generated_ids=ids,
                              batch_size=2)

    def make_delta(ids):
        return SamplingInputsDelta(num_ignore_eos=torch.tensor([0, 0]),
                                   random_offsets=torch.tensor([0, 0]),
                                   generated_ids=ids)

    wide = torch.arange(1, 21).reshape(2, 10)

    # empty scheduler history still keeps the latest token (and nothing more)
    inputs = make(torch.empty(2, 0, dtype=torch.int64))
    inputs.update_delta(make_delta(wide))
    assert inputs.generated_ids.tolist() == [[10], [20]]

    # at most scheduler width + 1 columns, right-aligned
    inputs = make(torch.zeros(2, 3, dtype=torch.int64))
    inputs.update_delta(make_delta(wide))
    assert inputs.generated_ids.tolist() == [[7, 8, 9, 10], [17, 18, 19, 20]]

    # a narrower delta is kept as is
    inputs = make(torch.zeros(2, 20, dtype=torch.int64))
    inputs.update_delta(make_delta(wide))
    assert inputs.generated_ids.tolist() == wide.tolist()

    # no request needs the history: drop it
    inputs = make(None)
    inputs.update_delta(make_delta(wide))
    assert inputs.generated_ids is None

    # ARSpec / DLLM opt out: scheduler history is untouched
    sched = torch.tensor([[1, 2], [3, 4]])
    inputs = make(sched)
    inputs.update_delta(make_delta(None))
    assert inputs.generated_ids is sched

    # other delta fields are still applied
    inputs = make(sched)
    inputs.update_delta(SamplingInputsDelta(num_ignore_eos=torch.tensor([5, 6])))
    assert inputs.num_ignore_eos.tolist() == [5, 6]
    assert inputs.generated_ids is sched
