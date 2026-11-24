# ruff: noqa: E402
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the ring-buffer write of ``causal_conv1d_fn``.

Covers prefill chunks shorter than the conv width (``q < width``): fresh
requests, continuations with ring wraparound, mixed-length batches
(cross-request isolation) and chunked-vs-one-shot prefill equivalence.
The prefill helper is pure PyTorch, so these tests run on CPU.
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip('triton')

from lmdeploy.pytorch.kernels.ascend.triton_ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_fn,
    causal_conv1d_ref,
)

WIDTH = 4
STATE_LEN = 6  # width + num_spec_tokens
DIM = 2

torch.manual_seed(0)
WEIGHT = torch.randn(DIM, WIDTH)


def _call(cache, chunks, hist_lens, slots):
    """Run one varlen prefill batch through the real function."""
    tokens = [t for c in chunks for t in c]
    q = torch.tensor([len(c) for c in chunks], dtype=torch.int32)
    h = torch.tensor(hist_lens, dtype=torch.int32)
    x = torch.tensor(tokens, dtype=torch.float32).repeat(DIM, 1)
    out = causal_conv1d_fn(
        x,
        WEIGHT,
        conv_states=cache,
        has_initial_state=h > 0,
        cache_indices=torch.tensor(slots, dtype=torch.int32),
        query_start_loc=F.pad(q.cumsum(0, dtype=torch.int32), (1, 0)),
        read_conv_offsets=(h[:, None] + torch.arange(-(WIDTH - 1), 0)) % STATE_LEN,
        write_conv_offsets=((h + q)[:, None] + torch.arange(-WIDTH, 0)) % STATE_LEN,
    )
    return out


def _preload(cache, slot, stream, upto):
    """Seat the trailing window of ``stream[:upto]`` as a prior state."""
    for j in range(WIDTH):
        idx = upto - WIDTH + j
        if idx >= 0:
            cache[slot, :, idx % STATE_LEN] = float(stream[idx])


def _assert_window(cache, slot, stream, total):
    """Write window must hold tokens #(total-4..total-1) at their slots."""
    for j in range(WIDTH):
        idx = total - WIDTH + j
        want = float(stream[idx]) if idx >= 0 else 0.0
        got = cache[slot, 0, idx % STATE_LEN].item()
        assert got == pytest.approx(want), f"slot {slot} pos {idx % STATE_LEN}: got {got}, want {want}"


def _assert_next_token(cache, slot, stream, total):
    """Next decode step must see the true trailing (width-1) history."""
    read = torch.tensor([(total - WIDTH + 1 + k) % STATE_LEN for k in range(WIDTH - 1)])
    hist_ring = cache[slot].index_select(-1, read)
    hist_true = torch.tensor(
        [[float(stream[k]) if k >= 0 else 0.0 for k in range(total - WIDTH + 1, total)] for _ in range(DIM)]
    )
    nxt = torch.full((DIM, 1), 2.0)
    got, _ = causal_conv1d_ref(nxt, WEIGHT, initial_states=hist_ring)
    exp, _ = causal_conv1d_ref(nxt, WEIGHT, initial_states=hist_true)
    torch.testing.assert_close(got, exp, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('q', [1, 2, 3, 4, 5, 8])
def test_fresh_request(q):
    stream = [10.0 * k for k in range(1, q + 1)]
    cache = torch.zeros(4, DIM, STATE_LEN)
    out = _call(cache, [stream], [0], [1])
    assert out.shape == (DIM, q)
    _assert_window(cache, 1, stream, q)
    _assert_next_token(cache, 1, stream, q)


@pytest.mark.parametrize('hist,q', [(4, 1), (5, 2), (3, 3), (6, 4), (7, 1), (2, 2), (1, 1)])
def test_continuation(hist, q):
    stream = [float(i + 1) for i in range(hist + q)]
    cache = torch.zeros(4, DIM, STATE_LEN)
    _preload(cache, 1, stream, hist)
    _call(cache, [stream[hist:]], [hist], [1])
    _assert_window(cache, 1, stream, hist + q)
    _assert_next_token(cache, 1, stream, hist + q)


def test_mixed_batch_cross_request_isolation():
    """A short request must not inherit the previous request's tokens."""
    long_stream = [float(i + 1) for i in range(9)]
    short_stream = [float(i + 100) for i in range(1)]  # q=1, after the long one
    cache = torch.zeros(4, DIM, STATE_LEN)
    _call(cache, [long_stream, short_stream], [0, 0], [1, 2])
    _assert_window(cache, 1, long_stream, 9)
    _assert_window(cache, 2, short_stream, 1)
    _assert_next_token(cache, 2, short_stream, 1)


def test_chunked_prefill_matches_one_shot():
    stream = [float(i + 1) for i in range(13)]
    chunks = [stream[:5], stream[5:6], stream[6:9], stream[9:10], stream[10:]]

    one_shot = torch.zeros(4, DIM, STATE_LEN)
    _call(one_shot, [stream], [0], [1])

    chunked = torch.zeros(4, DIM, STATE_LEN)
    hist = 0
    for c in chunks:
        _call(chunked, [c], [hist], [1])
        hist += len(c)

    _assert_window(chunked, 1, stream, 13)
    _assert_next_token(chunked, 1, stream, 13)
    for j in range(WIDTH):
        idx = 13 - WIDTH + j
        assert chunked[1, 0, idx % STATE_LEN].item() == one_shot[1, 0, idx % STATE_LEN].item()


def test_prefill_output_matches_reference():
    """The compute path must stay correct independently of the ring write."""
    stream = [float(i + 1) for i in range(7)]
    cache = torch.zeros(4, DIM, STATE_LEN)
    got = _call(cache, [stream[:2], stream[2:]], [0, 2], [1, 1])
    x = torch.tensor(stream, dtype=torch.float32).repeat(DIM, 1)
    exp, _ = causal_conv1d_ref(x, WEIGHT)
    torch.testing.assert_close(got, exp, atol=1e-5, rtol=1e-5)
