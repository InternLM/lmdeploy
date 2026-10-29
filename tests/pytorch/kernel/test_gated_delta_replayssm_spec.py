import pytest
import torch


def _step(batch, tmax, h, hv, kdim, vdim, dtype, seed):
    gen = torch.Generator(device='cuda').manual_seed(seed)
    q = torch.randn(batch, tmax, h, kdim, device='cuda', dtype=dtype, generator=gen)
    k = torch.randn_like(q)
    v = torch.randn(batch, tmax, hv, vdim, device='cuda', dtype=dtype, generator=gen)
    g = (-torch.rand(batch, tmax, hv, device='cuda', generator=gen) * 0.2).to(dtype)
    beta = torch.sigmoid(torch.randn(batch, tmax, hv, device='cuda', dtype=dtype, generator=gen))
    return q, k, v, g.contiguous(), beta.contiguous()


def _reference(q, k, v, g, beta, initial_state, q_seqlens):
    batch, tmax, h, kdim = q.shape
    hv, vdim = v.shape[2:]
    state = initial_state.clone()
    out = torch.zeros_like(v)
    for row in range(batch):
        for token in range(int(q_seqlens[row])):
            for value_head in range(hv):
                key_head = value_head // (hv // h)
                query = q[row, token, key_head].float()
                key = k[row, token, key_head].float()
                query /= torch.sqrt(torch.sum(query * query) + 1e-6)
                query *= kdim**-0.5
                key /= torch.sqrt(torch.sum(key * key) + 1e-6)
                state[row, value_head] *= torch.exp(g[row, token, value_head].float())
                delta = (v[row, token, value_head].float()
                         - torch.sum(state[row, value_head] * key, dim=-1))
                delta *= beta[row, token, value_head].float()
                state[row, value_head] += delta[:, None] * key[None, :]
                out[row, token, value_head] = torch.sum(
                    state[row, value_head] * query[None, :], dim=-1).to(out.dtype)
    return out, state


def _make_state(slots, h, hv, kdim, vdim, buffer_len, tmax, dtype, channelwise=False):
    from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaReplayCursor, GatedDeltaReplayState

    ring_dtype = torch.float32 if channelwise else dtype
    gate_shape = (slots, hv, buffer_len, kdim) if channelwise else (slots, hv, buffer_len)
    checkpoint = torch.zeros(slots, hv, vdim, kdim, device='cuda', dtype=torch.float32)
    return GatedDeltaReplayState(
        conv_state=torch.empty(0, device='cuda'),
        checkpoint=checkpoint,
        d_ring=torch.zeros(slots, hv, buffer_len, vdim, device='cuda', dtype=ring_dtype),
        k_ring=torch.zeros(slots, h, buffer_len, kdim, device='cuda', dtype=ring_dtype),
        g_ring=torch.zeros(gate_shape, device='cuda', dtype=torch.float32),
        cursor=GatedDeltaReplayCursor(
            torch.zeros(slots, 3, device='cuda', dtype=torch.int32), buffer_len, buffer_len, tmax),
        raw_v_ring=torch.zeros(slots, hv, buffer_len, vdim, device='cuda', dtype=dtype),
        raw_k_ring=torch.zeros(slots, h, buffer_len, kdim, device='cuda', dtype=dtype),
        beta_ring=torch.zeros(slots, hv, buffer_len, device='cuda', dtype=torch.float32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('buffer_len,tmax', [
    (8, 2),
    (8, 3),
    (8, 4),
    (16, 4),
    (16, 5),
    (16, 6),
    (16, 8),
    (32, 8),
])
@pytest.mark.parametrize('explicit_workspace', [False, True])
@pytest.mark.parametrize('channelwise', [False, True])
@pytest.mark.parametrize('strided_input', [False, True])
def test_mtp_replayssm_matches_committed_recurrent_state(buffer_len, tmax, explicit_workspace, channelwise,
                                                      strided_input):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm_spec import (
        commit_gdn_replayssm_spec_cursor,
        fold_gdn_replayssm_mtp_history,
        fused_recurrent_gdn_replayssm_spec,
        make_gdn_replayssm_spec_workspace,
    )

    dtype = torch.bfloat16
    batch, h, hv, kdim, vdim, slots = 3, 1, 2, 16, 24, 5
    state_ids = torch.tensor([0, 2, 4], device='cuda', dtype=torch.int64)
    if channelwise:
        h, hv, kdim, vdim = 2, 2, 128, 128
    state = _make_state(slots, h, hv, kdim, vdim, buffer_len, tmax, dtype, channelwise)
    # A batch-sized workspace must handle persistent slot IDs beyond its capacity.
    workspace = make_gdn_replayssm_spec_workspace(state, batch) if explicit_workspace else None
    committed = torch.zeros(batch, hv, vdim, kdim, device='cuda', dtype=torch.float32)

    for step in range(12):
        q, k, v, g, beta = _step(batch, tmax, h, hv, kdim, vdim, dtype, 100 + step)
        if strided_input:
            # Match upstream convolution's [B,D,T] output without packing.
            q, k, v = [x.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2) for x in (q, k, v)]
        if channelwise:
            g = -torch.rand(batch, tmax, hv, kdim, device='cuda') * 0.2
        q_seqlens = torch.tensor([tmax, tmax, 1], device='cuda', dtype=torch.int64)
        rejected = torch.tensor(
            [step % tmax, (step * 2 + 1) % tmax, min(1, tmax - 1)], device='cuda', dtype=torch.int32)

        was_flush = state.cursor.packed_cursor[state_ids, 2].bool().clone()
        expected_out, _ = _reference(q, k, v, g, beta, committed, q_seqlens)
        actual_out, _ = fused_recurrent_gdn_replayssm_spec(
            q, k, v, g, beta, state, state_ids, q_seqlens, workspace=workspace)
        assert actual_out.is_contiguous()
        torch.testing.assert_close(actual_out, expected_out, atol=4e-2, rtol=4e-2)
        if was_flush.any():
            torch.testing.assert_close(
                state.checkpoint[state_ids[was_flush]], committed[was_flush], atol=4e-3, rtol=4e-3)

        accepted = q_seqlens - rejected
        _, next_committed = _reference(q, k, v, g, beta, committed, accepted)
        committed = next_committed
        commit_gdn_replayssm_spec_cursor(
            state.cursor.packed_cursor, state_ids, q_seqlens, rejected, buffer_len, tmax)

        # A rejected suffix must never become readable persistent history.
        for row, slot in enumerate(state_ids.tolist()):
            start = int(state.cursor.packed_cursor[slot, 0])
            if int(rejected[row]) and not int(state.cursor.packed_cursor[slot, 2]):
                state.d_ring[slot, :, start:].fill_(float('nan'))
                state.k_ring[slot, :, start:].fill_(float('nan'))
                state.g_ring[slot, :, start:].fill_(float('nan'))
                state.raw_v_ring[slot, :, start:].fill_(float('nan'))
                state.raw_k_ring[slot, :, start:].fill_(float('nan'))
                state.beta_ring[slot, :, start:].fill_(float('nan'))

    fold_gdn_replayssm_mtp_history(state, state_ids)
    torch.testing.assert_close(state.checkpoint[state_ids], committed, atol=4e-3, rtol=4e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('channelwise,batch,heads,dim,tmax', [
    (False, 3, 3, 96, 3),
    (True, 3, 3, 96, 3),
    (True, 8, 64, 128, 6),
])
def test_mtp_strided_inputs_graph_replay_matches_packed_state(channelwise, batch, heads, dim, tmax):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm_spec import (
        commit_gdn_replayssm_spec_cursor,
        fused_recurrent_gdn_replayssm_spec,
        make_gdn_replayssm_spec_workspace,
    )

    torch.manual_seed(881)
    buffer_len = 16
    slots = batch + 5
    states = [_make_state(slots, heads, heads, dim, dim, buffer_len, tmax, torch.bfloat16, channelwise)
              for _ in range(2)]
    workspaces = [make_gdn_replayssm_spec_workspace(state, batch) for state in states]
    mixed = torch.randn(batch, 3 * heads * dim, tmax, device='cuda', dtype=torch.bfloat16)
    gate_shape = (batch, tmax, heads, dim) if channelwise else (batch, tmax, heads)
    gate = -torch.rand(gate_shape, device='cuda') * 0.2
    beta = torch.rand(batch, tmax, heads, device='cuda')
    ids = torch.arange(batch, device='cuda')
    qlens = torch.full((batch,), tmax, device='cuda', dtype=torch.int32)
    rejected = torch.zeros_like(qlens)
    fields = ('checkpoint', 'd_ring', 'k_ring', 'g_ring', 'raw_v_ring', 'raw_k_ring', 'beta_ring')

    def forward(index):
        data = mixed.transpose(1, 2)
        if index == 0:
            data = data.contiguous()
        q, k, v = [x.reshape(batch, tmax, heads, dim) for x in data.split(heads * dim, dim=-1)]
        return fused_recurrent_gdn_replayssm_spec(
            q, k, v, gate, beta, states[index], ids, qlens, workspace=workspaces[index])[0]

    graphs, outputs = [], []
    for index in range(2):
        forward(index)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward(index)
        graphs.append(graph)
        outputs.append(output)
    for state in states:
        for name in fields:
            getattr(state, name).zero_()
        state.cursor.packed_cursor.zero_()

    for step in range(32):
        mixed.normal_()
        gate.uniform_(-0.4, -0.01)
        beta.uniform_(0, 1)
        ids.copy_((torch.arange(batch, device='cuda') + step) % slots)
        if step % 5 == 0:
            ids[-1] = -1
        qlens.copy_((torch.arange(batch, device='cuda') + step) % (tmax + 1))
        rejected.copy_(torch.minimum(qlens, (torch.arange(batch, device='cuda') + step // 3) % (tmax + 1)))
        for graph in graphs:
            graph.replay()
        assert outputs[1].is_contiguous()
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        for name in fields:
            torch.testing.assert_close(getattr(states[0], name), getattr(states[1], name), atol=0, rtol=0)
        for state in states:
            commit_gdn_replayssm_spec_cursor(state.cursor.packed_cursor, ids, qlens, rejected, buffer_len, tmax)
        torch.testing.assert_close(states[0].cursor.packed_cursor, states[1].cursor.packed_cursor, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('channelwise', [False, True])
def test_mtp_replayssm_memcheck_matrix_invalid_rows_do_not_touch_persistent_state(channelwise):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm_spec import (
        commit_gdn_replayssm_spec_cursor,
        fused_recurrent_gdn_replayssm_spec,
    )

    batch, tmax, h, hv, kdim, vdim, slots, buffer_len = 2, 4, 1, 2, 16, 24, 3, 8
    state = _make_state(slots, h, hv, kdim, vdim, buffer_len, tmax, torch.float16, channelwise)
    tensors = (state.checkpoint, state.d_ring, state.k_ring, state.g_ring, state.raw_v_ring,
               state.raw_k_ring, state.beta_ring, state.cursor.packed_cursor)
    for tensor in tensors:
        tensor.copy_(torch.randn_like(tensor) if tensor.is_floating_point() else torch.zeros_like(tensor))
    before = tuple(tensor.clone() for tensor in tensors)
    q, k, v, g, beta = _step(batch, tmax, h, hv, kdim, vdim, torch.float16, 777)
    state_ids = torch.tensor([-1, 1], device='cuda', dtype=torch.int64)
    q_seqlens = torch.full((batch, ), tmax, device='cuda', dtype=torch.int64)
    rejected = torch.tensor([0, 2], device='cuda', dtype=torch.int32)

    if channelwise:
        g = -torch.rand(*g.shape, kdim, device='cuda') * 0.2
    out, _ = fused_recurrent_gdn_replayssm_spec(q, k, v, g, beta, state, state_ids, q_seqlens)
    commit_gdn_replayssm_spec_cursor(
        state.cursor.packed_cursor, state_ids, q_seqlens, rejected, buffer_len, tmax)

    assert torch.count_nonzero(out[0]) == 0
    for tensor, original in zip(tensors, before):
        assert torch.equal(tensor[0], original[0])
        assert torch.equal(tensor[2], original[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
def test_mtp_exact_fold_single_row_specialization():
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm_spec import (
        commit_gdn_replayssm_spec_cursor,
        fold_gdn_replayssm_mtp_history,
        fused_recurrent_gdn_replayssm_spec,
    )

    batch, tmax, h, hv, kdim, vdim, buffer_len = 1, 4, 1, 2, 16, 24, 8
    state = _make_state(1, h, hv, kdim, vdim, buffer_len, tmax, torch.bfloat16)
    q, k, v, g, beta = _step(batch, tmax, h, hv, kdim, vdim, torch.bfloat16, 991)
    state_ids = torch.zeros(1, device='cuda', dtype=torch.int64)
    q_seqlens = torch.full((1, ), tmax, device='cuda', dtype=torch.int64)
    rejected = torch.ones(1, device='cuda', dtype=torch.int32)

    fused_recurrent_gdn_replayssm_spec(q, k, v, g, beta, state, state_ids, q_seqlens)
    commit_gdn_replayssm_spec_cursor(
        state.cursor.packed_cursor, state_ids, q_seqlens, rejected, buffer_len, tmax)
    _, expected = _reference(q, k, v, g, beta, state.checkpoint.clone(), q_seqlens - rejected)
    fold_gdn_replayssm_mtp_history(state, state_ids)

    torch.testing.assert_close(state.checkpoint, expected, atol=4e-3, rtol=4e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('channelwise', [False, True])
def test_flush_raw_history_is_immutable_until_all_consumers_finish(channelwise):
    from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaReplayCursor, GatedDeltaReplayState
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm_spec import fused_recurrent_gdn_replayssm_spec

    torch.manual_seed(743)
    B, H, HV, K, V, R, T = 128, 16, 32, 128, 128, 8, 4

    def rand(*s, dtype=torch.bfloat16):
        return torch.randn(*s, device='cuda', dtype=dtype) * 0.2

    raw_k = rand(B, H, R, K)
    raw_v = rand(B, HV, R, V)
    g_ring = -torch.rand(B, HV, R, device='cuda') * 0.2
    beta_ring = torch.rand_like(g_ring)
    if channelwise:
        g_ring = -torch.rand(B, HV, R, K, device='cuda') * 0.2
    cursor = torch.zeros(B, 3, device='cuda', dtype=torch.int32)
    cursor[:, 0] = 5
    cursor[:, 2] = 1
    s = GatedDeltaReplayState(
        conv_state=torch.empty(0, device='cuda'),
        checkpoint=rand(B, HV, V, K, dtype=torch.float32),
        d_ring=rand(B, HV, R, V, dtype=torch.float32 if channelwise else torch.bfloat16),
        k_ring=rand(B, H, R, K, dtype=torch.float32 if channelwise else torch.bfloat16),
        g_ring=g_ring,
        cursor=GatedDeltaReplayCursor(cursor, R, R, T),
        raw_v_ring=raw_v,
        raw_k_ring=raw_k,
        beta_ring=beta_ring,
    )
    q = rand(B, T, H, K)
    k = rand(B, T, H, K)
    v = rand(B, T, HV, V)
    g = -torch.rand(B, T, HV, device='cuda') * 0.2
    beta = torch.rand(B, T, HV, device='cuda', dtype=torch.bfloat16)
    if channelwise:
        g = -torch.rand(B, T, HV, K, device='cuda') * 0.2
    ids = torch.arange(B, device='cuda', dtype=torch.int64)
    qlen = torch.full((B,), T, device='cuda', dtype=torch.int64)
    saved = {n: x.clone() for n, x in vars(s).items() if isinstance(x, torch.Tensor)}
    # Oracle consumes the entire committed prefix before any ring overwrite.
    h = s.checkpoint.clone()
    for t in range(5):
        key = raw_k[:, :, t].float().repeat_interleave(HV // H, 1)
        key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        gate = g_ring[:, :, t].exp()
        h = h * (gate[..., None, :] if channelwise else gate[..., None, None])
        d = (raw_v[:, :, t].float() - (h * key[:, :, None, :]).sum(-1)) * beta_ring[:, :, t, None]
        h = h + d[..., None] * key[:, :, None, :]
    expected = h.clone()
    outputs = []
    for t in range(T):
        key = k[:, t].float().repeat_interleave(HV // H, 1)
        query = q[:, t].float().repeat_interleave(HV // H, 1)
        key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        query = query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt() * K**-0.5
        gate = g[:, t].exp()
        h = h * (gate[..., None, :] if channelwise else gate[..., None, None])
        d = (v[:, t].float() - (h * key[:, :, None, :]).sum(-1)) * beta[:, t, :, None].float()
        h = h + d[..., None] * key[:, :, None, :]
        outputs.append((h * query[:, :, None, :]).sum(-1).to(v.dtype))
    expected_out = torch.stack(outputs, 1)

    def reset():
        for n, x in saved.items():
            getattr(s, n).copy_(x)

    def fn():
        return fused_recurrent_gdn_replayssm_spec(q, k, v, g, beta, s, ids, qlen)[0]

    reference = None
    for _ in range(3):
        reset()
        out = fn().clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(s.checkpoint, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(out, expected_out, atol=2e-4, rtol=1e-2)
        if reference is not None:
            torch.testing.assert_close(out, reference, atol=0, rtol=0)
        reference = out
    reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = fn()
    for _ in range(3):
        reset()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(s.checkpoint, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(graph_out, reference, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('gate_scale', [0.2, 0.001])
def test_kda_history_changing_graph_inputs_and_partial_acceptance(gate_scale):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm_spec import (
        commit_gdn_replayssm_spec_cursor,
        fold_gdn_replayssm_mtp_history,
        fused_recurrent_gdn_replayssm_spec,
        make_gdn_replayssm_spec_workspace,
    )

    torch.manual_seed(5219)
    batch, heads, dim, tmax, ring = 3, 2, 128, 6, 16
    state = _make_state(batch, heads, heads, dim, dim, ring, tmax, torch.bfloat16, True)
    state.checkpoint.normal_(std=0.02)
    initial = state.checkpoint.clone()
    truth = initial.clone()
    ids = torch.arange(batch, device='cuda')
    lengths = torch.full((batch,), tmax, device='cuda', dtype=torch.int32)
    rejected = torch.zeros_like(lengths)
    q, k, v, _, beta = _step(batch, tmax, heads, heads, dim, dim, torch.bfloat16, 5219)
    # Match prepare_kda_replay_gates and retain this FP32 buffer across Graph replays.
    beta = torch.rand_like(beta, dtype=torch.float32)
    q, k, v = [x.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2) for x in (q, k, v)]
    gate = -torch.rand(batch, tmax, heads, dim, device='cuda') * gate_scale
    workspace = make_gdn_replayssm_spec_workspace(state, batch)

    def step():
        out, _ = fused_recurrent_gdn_replayssm_spec(
            q, k, v, gate, beta, state, ids, lengths, workspace=workspace)
        commit_gdn_replayssm_spec_cursor(state.cursor.packed_cursor, ids, lengths, rejected, ring, tmax)
        return out

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = step()
    # Capturing must not become part of the independent accepted-input history.
    state.checkpoint.copy_(initial)
    state.cursor.packed_cursor.zero_()
    for name in ('d_ring', 'k_ring', 'g_ring', 'raw_v_ring', 'raw_k_ring', 'beta_ring'):
        getattr(state, name).zero_()
    for index in range(64):
        for tensor in (q, k, v):
            tensor.normal_()
        gate.uniform_(-gate_scale, 0)
        beta.uniform_(0, 1)
        accepted = ((ids + index) % (tmax + 1)).int()
        rejected.copy_(lengths - accepted)
        expected, _ = _reference(q, k, v, gate, beta, truth, lengths)
        _, truth = _reference(q, k, v, gate, beta, truth, accepted)
        graph.replay()
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=5e-3)
    fold_gdn_replayssm_mtp_history(state, ids)
    torch.testing.assert_close(state.checkpoint, truth, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('channelwise', [False, True])
def test_empty_partial_fold_preserves_checkpoint_and_cursor_in_graph(channelwise):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm_spec import fold_gdn_replayssm_mtp_history
    state = _make_state(3, 1, 2, 16, 24, 8, 3, torch.bfloat16, channelwise=channelwise)
    state.checkpoint.normal_()
    # Empty rows must not read poisoned raw history, and invalid IDs must not
    # touch any slot. A non-empty row still folds normally in the same graph.
    state.raw_k_ring.fill_(float('nan'))
    state.raw_v_ring.fill_(float('nan'))
    state.cursor.packed_cursor[0] = torch.tensor([0, 3, 1], device='cuda')
    ids = torch.tensor([0, 1, -1], device='cuda')
    graph = torch.cuda.CUDAGraph()
    fold_gdn_replayssm_mtp_history(state, ids)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        fold_gdn_replayssm_mtp_history(state, ids)
    before, cursor = state.checkpoint.clone(), state.cursor.packed_cursor.clone()
    graph.replay()
    assert torch.equal(state.checkpoint, before)
    assert torch.equal(state.cursor.packed_cursor, cursor)
    state.raw_k_ring[1].zero_()
    state.raw_v_ring[1].zero_()
    state.g_ring[1].fill_(-0.5)
    state.cursor.packed_cursor[1, 0] = 1
    graph.replay()
    torch.testing.assert_close(state.checkpoint[1], before[1] * torch.exp(torch.tensor(-0.5, device='cuda')))
    assert torch.equal(state.checkpoint[[0, 2]], before[[0, 2]])
    assert state.cursor.packed_cursor[1, 0] == 1  # reset belongs to the agent


@pytest.mark.parametrize('batch,heads,dim,tmax,channelwise,evict,head_major', [
    (0, 64, 128, 6, True, False, False),
    (1, 64, 128, 6, True, False, False),
    (2, 64, 128, 6, True, False, False),
    (7, 64, 128, 6, True, False, False),
    (8, 64, 128, 6, True, True, True),
    (9, 64, 128, 6, True, False, True),
    (16, 64, 128, 6, True, True, True),
    (17, 64, 128, 6, True, False, True),
    (16, 32, 128, 6, True, False, True),
    (8, 64, 128, 5, True, False, False),
    (8, 64, 64, 6, True, False, False),
    (8, 64, 128, 6, False, False, False),
])
def test_mtp_launch_policy_and_fold_dispatch(monkeypatch, batch, heads, dim, tmax,
                                            channelwise, evict, head_major):
    from types import SimpleNamespace

    from lmdeploy.pytorch.kernels.cuda import gated_delta_replayssm_spec as kernels

    class Recorder:
        def __getitem__(self, grid):
            self.grid = grid

            def launch(*args, **kwargs):
                self.kwargs = kwargs

            return launch

    def empty(*shape, dtype=torch.float32):
        return torch.empty(shape, device='meta', dtype=dtype)

    ring = empty(batch + 1, heads, 16, dim)
    state = SimpleNamespace(
        checkpoint=empty(batch + 1, heads, dim, dim), d_ring=ring, k_ring=ring,
        raw_v_ring=ring, raw_k_ring=ring,
        g_ring=ring if channelwise else empty(batch + 1, heads, 16),
        beta_ring=empty(batch + 1, heads, 16),
        cursor=SimpleNamespace(R=16, Tmax=tmax, packed_cursor=empty(batch + 1, 3, dtype=torch.int32)))
    q = empty(batch, tmax, heads, dim)
    gates = q if channelwise else empty(batch, tmax, heads)
    ids = empty(batch, dtype=torch.int64)
    qlens = empty(batch, dtype=torch.int32)
    recorder = Recorder()
    monkeypatch.setattr(kernels, '_snapshot_replayssm_raw_kernel', Recorder())
    monkeypatch.setattr(kernels, 'gdn_replayssm_spec_circular_kernel', recorder)
    kernels.fused_recurrent_gdn_replayssm_spec(q, q, q, gates, empty(batch, tmax, heads), state, ids, qlens)
    assert tuple(recorder.kwargs[key] for key in ('BV', 'num_warps', 'num_stages')) == (32, 1, 1)
    assert recorder.kwargs['EVICT_CHECKPOINT'] == evict
    assert recorder.kwargs['HEAD_MAJOR'] == head_major
    assert recorder.grid == ((dim // 32, heads, batch) if head_major else (dim // 32, batch, heads))

    monkeypatch.setattr(kernels, 'gdn_replayssm_exact_fold_kernel', recorder)
    kernels.fold_gdn_replayssm_mtp_history(state, ids)
    expected = (32, 2, 3) if batch <= 1 else (16, 1, 1)
    assert tuple(recorder.kwargs[key] for key in ('BV', 'num_warps', 'num_stages')) == expected
    assert recorder.grid == (dim // expected[0], batch, heads)
    monkeypatch.setattr(kernels, 'commit_gdn_replayssm_spec_cursor_kernel', recorder)
    kernels.commit_gdn_replayssm_spec_cursor(state.cursor.packed_cursor, ids, qlens, qlens, 16, tmax)
    assert recorder.kwargs['BLOCK'] == 256
    assert recorder.grid == ((batch + 255) // 256, )
