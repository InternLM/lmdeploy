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
    from lmdeploy.pytorch.nn.gated_delta import GatedDeltaReplayCursor, GatedDeltaReplayState

    ring_dtype = torch.float32 if channelwise else dtype
    gate_shape = (slots, hv, buffer_len, kdim) if channelwise else (slots, hv, buffer_len)
    checkpoint = torch.zeros(slots, hv, vdim, kdim, device='cuda', dtype=torch.float32)
    return GatedDeltaReplayState(
        mode='mtp',
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
    # Production retains one slot-capacity workspace across smaller graph buckets.
    workspace = make_gdn_replayssm_spec_workspace(state, slots) if explicit_workspace else None
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
@pytest.mark.parametrize('channelwise', [False, True])
def test_mtp_strided_inputs_graph_replay_matches_packed_state(channelwise):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm_spec import (
        commit_gdn_replayssm_spec_cursor,
        fused_recurrent_gdn_replayssm_spec,
        make_gdn_replayssm_spec_workspace,
    )

    torch.manual_seed(881)
    batch, heads, dim, tmax, buffer_len = 3, 3, 96, 3, 16
    states = [_make_state(batch, heads, heads, dim, dim, buffer_len, tmax, torch.bfloat16, channelwise)
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
        ids.copy_((torch.arange(batch, device='cuda') + step) % batch)
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
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm_spec import fused_recurrent_gdn_replayssm_spec
    from lmdeploy.pytorch.nn.gated_delta import GatedDeltaReplayCursor, GatedDeltaReplayState

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
        mode='mtp',
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
