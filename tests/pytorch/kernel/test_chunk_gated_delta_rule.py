import pytest
import torch
import torch.nn.functional as F

fla = pytest.importorskip('fla.ops.gated_delta_rule')
fla_chunk_gated_delta_rule = fla.chunk_gated_delta_rule

from lmdeploy.pytorch.kernels.cuda.chunk_gated_delta_rule import (
    _chunk_count_bucket, chunk_gated_delta_rule, prepare_chunk_indices,
    prepare_chunk_offsets)


@pytest.mark.parametrize(
    ('num_chunks', 'expected'),
    [(1, 4), (4, 4), (5, 16), (16, 16), (17, 64), (64, 64),
     (65, 128), (67, 128), (128, 128), (129, 129), (513, 129)],
)
def test_chunk_count_bucket(num_chunks, expected):
    assert _chunk_count_bucket(num_chunks) == expected


def test_chunk_kernels_autotune_by_chunk_count_bucket():
    from lmdeploy.pytorch.kernels.cuda import chunk_gated_delta_rule as kernel

    names = [
        'chunk_local_cumsum_scalar_kernel',
        'chunk_gated_delta_rule_fwd_kkt_solve_kernel',
        'recompute_w_u_fwd_kernel',
        'chunk_gated_delta_rule_fwd_kernel_h_blockdim64',
        'chunk_fwd_kernel_o',
    ]
    for name in names:
        decorated_kernel = getattr(kernel, name)
        while not hasattr(decorated_kernel, 'keys'):
            decorated_kernel = decorated_kernel.fn
        assert 'NT_BUCKET' in decorated_kernel.keys


def _cuda_available():
    return torch.cuda.is_available()


def _make_inputs(length, num_heads=4, num_value_heads=8, key_dim=64, value_dim=64,
                 dtype=torch.bfloat16, initial_state=True):
    q = F.normalize(torch.randn(1, length, num_heads, key_dim, device='cuda', dtype=dtype), dim=-1)
    k = F.normalize(torch.randn(1, length, num_heads, key_dim, device='cuda', dtype=dtype), dim=-1)
    v = torch.randn(1, length, num_value_heads, value_dim, device='cuda', dtype=dtype)
    g = -torch.rand(1, length, num_value_heads, device='cuda', dtype=dtype) * 0.1
    beta = torch.rand(1, length, num_value_heads, device='cuda', dtype=dtype)
    state = None
    if initial_state:
        state = torch.randn(1, num_value_heads, value_dim, key_dim, device='cuda') * 0.05
    return q, k, v, g, beta, state


def _run_fla(q, k, v, g, beta, initial_state, cu_seqlens=None):
    return fla_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=q.shape[-1]**-0.5,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
        transpose_state_layout=True,
    )


@pytest.mark.skipif(not _cuda_available(), reason='CUDA is not available')
@pytest.mark.parametrize('length', [1, 63, 64, 65, 127, 128, 200])
@pytest.mark.parametrize('with_initial_state', [False, True])
def test_chunk_gated_delta_rule_matches_fla(length, with_initial_state):
    torch.manual_seed(length)
    inputs = _make_inputs(length, initial_state=with_initial_state)
    q, k, v, g, beta, initial_state = inputs

    out, final_state, chunk_states = chunk_gated_delta_rule(
        *inputs[:5],
        scale=q.shape[-1]**-0.5,
        initial_state=initial_state,
        output_final_state=True,
    )
    ref_out, ref_final_state = _run_fla(*inputs)

    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(final_state.float(), ref_final_state.float(), atol=2e-2, rtol=2e-2)
    assert chunk_states.shape == (1, (length + 63) // 64, v.shape[2], v.shape[3], q.shape[3])
    expected_initial_state = initial_state if initial_state is not None else torch.zeros_like(chunk_states[:, 0])
    torch.testing.assert_close(
        chunk_states[:, 0].float(),
        expected_initial_state.to(chunk_states.dtype).float(),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(not _cuda_available(), reason='CUDA is not available')
@pytest.mark.parametrize(
    ('length', 'num_heads', 'num_value_heads', 'key_dim', 'value_dim'),
    [
        # original large fused-QKV case: GVA 1:2, K=V=128, straddles a 64-chunk.
        (65, 16, 32, 128, 128),
        # Qwen3-Next TP4 per-card head counts: H=4, HV=8.
        (65, 4, 8, 128, 128),
        # full production prompt length under the default 8192 prefill budget.
        (4288, 4, 8, 128, 128),
        # equal heads (no GVA), small K, entirely within one chunk.
        (63, 4, 4, 64, 64),
        # GVA 1:2 with small K across multiple chunks.
        (129, 8, 16, 64, 64),
        # K=V=256, the kernel's supported head-dim ceiling.
        (200, 4, 8, 256, 256),
        # asymmetric K != V (key_dim < value_dim).
        (65, 4, 8, 64, 128),
    ],
)
def test_chunk_gated_delta_rule_fused_qkv_view_production_shape(
        length, num_heads, num_value_heads, key_dim, value_dim):
    """Qwen passes value as a non-contiguous view of fused QKV storage.

    Covers GVA ratios (1:1 and 1:2), K==V and K!=V, several head dims up to the
    256 ceiling, and lengths that are sub-chunk, chunk-aligned, and multi-chunk.
    """
    torch.manual_seed(1)
    q_width = num_heads * key_dim
    v_width = num_value_heads * value_dim
    mixed_qkv = torch.randn(1, length, 2 * q_width + v_width, device='cuda', dtype=torch.bfloat16)
    q, k, v = torch.split(mixed_qkv, [q_width, q_width, v_width], dim=-1)
    q = F.normalize(q.unflatten(-1, (num_heads, key_dim)), dim=-1)
    k = F.normalize(k.unflatten(-1, (num_heads, key_dim)), dim=-1)
    v = v.unflatten(-1, (num_value_heads, value_dim))
    g = -torch.rand(1, length, num_value_heads, device='cuda', dtype=torch.bfloat16) * 0.1
    beta = torch.rand(1, length, num_value_heads, device='cuda', dtype=torch.bfloat16)
    initial_state = torch.randn(1, num_value_heads, value_dim, key_dim, device='cuda') * 0.02

    assert not v.is_contiguous()
    assert v.stride(1) == mixed_qkv.shape[-1]
    out, final_state, _ = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True)
    contiguous_out, contiguous_final_state, _ = chunk_gated_delta_rule(
        q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(),
        initial_state=initial_state.contiguous(), output_final_state=True)
    ref_out, ref_final_state = _run_fla(q, k, v, g, beta, initial_state)

    torch.testing.assert_close(out, contiguous_out, atol=0, rtol=0)
    torch.testing.assert_close(final_state, contiguous_final_state, atol=0, rtol=0)
    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(final_state.float(), ref_final_state.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not _cuda_available(), reason='CUDA is not available')
def test_chunk_gated_delta_rule_packed_varlen():
    torch.manual_seed(2)
    lengths = [63, 65, 129]
    total_length = sum(lengths)
    q, k, v, g, beta, _ = _make_inputs(total_length)
    cu_seqlens = torch.tensor([0, 63, 128, 257], device='cuda', dtype=torch.int32)
    initial_state = torch.randn(len(lengths), v.shape[2], v.shape[3], q.shape[3], device='cuda') * 0.05

    out, final_state, chunk_states = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ref_out, ref_final_state = _run_fla(q, k, v, g, beta, initial_state, cu_seqlens)

    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(final_state.float(), ref_final_state.float(), atol=2e-2, rtol=2e-2)
    assert chunk_states.shape == (1, 6, v.shape[2], v.shape[3], q.shape[3])
    for sequence, chunk_offset in enumerate([0, 1, 3]):
        torch.testing.assert_close(
            chunk_states[0, chunk_offset].float(),
            initial_state[sequence].to(chunk_states.dtype).float(),
            atol=0,
            rtol=0,
        )


@pytest.mark.skipif(not _cuda_available(), reason='CUDA is not available')
def test_chunk_gated_delta_rule_prefill_decode_handoff():
    pytest.importorskip('tilelang')
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule

    torch.manual_seed(3)
    q, k, v, g, beta, initial_state = _make_inputs(
        65, num_heads=4, num_value_heads=8, key_dim=64, value_dim=64)
    _, local_state, _ = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True)
    _, fla_state = _run_fla(q, k, v, g, beta, initial_state)
    state_indices = torch.tensor([0], device='cuda', dtype=torch.int64)

    for _ in range(3):
        decode_inputs = _make_inputs(
            1, num_heads=4, num_value_heads=8, key_dim=64, value_dim=64, initial_state=False)
        dq, dk, dv, dg, dbeta, _ = decode_inputs
        local_out, local_state = fused_recurrent_gated_delta_rule(
            dq, dk, dv, g=dg, beta=dbeta, initial_state=local_state,
            output_final_state=True, state_indices=state_indices, transpose_state_layout=True)
        fla_out, fla_state = fused_recurrent_gated_delta_rule(
            dq, dk, dv, g=dg, beta=dbeta, initial_state=fla_state,
            output_final_state=True, state_indices=state_indices, transpose_state_layout=True)
        torch.testing.assert_close(local_out.float(), fla_out.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(local_state.float(), fla_state.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not _cuda_available(), reason='CUDA is not available')
def test_chunk_gated_delta_rule_reuses_precomputed_metadata():
    torch.manual_seed(5)
    lengths = [63, 64, 65]
    total_length = sum(lengths)
    inputs = _make_inputs(total_length)
    q, k, v, g, beta, initial_state = inputs
    cu_seqlens = torch.tensor([0, 63, 127, 192], device='cuda', dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, 64)

    expected = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=initial_state.expand(len(lengths), -1, -1, -1).contiguous(),
        output_final_state=True, cu_seqlens=cu_seqlens)
    actual = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=initial_state.expand(len(lengths), -1, -1, -1).contiguous(),
        output_final_state=True, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, chunk_offsets=chunk_offsets)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)


@pytest.mark.skipif(not _cuda_available(), reason='CUDA is not available')
def test_chunk_gated_delta_rule_tp4_service_shape_with_metadata():
    torch.manual_seed(6)
    length, num_heads, num_value_heads = 4288, 4, 8
    key_dim = value_dim = 128
    q_width = num_heads * key_dim
    v_width = num_value_heads * value_dim
    mixed_qkv = torch.randn(1, length, 2 * q_width + v_width, device='cuda', dtype=torch.bfloat16)
    q, k, v = torch.split(mixed_qkv, [q_width, q_width, v_width], dim=-1)
    q = F.normalize(q.unflatten(-1, (num_heads, key_dim)), dim=-1)
    k = F.normalize(k.unflatten(-1, (num_heads, key_dim)), dim=-1)
    v = v.unflatten(-1, (num_value_heads, value_dim))
    g = -torch.rand(1, length, num_value_heads, device='cuda', dtype=torch.bfloat16) * 0.1
    beta = torch.rand(1, length, num_value_heads, device='cuda', dtype=torch.bfloat16)
    initial_state = torch.randn(1, num_value_heads, value_dim, key_dim, device='cuda') * 0.02
    cu_seqlens = torch.tensor([0, length], device='cuda', dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, 64)

    out, final_state, chunk_states = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets)
    ref_out, ref_final_state = _run_fla(q, k, v, g, beta, initial_state, cu_seqlens)

    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(final_state.float(), ref_final_state.float(), atol=2e-2, rtol=2e-2)
    assert chunk_states.shape == (1, 67, num_value_heads, value_dim, key_dim)


@pytest.mark.skipif(not _cuda_available(), reason='CUDA is not available')
def test_update_chunked_gated_delta_rule_meta_prepares_once(monkeypatch):
    from types import SimpleNamespace

    from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
    from lmdeploy.pytorch.kernels.cuda import chunk_gated_delta_rule as kernel

    calls = {'indices': 0, 'offsets': 0}
    real_indices = kernel.prepare_chunk_indices
    real_offsets = kernel.prepare_chunk_offsets

    def counted_indices(*args, **kwargs):
        calls['indices'] += 1
        return real_indices(*args, **kwargs)

    def counted_offsets(*args, **kwargs):
        calls['offsets'] += 1
        return real_offsets(*args, **kwargs)

    monkeypatch.setattr(kernel, 'prepare_chunk_indices', counted_indices)
    monkeypatch.setattr(kernel, 'prepare_chunk_offsets', counted_offsets)
    metadata = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 65], device='cuda', dtype=torch.int32))

    CudaOpsBackend.update_chunked_gated_delta_rule_meta(metadata, None)

    assert calls == {'indices': 1, 'offsets': 1}
    assert metadata.gated_delta_chunk_indices.shape == (2, 2)
    assert metadata.gated_delta_chunk_offsets.shape == (2, )


@pytest.mark.skipif(not _cuda_available(), reason='CUDA is not available')
def test_chunk_gated_delta_rule_rejects_invalid_varlen_inputs():
    q, k, v, g, beta, _ = _make_inputs(64)
    state = torch.zeros(2, v.shape[2], v.shape[3], q.shape[3], device='cuda')

    with pytest.raises(ValueError, match='physical batch size'):
        chunk_gated_delta_rule(
            q.expand(2, -1, -1, -1), k.expand(2, -1, -1, -1), v.expand(2, -1, -1, -1),
            g.expand(2, -1, -1), beta.expand(2, -1, -1), initial_state=state,
            cu_seqlens=torch.tensor([0, 64], device='cuda', dtype=torch.int32))

    with pytest.raises(ValueError, match='initial_state must have shape'):
        chunk_gated_delta_rule(
            q, k, v, g, beta, initial_state=state[:1],
            cu_seqlens=torch.tensor([0, 32, 64], device='cuda', dtype=torch.int32))


@pytest.mark.skipif(not _cuda_available(), reason='CUDA is not available')
def test_cuda_backend_updates_only_selected_state_rows():
    pytest.importorskip('tilelang')
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaGatedDeltaRuleImpl

    torch.manual_seed(4)
    lengths = [63, 65]
    total_length = sum(lengths)
    q, k, v, g, beta, _ = _make_inputs(total_length)
    cu_seqlens = torch.tensor([0, 63, 128], device='cuda', dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device='cuda', dtype=torch.int64)
    state_bank = torch.randn(5, v.shape[2], v.shape[3], q.shape[3], device='cuda', dtype=torch.bfloat16)
    state_before = state_bank.clone()
    selected_state = state_bank.index_select(0, state_indices).clone()

    expected_out, expected_state, expected_chunks = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=selected_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    impl = CudaGatedDeltaRuleImpl()
    out, returned_bank, chunk_states = impl.chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=state_bank,
        state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        transpose_state_layout=True,
    )

    assert returned_bank.data_ptr() == state_bank.data_ptr()
    torch.testing.assert_close(out, expected_out, atol=0, rtol=0)
    torch.testing.assert_close(chunk_states, expected_chunks, atol=0, rtol=0)
    torch.testing.assert_close(state_bank.index_select(0, state_indices), expected_state.to(state_bank.dtype), atol=0, rtol=0)
    untouched = torch.tensor([0, 2, 4], device='cuda', dtype=torch.int64)
    torch.testing.assert_close(
        state_bank.index_select(0, untouched),
        state_before.index_select(0, untouched),
        atol=0,
        rtol=0,
    )
