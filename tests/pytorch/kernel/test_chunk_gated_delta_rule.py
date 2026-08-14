import pytest
import torch
import torch.nn.functional as F
import triton

from lmdeploy.pytorch.kernels.cuda.chunk_gated_delta_rule import (
    _chunk_count_bucket,
    chunk_conv_states,
    chunk_gated_delta_rule,
    prepare_chunk_indices,
    prepare_chunk_offsets,
)

fla = pytest.importorskip('fla.ops.gated_delta_rule')
fla_chunk_gated_delta_rule = fla.chunk_gated_delta_rule


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

    Covers GVA ratios (1:1 and 1:2), K==V and K!=V, several head dims up to the 256 ceiling, and lengths that are sub-
    chunk, chunk-aligned, and multi-chunk.
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


def test_cuda_backend_dispatches_to_fla_without_chunk_metadata():
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaGatedDeltaRuleImpl

    impl = CudaGatedDeltaRuleImpl.__new__(CudaGatedDeltaRuleImpl)
    calls = []
    expected_out = torch.randn(1, 2, 1, 2)
    expected_state = torch.randn(1, 1, 2, 2)

    def fla_chunk_func(*args, **kwargs):
        calls.append(('fla', kwargs))
        return expected_out, expected_state

    def chunk_func_with_states(*args, **kwargs):
        raise AssertionError('chunk-state implementation must not be called')

    impl.fla_chunk_func = fla_chunk_func
    impl.chunk_func_with_states = chunk_func_with_states
    state_bank = torch.zeros(2, 1, 2, 2)
    state_indices = torch.tensor([1])
    q = k = v = torch.empty(1, 2, 1, 2)
    g = beta = torch.empty(1, 2, 1)

    out, returned_bank, chunk_states = impl.chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=state_bank,
        state_indices=state_indices,
        output_final_state=True,
        transpose_state_layout=True,
    )

    assert out is expected_out
    assert returned_bank.data_ptr() == state_bank.data_ptr()
    assert chunk_states is None
    assert calls[0][0] == 'fla'
    assert 'chunk_indices' not in calls[0][1]
    assert 'chunk_offsets' not in calls[0][1]
    assert calls[0][1]['transpose_state_layout'] is True
    torch.testing.assert_close(state_bank[1], expected_state[0])


def test_cuda_backend_falls_back_to_chunk_state_implementation_without_fla():
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaGatedDeltaRuleImpl

    impl = CudaGatedDeltaRuleImpl.__new__(CudaGatedDeltaRuleImpl)
    calls = []
    expected_out = torch.randn(1, 2, 1, 2)
    expected_state = torch.randn(1, 1, 2, 2)
    expected_chunks = torch.randn(1, 1, 1, 2, 2)

    def chunk_func_with_states(*args, **kwargs):
        calls.append(kwargs)
        return expected_out, expected_state, expected_chunks

    impl.fla_chunk_func = None
    impl.chunk_func_with_states = chunk_func_with_states
    state_bank = torch.zeros(2, 1, 2, 2)
    state_indices = torch.tensor([1])
    q = k = v = torch.empty(1, 2, 1, 2)
    g = beta = torch.empty(1, 2, 1)

    out, returned_bank, chunk_states = impl.chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=state_bank,
        state_indices=state_indices,
        output_final_state=True,
    )

    assert out is expected_out
    assert returned_bank.data_ptr() == state_bank.data_ptr()
    assert chunk_states is expected_chunks
    assert calls[0]['chunk_indices'] is None
    assert calls[0]['chunk_offsets'] is None
    torch.testing.assert_close(state_bank[1], expected_state[0])


def test_cuda_backend_dispatches_to_chunk_state_implementation():
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaGatedDeltaRuleImpl

    impl = CudaGatedDeltaRuleImpl.__new__(CudaGatedDeltaRuleImpl)
    calls = []
    expected_out = torch.randn(1, 2, 1, 2)
    expected_state = torch.randn(1, 1, 2, 2)
    expected_chunks = torch.randn(1, 1, 1, 2, 2)

    def fla_chunk_func(*args, **kwargs):
        raise AssertionError('FLA implementation must not be called')

    def chunk_func_with_states(*args, **kwargs):
        calls.append(kwargs)
        return expected_out, expected_state, expected_chunks

    impl.fla_chunk_func = fla_chunk_func
    impl.chunk_func_with_states = chunk_func_with_states
    state_bank = torch.zeros(2, 1, 2, 2)
    state_indices = torch.tensor([1])
    chunk_indices = torch.tensor([[0, 0]], dtype=torch.int32)
    chunk_offsets = torch.tensor([0, 1], dtype=torch.int32)
    q = k = v = torch.empty(1, 2, 1, 2)
    g = beta = torch.empty(1, 2, 1)

    out, returned_bank, chunk_states = impl.chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=state_bank,
        state_indices=state_indices,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        output_final_state=True,
    )

    assert out is expected_out
    assert returned_bank.data_ptr() == state_bank.data_ptr()
    assert chunk_states is expected_chunks
    assert calls[0]['chunk_indices'] is chunk_indices
    assert calls[0]['chunk_offsets'] is chunk_offsets
    torch.testing.assert_close(state_bank[1], expected_state[0])


@pytest.mark.skipif(not _cuda_available(), reason='CUDA is not available')
def test_cuda_backend_updates_only_selected_state_rows():
    pytest.importorskip('tilelang')
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaGatedDeltaRuleImpl

    torch.manual_seed(4)
    lengths = [63, 65]
    total_length = sum(lengths)
    q, k, v, g, beta, _ = _make_inputs(total_length)
    cu_seqlens = torch.tensor([0, 63, 128], device='cuda', dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, 64)
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
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        output_final_state=True,
        transpose_state_layout=True,
    )

    assert returned_bank.data_ptr() == state_bank.data_ptr()
    torch.testing.assert_close(out, expected_out, atol=0, rtol=0)
    torch.testing.assert_close(chunk_states, expected_chunks, atol=0, rtol=0)
    updated = state_bank.index_select(0, state_indices)
    torch.testing.assert_close(updated, expected_state.to(state_bank.dtype), atol=0, rtol=0)
    untouched = torch.tensor([0, 2, 4], device='cuda', dtype=torch.int64)
    torch.testing.assert_close(
        state_bank.index_select(0, untouched),
        state_before.index_select(0, untouched),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(not _cuda_available(), reason='CUDA is not available')
def test_cuda_backend_fla_and_local_branches_match():
    """The FLA and local chunk branches must compute the same result.

    Both branches run the same chunked gated-delta math; the only intended
    difference is that the local branch (explicit ``chunk_indices``) additionally
    returns per-chunk-boundary states. This guards the FLA fallback path end to
    end on real CUDA rather than only its routing.
    """
    pytest.importorskip('tilelang')
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaGatedDeltaRuleImpl, has_fla

    if not has_fla():
        pytest.skip('FLA chunk implementation is not available')

    torch.manual_seed(9)
    lengths = [63, 65]
    total_length = sum(lengths)
    q, k, v, g, beta, _ = _make_inputs(total_length)
    cu_seqlens = torch.tensor([0, 63, 128], device='cuda', dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, 64)
    state_indices = torch.tensor([3, 1], device='cuda', dtype=torch.int64)
    state_bank = torch.randn(5, v.shape[2], v.shape[3], q.shape[3], device='cuda', dtype=torch.bfloat16)

    impl = CudaGatedDeltaRuleImpl()

    # FLA branch: no precomputed chunk metadata.
    fla_bank = state_bank.clone()
    fla_out, fla_returned, fla_chunk_states = impl.chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=fla_bank,
        state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        transpose_state_layout=True,
    )

    # Local branch: explicit chunk metadata.
    local_bank = state_bank.clone()
    local_out, local_returned, local_chunk_states = impl.chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=local_bank,
        state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        output_final_state=True,
        transpose_state_layout=True,
    )

    # Only the local branch exposes per-chunk-boundary states.
    assert fla_chunk_states is None
    assert local_chunk_states is not None
    assert local_chunk_states.shape == (1, len(chunk_indices), v.shape[2], v.shape[3], q.shape[3])

    # Both branches must produce the same attention output and terminal states.
    torch.testing.assert_close(fla_out.float(), local_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(fla_returned.float(), local_returned.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not _cuda_available(), reason='CUDA is not available')
@pytest.mark.parametrize('boundary_chunk', [0, 1, 2])
def test_chunk_states_restore_and_run_suffix_reaches_final_state(boundary_chunk):
    """Per-chunk-boundary states support Marconi-style prefix restoration.

    ``chunk_states[:, c]`` is the recurrent state at the *start* of chunk ``c``
    (after consuming the first ``c * 64`` tokens). Restoring from that state and
    running the remaining suffix must reproduce both the attention output over the
    suffix and the terminal ``final_state`` of the full run. This is the invariant
    that lets a prefix-cache checkpoint stored at a chunk boundary be reused as
    the starting state for any longer prompt sharing that prefix.

    Because ``chunk_states[:, -1]`` is the state at the *start* of the last chunk
    rather than after it, it is generally **not** equal to ``final_state``; the
    constructive suffix check below is the stronger guarantee.
    """
    torch.manual_seed(10)
    length = 200
    q, k, v, g, beta, initial_state = _make_inputs(length)
    scale = q.shape[-1]**-0.5

    out_full, final_state, chunk_states = chunk_gated_delta_rule(
        q, k, v, g, beta, scale=scale, initial_state=initial_state, output_final_state=True)

    num_chunks = chunk_states.shape[1]
    # The last boundary's start-state covers the prefix up to (num_chunks-1)*64,
    # not the terminal state after the full sequence, so it must differ from it.
    assert not torch.allclose(chunk_states[:, -1].float(), final_state.float())

    c = min(boundary_chunk, num_chunks - 1)
    boundary = c * 64
    suffix_len = length - boundary
    if suffix_len <= 0:
        pytest.skip('suffix is empty for this boundary')

    out_suffix, final_state_suffix, _ = chunk_gated_delta_rule(
        q[:, boundary:],
        k[:, boundary:],
        v[:, boundary:],
        g[:, boundary:],
        beta[:, boundary:],
        scale=scale,
        initial_state=chunk_states[:, c],
        output_final_state=True,
    )

    torch.testing.assert_close(final_state_suffix.float(), final_state.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out_suffix.float(), out_full[:, boundary:].float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not _cuda_available(), reason='CUDA is not available')
@pytest.mark.parametrize('conv_kernel_size', [4, 7])
def test_chunk_conv_states_matches_gather(conv_kernel_size):
    """Conv state at every chunk boundary must equal the raw-token gather.

    Mirrors the per-sequence-end extraction in CausalConv1dFunc.conv1d_func
    (``x[0, cu_seqlens[1:] + arange(-W, 0)]``) but at each chunk boundary,
    clamped to the sequence start (no cross-sequence reads, zero-padded below).
    """
    torch.manual_seed(7)
    W = conv_kernel_size
    lengths = [63, 65, 130]
    total = sum(lengths)
    dim = 32
    x = torch.randn(1, total, dim, device='cuda', dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 63, 128, 258], device='cuda', dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)

    out = chunk_conv_states(x, W, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    assert out.shape == (1, len(chunk_indices), dim, W)

    ref = torch.zeros_like(out)
    for c in range(len(chunk_indices)):
        seq_id = int(chunk_indices[c, 0].item())
        local_c = int(chunk_indices[c, 1].item())
        bos = int(cu_seqlens[seq_id].item())
        boundary = bos + local_c * 64
        for j in range(W):
            pos = boundary - W + j
            if pos >= bos:
                ref[0, c, :, j] = x[0, pos]

    torch.testing.assert_close(out, ref, atol=0, rtol=0)

    # boundary 0 is the first chunk of the first sequence: all W positions
    # precede bos=0, so the conv state must be all zero (fresh sequence).
    torch.testing.assert_close(out[0, 0], torch.zeros_like(out[0, 0]), atol=0, rtol=0)

    # re-assert the kernel for the last chunk slot of sequence 0 against an
    # independent gather written in the opposite (dim-last) layout, confirming
    # the stored row-major [dim, W] block matches a transposed [W, dim] gather.
    seq_ids = chunk_indices[:, 0]
    last_seq0_slot = int((seq_ids == 0).nonzero()[-1].item())
    bos0 = int(cu_seqlens[0].item())
    boundary_last = bos0 + int(chunk_indices[last_seq0_slot, 1].item()) * 64
    expected_tail = torch.zeros(W, dim, device=x.device, dtype=x.dtype)
    for j in range(W):
        pos = boundary_last - W + j
        if pos >= bos0:
            expected_tail[j] = x[0, pos]
    torch.testing.assert_close(out[0, last_seq0_slot].transpose(-2, -1), expected_tail, atol=0, rtol=0)


@pytest.mark.skipif(not _cuda_available(), reason='CUDA is not available')
@pytest.mark.parametrize('boundary_chunk', [0, 1, 2])
@pytest.mark.parametrize('conv_kernel_size', [4, 7])
def test_chunk_conv_states_restore_and_run_suffix_reaches_full(boundary_chunk, conv_kernel_size):
    """Conv half of the Marconi restore invariant.

    ``chunk_conv_states[:, c]`` stores the ``W`` raw conv-input tokens at
    ``[boundary_c - W, boundary_c - 1]``. Seeding a causal conv from its last
    ``W - 1`` tokens (the conv state) and running the suffix ``x[boundary:]``
    must reproduce the full conv output over that suffix. This is the conv
    counterpart to ``test_chunk_states_restore_and_run_suffix_reaches_final_state``
    and, unlike ``test_chunk_conv_states_matches_gather``, uses an independent
    ``F.conv1d`` reference (not a gather derived from ``chunk_indices``), so a
    chunk-index offset cannot be masked by both sides agreeing.
    """
    torch.manual_seed(11)
    W = conv_kernel_size
    length, dim = 200, 48
    # channel-last x, matching CausalConv1dFunc usage
    x = torch.randn(1, length, dim, device='cuda', dtype=torch.bfloat16)
    conv_states = chunk_conv_states(x, W)

    num_chunks = conv_states.shape[1]
    c = min(boundary_chunk, num_chunks - 1)
    boundary = c * 64
    suffix_len = length - boundary
    if suffix_len <= 0:
        pytest.skip('suffix is empty for this boundary')

    # Full conv reference (channel-first F.conv1d), independent of chunk_indices.
    # Seed with W-1 zeros so the full output starts at position 0 (matching the
    # suffix run, which always seeds from a boundary state); a plain conv1d would
    # drop the first W-1 positions and not align with the suffix slice.
    weight = torch.randn(dim, W, device='cuda', dtype=torch.bfloat16)
    bias = torch.randn(dim, device='cuda', dtype=torch.bfloat16)
    x_cf = x.transpose(1, 2)  # [1, dim, length]
    zero_seed = torch.zeros(1, dim, W - 1, device='cuda', dtype=torch.bfloat16)
    out_full = torch.nn.functional.conv1d(
        torch.cat([zero_seed, x_cf], dim=-1).float(),
        weight.float().unsqueeze(1), bias.float(), padding=0, groups=dim)
    out_full = out_full[..., :length].to(torch.bfloat16)  # [1, dim, length]

    # Suffix conv: prepend the W-1 saved tokens as the initial state.
    # conv_states[0, c] is [dim, W]; columns [1:W] are the W-1 most-recent
    # tokens immediately preceding the boundary -> the conv initial state.
    init = conv_states[0, c, :, 1:W].clone()  # [dim, W-1]
    x_suffix = x_cf[:, :, boundary:]           # [1, dim, suffix_len]
    x_seed = torch.cat([init.unsqueeze(0), x_suffix], dim=-1)  # [1, dim, W-1 + suffix]
    out_suffix = torch.nn.functional.conv1d(
        x_seed.float(), weight.float().unsqueeze(1), bias.float(), padding=0, groups=dim)
    out_suffix = out_suffix[..., :suffix_len].to(torch.bfloat16)  # [1, dim, suffix_len]

    torch.testing.assert_close(out_suffix, out_full[:, :, boundary:boundary + suffix_len],
                               atol=0, rtol=0)

    # Boundary 0 of a fresh sequence: all W positions precede bos=0, so the
    # saved state is all zero and seeding from it is a no-op (== no init).
    if c == 0:
        torch.testing.assert_close(conv_states[0, 0], torch.zeros_like(conv_states[0, 0]),
                                   atol=0, rtol=0)


@pytest.mark.skipif(not _cuda_available(), reason='CUDA is not available')
def test_chunk_conv_states_dense_and_contiguous_view():
    """Dense (no cu_seqlens) path and a non-contiguous view input both work."""
    torch.manual_seed(8)
    W = 4
    length, dim = 200, 48
    x_dense = torch.randn(1, length, dim, device='cuda', dtype=torch.bfloat16)
    out_dense = chunk_conv_states(x_dense, W)
    assert out_dense.shape == (1, triton.cdiv(length, 64), dim, W)

    # non-contiguous slice view (e.g. a channel subset) should be handled by the
    # entry contiguous() guard, matching the gated-delta kernel's input_guard.
    x_view = x_dense[:, :, :32]
    assert not x_view.is_contiguous()
    out_view = chunk_conv_states(x_view, W)
    assert out_view.shape == (1, triton.cdiv(length, 64), 32, W)
    torch.testing.assert_close(out_view, chunk_conv_states(x_view.contiguous(), W), atol=0, rtol=0)
