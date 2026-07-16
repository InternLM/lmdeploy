from dataclasses import replace

import pytest
import torch

from tests.turbomind.linear_attn import turbomind_gated_delta_rule
from tests.turbomind.linear_attn.benchmark import (
    CP_FALLBACK_SEGMENT_LOG_DECAY,
    SuiteRequest,
    apply_cp_pattern,
    make_families,
    make_parser,
    select_runs,
    validate_cp_request,
)
from tests.turbomind.linear_attn.cases import (
    Fixed,
    Heads,
    InputCase,
    InputTensors,
    RunCase,
    Varlen,
    make_grouped_state_buffer,
    make_input_tensors,
    make_packed_qkv_views,
    make_state_buffer,
)
from tests.turbomind.linear_attn.reference import chunk_gated_delta_rule_fwd

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for native delta-rule tests')


def _device_capability():
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


non_sm90_or_sm120 = torch.cuda.is_available() and _device_capability() not in ((9, 0), (12, 0))
is_sm120 = _device_capability() == (12, 0)


def _native_state_dtype(state_dtype: str) -> str:
    if state_dtype == 'f16':
        return 'f16'
    return state_dtype


def generated_smoke_runs():
    if not torch.cuda.is_available():
        chunk_size = 64
    else:
        major, minor = torch.cuda.get_device_capability()
        arch = major * 100 + minor * 10
        chunk_size = 16 if arch < 900 else 32 if arch == 1200 else 64
    return select_runs(SuiteRequest(suite='smoke', chunk_sizes=(chunk_size,)), make_families())


def _packed_slice(tensor, q_offsets, batch_idx):
    start = int(q_offsets[batch_idx].item())
    end = int(q_offsets[batch_idx + 1].item())
    return tensor[:, start:end]


def _assert_varlen_close(actual, expected, q_offsets, rtol=3e-2, atol=3e-2):
    for batch_idx in range(int(q_offsets.numel()) - 1):
        torch.testing.assert_close(
            _packed_slice(actual, q_offsets, batch_idx),
            _packed_slice(expected, q_offsets, batch_idx),
            rtol=rtol,
            atol=atol,
        )


def _dense_offsets(run: RunCase, device):
    assert isinstance(run.input.layout, Fixed)
    return torch.arange(
        0,
        (run.input.layout.batch_size + 1) * run.input.layout.seq_len,
        run.input.layout.seq_len,
        device=device,
        dtype=torch.int32,
    )


def _run_q_offsets(run: RunCase, tensors):
    if tensors.offsets is not None:
        return tensors.offsets.to(dtype=torch.int32).contiguous()
    if isinstance(run.input.layout, Fixed) and run.input.layout.seq_len == 1:
        return None
    return _dense_offsets(run, tensors.q.device)


def _native_plan(run: RunCase, *, chunk_size):
    tm = turbomind_gated_delta_rule._load_native_bridge()
    if tm is None:
        pytest.skip('TurboMind native delta-rule bridge is unavailable')

    device = torch.device('cuda')
    bridge = turbomind_gated_delta_rule.NativeBridge(tm)
    tensors = make_input_tensors(run.input, device=device)
    native_tensors = turbomind_gated_delta_rule._native_aligned_tensors(tensors)
    state = make_state_buffer(tensors.h0, run, device)
    run_inputs, kwargs, recurrent = turbomind_gated_delta_rule._state_kwargs(
        run,
        tensors,
        native_tensors,
        device,
        bridge,
        state,
        run.state_dtype,
    )
    return bridge.plan(
        run_inputs,
        q_offsets=kwargs.get('q_offsets'),
        state_dtype=_native_state_dtype(run.state_dtype),
        mode='recurrent' if recurrent else 'chunked',
        chunk_size=chunk_size,
        cp_mode='auto',
        num_head_groups=1,
        heads_per_block=run.input.heads.hv,
    )


def _recurrent_and_chunked_smoke_runs():
    runs = generated_smoke_runs()
    recurrent = next(
        run
        for run in runs
        if isinstance(run.input.layout, Fixed) and run.input.layout.seq_len == 1
    )
    chunked = next(
        run
        for run in runs
        if not isinstance(run.input.layout, Fixed) or run.input.layout.seq_len > 1
    )
    return recurrent, chunked


def _expected_arch_class() -> str:
    major, minor = torch.cuda.get_device_capability()
    arch = major * 100 + minor * 10
    if arch < 900:
        return 'pre_sm90'
    if arch == 900:
        return 'sm90'
    if arch == 1200:
        return 'sm120'
    pytest.skip(f'no successful GDR entry is defined for arch={arch}')


def _canonical_native_inputs(inputs: InputTensors) -> InputTensors:
    aligned = turbomind_gated_delta_rule._native_aligned_tensors(inputs)
    native = make_packed_qkv_views(aligned)
    assert native.g.dtype == torch.float32
    assert native.beta.dtype == torch.float32
    assert native.g.stride() == native.beta.stride()
    assert native.g.stride(2) == 1
    assert native.g.stride(1) % 4 == 0
    assert native.g.stride(0) % 4 == 0
    return native


def _supported_arch_cases():
    if not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    arch = major * 100 + minor * 10
    if arch < 900:
        input_dtypes = [torch.float16]
        if torch.cuda.is_bf16_supported():
            input_dtypes.append(torch.bfloat16)
        chunk_size = 16
    elif arch == 900:
        input_dtypes = [torch.bfloat16]
        chunk_size = 64
    elif arch == 1200:
        input_dtypes = [torch.bfloat16]
        chunk_size = 32
    else:
        return []

    cases = []
    for input_dtype in input_dtypes:
        activation_state = 'f16' if input_dtype == torch.float16 else 'bf16'
        for state_dtype in (activation_state, 'f32'):
            cases.append((input_dtype, state_dtype, 1, 'recurrent'))
            cases.append((input_dtype, state_dtype, chunk_size, 'chunked'))
    return cases


@cuda_required
@pytest.mark.parametrize(
    'input_dtype,state_dtype,chunk_size,mode',
    _supported_arch_cases(),
    ids=lambda value: str(value).replace('torch.', ''),
)
def test_registered_arch_matrix_matches_reference(
    input_dtype, state_dtype, chunk_size, mode
):
    seq_len = 1 if mode == 'recurrent' else chunk_size + 7
    case = InputCase(
        layout=Fixed(batch_size=2, seq_len=seq_len),
        heads=Heads(hq=2, hv=10),
        input_dtype=input_dtype,
        has_h0=True,
        seed=72000 + chunk_size,
    )
    run = RunCase(input=case, state_dtype=state_dtype, chunk_size=chunk_size)
    inputs = make_input_tensors(case, device='cuda')
    native_inputs = _canonical_native_inputs(inputs)
    state = make_state_buffer(inputs.h0, run, torch.device('cuda'))
    initial = state.storage.float().clone()
    q_offsets = None if mode == 'recurrent' else _dense_offsets(run, torch.device('cuda'))
    finished = torch.zeros(2, device='cuda', dtype=torch.bool)
    bridge = turbomind_gated_delta_rule.NativeBridge(
        turbomind_gated_delta_rule._require_native_bridge())
    native_state_dtype = _native_state_dtype(state_dtype)
    plan = bridge.plan(
        native_inputs,
        state_dtype=native_state_dtype,
        mode=mode,
        chunk_size=None,
        cp_mode='off',
        q_offsets=q_offsets,
        num_head_groups=1,
        heads_per_block=10,
    )
    assert plan['kernel'] == {
        'arch': _expected_arch_class(),
        'mode': mode,
        'input_dtype': 'f16' if input_dtype == torch.float16 else 'bf16',
        'state_dtype': native_state_dtype,
        'head_dim': 128,
        'chunk_size': chunk_size,
    }
    expected_o, expected_state = chunk_gated_delta_rule_fwd(
        inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta,
        initial_state=initial, chunk_size=chunk_size)
    state.reset(inputs.h0)
    actual_o = turbomind_gated_delta_rule.chunk_gated_delta_rule_fwd(
        native_inputs.q, native_inputs.k, native_inputs.v,
        native_inputs.g, native_inputs.beta,
        state_ptrs=state.ptrs[:, None], q_offsets=q_offsets,
        finished=finished,
        state_dtype=native_state_dtype, mode=mode, cp_mode='off',
        num_head_groups=1, heads_per_block=10, plan=plan)
    torch.testing.assert_close(actual_o, expected_o, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(state.storage.float(), expected_state, rtol=8e-2, atol=8e-2)


@cuda_required
def test_auto_plan_binds_one_complete_registered_operation():
    if _expected_arch_class() not in ('sm90', 'sm120'):
        pytest.skip('optimized full-operation entries require SM90 or SM120')
    recurrent_run, chunked_run = _recurrent_and_chunked_smoke_runs()

    recurrent_plan = _native_plan(recurrent_run, chunk_size=None)
    chunked_plan = _native_plan(chunked_run, chunk_size=None)

    assert recurrent_plan['kernel'] == {
        'arch': _expected_arch_class(),
        'mode': 'recurrent',
        'input_dtype': 'bf16',
        'state_dtype': 'f32',
        'head_dim': 128,
        'chunk_size': 1,
    }
    assert chunked_plan['kernel'] == {
        'arch': _expected_arch_class(),
        'mode': 'chunked',
        'input_dtype': 'bf16',
        'state_dtype': 'f32',
        'head_dim': 128,
        'chunk_size': 64 if _expected_arch_class() == 'sm90' else 32,
    }
    if _expected_arch_class() == 'sm90':
        assert chunked_plan['problem']['chunk_size'] == 64
    else:
        assert chunked_plan['problem']['chunk_size'] == 32
        assert chunked_plan['cp']['enabled'] is False


def test_benchmark_chunk_size_defaults_to_auto():
    args = make_parser().parse_args([])
    assert args.chunk_size is None
    assert args.cp_pattern == 'auto'


@pytest.mark.parametrize('cp_pattern', ('auto', 'warmup', 'fallback', 'alternating'))
def test_benchmark_cp_pattern_choices(cp_pattern):
    args = make_parser().parse_args(['--cp-pattern', cp_pattern])
    assert args.cp_pattern == cp_pattern


def test_cp_pattern_supports_cp_backends():
    validate_cp_request('auto', 'alternating', 'turbomind')
    validate_cp_request('auto', 'alternating', 'flashqla')
    with pytest.raises(ValueError, match='cp_pattern_requires_cp_backend'):
        validate_cp_request('auto', 'alternating', 'reference')


def _cp_pattern_inputs(case):
    if isinstance(case.layout, Fixed):
        g_shape = (case.layout.batch_size, case.layout.seq_len, case.heads.hv)
    else:
        g_shape = (1, case.layout.total_tokens, case.heads.hv)
    unused = torch.empty(0)
    return InputTensors(
        q=unused,
        k=unused,
        v=unused,
        g=torch.randn(g_shape),
        beta=unused,
        h0=None,
        offsets=None,
    )


def test_cp_pattern_full_modes_control_segment_decay():
    case = InputCase(
        layout=Fixed(batch_size=1, seq_len=128),
        heads=Heads(hq=2, hv=4),
        input_dtype=torch.bfloat16,
        has_h0=False,
        seed=72001,
    )
    inputs = _cp_pattern_inputs(case)

    assert apply_cp_pattern(case, inputs, 'auto', 64) is inputs

    warmup = apply_cp_pattern(case, inputs, 'warmup', 64)
    torch.testing.assert_close(warmup.g, torch.full_like(warmup.g, -1.0), rtol=0, atol=0)

    fallback = apply_cp_pattern(case, inputs, 'fallback', 64)
    segment_sums = fallback.g.reshape(1, 2, 64, 4).sum(dim=2)
    torch.testing.assert_close(
        segment_sums,
        torch.full_like(segment_sums, CP_FALLBACK_SEGMENT_LOG_DECAY),
        rtol=0,
        atol=1e-6,
    )


def test_cp_pattern_alternates_seeded_balanced_head_starts():
    case = InputCase(
        layout=Fixed(batch_size=2, seq_len=192),
        heads=Heads(hq=2, hv=4),
        input_dtype=torch.bfloat16,
        has_h0=False,
        seed=72002,
    )
    inputs = _cp_pattern_inputs(case)

    first = apply_cp_pattern(case, inputs, 'alternating', 64)
    second = apply_cp_pattern(case, inputs, 'alternating', 64)
    torch.testing.assert_close(first.g, second.g, rtol=0, atol=0)

    for batch in range(2):
        starts_with_warmup = first.g[batch, 0] == -1.0
        assert int(starts_with_warmup.sum()) == 2
        for local_segment in range(3):
            segment = first.g[batch, local_segment * 64:(local_segment + 1) * 64]
            uses_warmup = segment[0] == -1.0
            expected_warmup = starts_with_warmup if local_segment % 2 == 0 else ~starts_with_warmup
            torch.testing.assert_close(uses_warmup, expected_warmup, rtol=0, atol=0)
            segment_sums = segment.sum(dim=0)
            torch.testing.assert_close(
                segment_sums[uses_warmup],
                torch.full_like(segment_sums[uses_warmup], -64.0),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                segment_sums[~uses_warmup],
                torch.full_like(segment_sums[~uses_warmup], CP_FALLBACK_SEGMENT_LOG_DECAY),
                rtol=0,
                atol=1e-6,
            )


def test_cp_pattern_alternating_resets_head_parity_for_varlen_sequences():
    case = InputCase(
        layout=Varlen(offsets=(0, 128, 320)),
        heads=Heads(hq=1, hv=3),
        input_dtype=torch.bfloat16,
        has_h0=False,
        seed=72003,
    )
    inputs = _cp_pattern_inputs(case)

    controlled = apply_cp_pattern(case, inputs, 'alternating', 64)
    for begin, end in zip(case.layout.offsets[:-1], case.layout.offsets[1:]):
        starts_with_warmup = controlled.g[0, begin] == -1.0
        assert int(starts_with_warmup.sum()) in (1, 2)
        for local_segment, segment_begin in enumerate(range(begin, end, 64)):
            uses_warmup = controlled.g[0, segment_begin] == -1.0
            expected_warmup = starts_with_warmup if local_segment % 2 == 0 else ~starts_with_warmup
            torch.testing.assert_close(uses_warmup, expected_warmup, rtol=0, atol=0)


def test_cp_pattern_alternates_with_one_head():
    case = InputCase(
        layout=Fixed(batch_size=1, seq_len=128),
        heads=Heads(hq=1, hv=1),
        input_dtype=torch.bfloat16,
        has_h0=False,
        seed=72004,
    )
    inputs = _cp_pattern_inputs(case)
    first = apply_cp_pattern(case, inputs, 'alternating', 64)
    second = apply_cp_pattern(case, inputs, 'alternating', 64)

    torch.testing.assert_close(first.g, second.g, rtol=0, atol=0)
    first_segment_warmup = bool((first.g[0, 0, 0] == -1.0).item())
    second_segment_warmup = bool((first.g[0, 64, 0] == -1.0).item())
    assert first_segment_warmup != second_segment_warmup


@cuda_required
@pytest.mark.skipif(non_sm90_or_sm120, reason='GDR target requires SM90 or SM120')
def test_auto_chunk_size_plan_distinguishes_recurrent_and_chunked():
    recurrent_run, chunked_run = _recurrent_and_chunked_smoke_runs()

    recurrent_plan = _native_plan(recurrent_run, chunk_size=None)
    chunked_plan = _native_plan(chunked_run, chunk_size=None)

    assert recurrent_plan['problem']['chunk_size'] == 1
    assert recurrent_plan['problem']['recurrent'] is True
    assert chunked_plan['problem']['chunk_size'] > 1
    assert chunked_plan['problem']['recurrent'] is False


@cuda_required
@pytest.mark.skipif(non_sm90_or_sm120, reason='GDR target requires SM90 or SM120')
def test_explicit_recurrent_chunk_one_matches_reference():
    run, _ = _recurrent_and_chunked_smoke_runs()
    tensors = make_input_tensors(run.input, device='cuda')
    state = make_state_buffer(tensors.h0, run, torch.device('cuda'))
    expected_o, expected_state = chunk_gated_delta_rule_fwd(
        tensors.q,
        tensors.k,
        tensors.v,
        tensors.g,
        tensors.beta,
        initial_state=state.storage.float().clone(),
        chunk_size=run.chunk_size,
    )
    state.reset(tensors.h0)

    actual_o = turbomind_gated_delta_rule.chunk_gated_delta_rule_fwd(
        tensors.q,
        tensors.k,
        tensors.v,
        tensors.g,
        tensors.beta,
        state_ptrs=state.ptrs,
        finished=torch.zeros(run.input.real_batch_size, device='cuda', dtype=torch.bool),
        state_dtype=_native_state_dtype(run.state_dtype),
        chunk_size=1,
        cp_mode='auto',
    )

    torch.testing.assert_close(actual_o, expected_o, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(state.storage.float(), expected_state, rtol=8e-2, atol=8e-2)


@cuda_required
@pytest.mark.skipif(non_sm90_or_sm120, reason='GDR target requires SM90 or SM120')
def test_omitted_run_chunk_size_reuses_explicit_chunked_plan():
    run, _ = _recurrent_and_chunked_smoke_runs()
    device = torch.device('cuda')
    tensors = make_input_tensors(run.input, device=device)
    state = make_state_buffer(tensors.h0, run, device)
    q_offsets = _dense_offsets(run, device)
    finished = torch.zeros(run.input.real_batch_size, device=device, dtype=torch.bool)
    bridge = turbomind_gated_delta_rule.NativeBridge(
        turbomind_gated_delta_rule._require_native_bridge()
    )
    plan = bridge.plan(
        tensors,
        q_offsets=q_offsets,
        state_dtype=_native_state_dtype(run.state_dtype),
        mode='chunked',
        chunk_size=run.chunk_size,
        cp_mode='off',
        num_head_groups=1,
        heads_per_block=run.input.heads.hv,
    )
    expected_o, expected_state = chunk_gated_delta_rule_fwd(
        tensors.q,
        tensors.k,
        tensors.v,
        tensors.g,
        tensors.beta,
        initial_state=state.storage.float().clone(),
        chunk_size=run.chunk_size,
    )
    state.reset(tensors.h0)

    actual_o = turbomind_gated_delta_rule.chunk_gated_delta_rule_fwd(
        tensors.q,
        tensors.k,
        tensors.v,
        tensors.g,
        tensors.beta,
        state_ptrs=state.ptrs,
        finished=finished,
        state_dtype=_native_state_dtype(run.state_dtype),
        chunk_size=None,
        cp_mode='off',
        plan=plan,
    )

    torch.testing.assert_close(actual_o, expected_o, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(state.storage.float(), expected_state, rtol=8e-2, atol=8e-2)


@cuda_required
@pytest.mark.skipif(non_sm90_or_sm120, reason='GDR target requires SM90 or SM120')
def test_bound_chunked_plan_ignores_conflicting_runtime_chunk_size(monkeypatch):
    _, run = _recurrent_and_chunked_smoke_runs()
    device = torch.device('cuda')
    tensors = make_input_tensors(run.input, device=device)
    state = make_state_buffer(tensors.h0, run, device)
    finished = torch.zeros(run.input.real_batch_size, device=device, dtype=torch.bool)
    out = torch.empty_like(tensors.v)
    plan = {
        'kernel': {'mode': 'chunked'},
        'problem': {'chunk_size': run.chunk_size},
        'workspace_bytes': 0,
    }
    captured = {}

    def capture_run(self, tensors, *, plan, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(turbomind_gated_delta_rule.NativeBridge, 'run', capture_run)

    actual_o = turbomind_gated_delta_rule.chunk_gated_delta_rule_fwd(
        tensors.q,
        tensors.k,
        tensors.v,
        tensors.g,
        tensors.beta,
        state_ptrs=state.ptrs,
        finished=finished,
        state_dtype=_native_state_dtype(run.state_dtype),
        chunk_size=1,
        cp_mode='off',
        out=out,
        state_tma_descs=torch.empty(0, device=device, dtype=torch.uint8),
        plan=plan,
    )

    assert actual_o is out
    assert captured['q_offsets'] is not None
    torch.testing.assert_close(captured['q_offsets'], _run_q_offsets(run, tensors))
    assert captured['state_ptrs'].ndim == 1


@cuda_required
@pytest.mark.skipif(non_sm90_or_sm120, reason='GDR target requires SM90 or SM120')
@pytest.mark.parametrize('run', generated_smoke_runs(), ids=lambda run: run.name)
def test_delta_rule_wrapper_all_generated_smoke_matches_reference(run):
    if not turbomind_gated_delta_rule.is_available():
        pytest.skip('TurboMind native delta-rule bridge is unavailable')

    tensors = make_input_tensors(run.input, device='cuda')
    state = make_state_buffer(tensors.h0, run, torch.device('cuda'))
    expected_o, expected_state = chunk_gated_delta_rule_fwd(
        tensors.q,
        tensors.k,
        tensors.v,
        tensors.g,
        tensors.beta,
        initial_state=state.storage.float().clone(),
        cu_seqlens=tensors.offsets,
        chunk_size=run.chunk_size,
    )
    state.reset(tensors.h0)
    actual_o = torch.empty_like(tensors.v)
    state_dtype = _native_state_dtype(run.state_dtype)
    q_offsets = _run_q_offsets(run, tensors)

    if q_offsets is None:
        finished = torch.zeros(run.input.real_batch_size, device='cuda', dtype=torch.bool)
    else:
        finished = torch.zeros(q_offsets.numel() - 1, device='cuda', dtype=torch.bool)

    turbomind_gated_delta_rule.chunk_gated_delta_rule_fwd(
        tensors.q,
        tensors.k,
        tensors.v,
        tensors.g,
        tensors.beta,
        state_ptrs=state.ptrs,
        q_offsets=q_offsets,
        finished=finished,
        state_dtype=state_dtype,
        cp_mode='auto',
        out=actual_o,
    )

    if tensors.offsets is None:
        torch.testing.assert_close(actual_o, expected_o, rtol=8e-2, atol=8e-2)
    else:
        _assert_varlen_close(actual_o, expected_o, tensors.offsets, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(state.storage.float(), expected_state, rtol=8e-2, atol=8e-2)


@cuda_required
@pytest.mark.skipif(non_sm90_or_sm120, reason='optimized GDR target requires SM90 or SM120')
def test_packed_strided_qkv_matches_normalized_reference():
    _, run = _recurrent_and_chunked_smoke_runs()
    reference_inputs = make_input_tensors(run.input, device='cuda')
    native_inputs = _canonical_native_inputs(reference_inputs)
    state = make_state_buffer(reference_inputs.h0, run, torch.device('cuda'))
    initial = state.storage.float().clone()
    expected_o, expected_state = chunk_gated_delta_rule_fwd(
        reference_inputs.q,
        reference_inputs.k,
        reference_inputs.v,
        reference_inputs.g,
        reference_inputs.beta,
        initial_state=initial,
        cu_seqlens=reference_inputs.offsets,
        chunk_size=run.chunk_size,
    )
    state.reset(reference_inputs.h0)
    actual_o = turbomind_gated_delta_rule.chunk_gated_delta_rule_fwd(
        native_inputs.q,
        native_inputs.k,
        native_inputs.v,
        native_inputs.g,
        native_inputs.beta,
        state_ptrs=state.ptrs[:, None],
        q_offsets=_run_q_offsets(run, native_inputs),
        finished=torch.zeros(run.input.real_batch_size, device='cuda', dtype=torch.bool),
        state_dtype=_native_state_dtype(run.state_dtype),
        mode='chunked',
        num_head_groups=1,
        heads_per_block=run.input.heads.hv,
    )
    hq = native_inputs.q.shape[2]
    hv = native_inputs.v.shape[2]
    conv_dim = (2 * hq + hv) * 128
    storage_ptr = native_inputs.q.untyped_storage().data_ptr()
    assert native_inputs.k.untyped_storage().data_ptr() == storage_ptr
    assert native_inputs.v.untyped_storage().data_ptr() == storage_ptr
    assert native_inputs.q.storage_offset() == 0
    assert native_inputs.k.storage_offset() == hq * 128
    assert native_inputs.v.storage_offset() == 2 * hq * 128
    assert native_inputs.q.stride(1) == conv_dim
    assert native_inputs.k.stride(1) == conv_dim
    assert native_inputs.v.stride(1) == conv_dim
    assert native_inputs.q.stride()[2:] == (128, 1)
    assert native_inputs.k.stride()[2:] == (128, 1)
    assert native_inputs.v.stride()[2:] == (128, 1)
    torch.testing.assert_close(actual_o, expected_o, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(state.storage.float(), expected_state, rtol=8e-2, atol=8e-2)


@cuda_required
@pytest.mark.skipif(non_sm90_or_sm120, reason='optimized GDR target requires SM90 or SM120')
def test_grouped_partial_head_block_and_nonzero_layer_offset_match_reference():
    case = InputCase(
        layout=Fixed(batch_size=2, seq_len=1),
        heads=Heads(hq=1, hv=5),
        input_dtype=torch.bfloat16,
        has_h0=True,
        seed=71001,
    )
    inputs = make_input_tensors(case, device='cuda')
    native_inputs = _canonical_native_inputs(inputs)
    grouped = make_grouped_state_buffer(
        inputs.h0,
        state_dtype=torch.float32,
        layers_per_block=2,
        heads_per_block=3,
        layer=1,
    )
    untouched = grouped.logical(0).clone()
    padding = grouped.blocks[:, -1, :, 2:].clone()
    initial = grouped.logical(1).clone()
    expected_o, expected_state = chunk_gated_delta_rule_fwd(
        inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta,
        initial_state=initial, chunk_size=1)
    actual_o = turbomind_gated_delta_rule.chunk_gated_delta_rule_fwd(
        native_inputs.q, native_inputs.k, native_inputs.v,
        native_inputs.g, native_inputs.beta,
        state_ptrs=grouped.ptrs,
        finished=torch.zeros(2, device='cuda', dtype=torch.bool),
        state_dtype='f32',
        mode='recurrent',
        state_layer_offset=3 * 128 * 128,
        num_head_groups=2,
        heads_per_block=3,
        layer_groups=1,
        layers_per_block=2,
    )
    torch.testing.assert_close(actual_o, expected_o, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(grouped.logical(1), expected_state, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(grouped.logical(0), untouched, rtol=0, atol=0)
    torch.testing.assert_close(grouped.blocks[:, -1, :, 2:], padding, rtol=0, atol=0)


@cuda_required
@pytest.mark.skipif(non_sm90_or_sm120, reason='optimized GDR target requires SM90 or SM120')
def test_chunked_nonzero_physical_start_matches_compact_reference():
    case = InputCase(
        layout=Fixed(batch_size=1, seq_len=113),
        heads=Heads(hq=1, hv=4),
        input_dtype=torch.bfloat16,
        has_h0=True,
        seed=71002,
    )
    inputs = make_input_tensors(case, device='cuda')
    native_inputs = _canonical_native_inputs(inputs)
    q_offsets = torch.tensor([7, 47, 113], device='cuda', dtype=torch.int32)
    compact_offsets = torch.tensor([0, 40, 106], device='cuda', dtype=torch.int32)
    compact = replace(
        inputs,
        q=inputs.q[:, 7:113].contiguous(),
        k=inputs.k[:, 7:113].contiguous(),
        v=inputs.v[:, 7:113].contiguous(),
        g=inputs.g[:, 7:113].contiguous(),
        beta=inputs.beta[:, 7:113].contiguous(),
        offsets=compact_offsets,
    )
    initial = inputs.h0.repeat(2, 1, 1, 1)
    expected_o, expected_state = chunk_gated_delta_rule_fwd(
        compact.q, compact.k, compact.v, compact.g, compact.beta,
        initial_state=initial.clone(), cu_seqlens=compact_offsets,
        chunk_size=64 if _expected_arch_class() == 'sm90' else 32)
    storage = initial.clone()
    ptrs = torch.tensor([storage[i].data_ptr() for i in range(2)], device='cuda', dtype=torch.int64)[:, None]
    actual_o = turbomind_gated_delta_rule.chunk_gated_delta_rule_fwd(
        native_inputs.q, native_inputs.k, native_inputs.v,
        native_inputs.g, native_inputs.beta,
        state_ptrs=ptrs, q_offsets=q_offsets,
        finished=torch.zeros(2, device='cuda', dtype=torch.bool),
        state_dtype='f32', mode='chunked', num_head_groups=1, heads_per_block=4)
    torch.testing.assert_close(actual_o[:, 7:113], expected_o, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(storage, expected_state, rtol=8e-2, atol=8e-2)


@cuda_required
@pytest.mark.skipif(non_sm90_or_sm120, reason='optimized GDR target requires SM90 or SM120')
@pytest.mark.parametrize('mode', ('recurrent', 'chunked'))
def test_finished_sequence_produces_output_without_storing_state(mode):
    chunk_size = 1 if mode == 'recurrent' else (64 if _expected_arch_class() == 'sm90' else 32)
    case = InputCase(
        layout=Fixed(batch_size=2, seq_len=1 if mode == 'recurrent' else chunk_size + 7),
        heads=Heads(hq=2, hv=10),
        input_dtype=torch.bfloat16,
        has_h0=True,
        seed=71003,
    )
    run = RunCase(input=case, state_dtype='f32', chunk_size=chunk_size)
    inputs = make_input_tensors(case, device='cuda')
    native_inputs = _canonical_native_inputs(inputs)
    state = make_state_buffer(inputs.h0, run, torch.device('cuda'))
    initial = state.storage.float().clone()
    expected_o, expected_state = chunk_gated_delta_rule_fwd(
        inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta,
        initial_state=initial.clone(), chunk_size=chunk_size)
    finished = torch.zeros(run.input.real_batch_size, device='cuda', dtype=torch.bool)
    finished[0] = True
    q_offsets = None if mode == 'recurrent' else _dense_offsets(run, torch.device('cuda'))
    actual_o = turbomind_gated_delta_rule.chunk_gated_delta_rule_fwd(
        native_inputs.q, native_inputs.k, native_inputs.v,
        native_inputs.g, native_inputs.beta,
        state_ptrs=state.ptrs[:, None], q_offsets=q_offsets, finished=finished,
        state_dtype='f32', mode=mode, cp_mode='off',
        num_head_groups=1, heads_per_block=10)
    torch.testing.assert_close(actual_o, expected_o, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(state.storage[0].float(), initial[0], rtol=0, atol=0)
    torch.testing.assert_close(state.storage[1:].float(), expected_state[1:], rtol=8e-2, atol=8e-2)


@cuda_required
@pytest.mark.skipif(non_sm90_or_sm120, reason='optimized GDR target requires SM90 or SM120')
def test_one_chunked_plan_reuses_grouped_state_across_layer_offsets():
    chunk_size = 64 if _expected_arch_class() == 'sm90' else 32
    case = InputCase(
        layout=Fixed(batch_size=2, seq_len=chunk_size + 7),
        heads=Heads(hq=2, hv=10),
        input_dtype=torch.bfloat16,
        has_h0=True,
        seed=71004,
    )
    inputs = make_input_tensors(case, device='cuda')
    native_inputs = _canonical_native_inputs(inputs)
    grouped = make_grouped_state_buffer(
        inputs.h0,
        state_dtype=torch.float32,
        layers_per_block=2,
        heads_per_block=4,
        layer=0,
    )
    grouped.set_logical(1, inputs.h0.float() * 0.5)
    q_offsets = torch.arange(
        0,
        3 * case.layout.seq_len,
        case.layout.seq_len,
        device='cuda',
        dtype=torch.int32,
    )
    finished = torch.zeros(2, device='cuda', dtype=torch.bool)
    bridge = turbomind_gated_delta_rule.NativeBridge(
        turbomind_gated_delta_rule._require_native_bridge())
    plan = bridge.plan(
        native_inputs,
        state_dtype='f32',
        mode='chunked',
        chunk_size=None,
        cp_mode='off',
        q_offsets=q_offsets,
        num_head_groups=3,
        heads_per_block=4,
    )

    layer_one_before = grouped.logical(1).clone()
    for layer in (0, 1):
        initial = grouped.logical(layer).clone()
        expected_o, expected_state = chunk_gated_delta_rule_fwd(
            inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta,
            initial_state=initial, chunk_size=chunk_size)
        actual_o = turbomind_gated_delta_rule.chunk_gated_delta_rule_fwd(
            native_inputs.q, native_inputs.k, native_inputs.v,
            native_inputs.g, native_inputs.beta,
            state_ptrs=grouped.ptrs,
            q_offsets=q_offsets,
            finished=finished,
            state_dtype='f32',
            mode='chunked',
            state_layer_offset=layer * 4 * 128 * 128,
            num_head_groups=3,
            heads_per_block=4,
            layer_groups=1,
            layers_per_block=2,
            plan=plan,
        )
        torch.testing.assert_close(actual_o, expected_o, rtol=8e-2, atol=8e-2)
        torch.testing.assert_close(grouped.logical(layer), expected_state, rtol=8e-2, atol=8e-2)
        if layer == 0:
            torch.testing.assert_close(grouped.logical(1), layer_one_before, rtol=0, atol=0)


@cuda_required
@pytest.mark.skipif(_device_capability() != (9, 0), reason='exact CP is retained only on SM90')
def test_sm90_auto_exact_cp_matches_normalized_reference():
    case = InputCase(
        layout=Fixed(batch_size=1, seq_len=4096),
        heads=Heads(hq=2, hv=8),
        input_dtype=torch.bfloat16,
        has_h0=True,
        seed=71005,
    )
    run = RunCase(input=case, state_dtype='f32', chunk_size=64)
    inputs = make_input_tensors(case, device='cuda')
    native_inputs = _canonical_native_inputs(inputs)
    state = make_state_buffer(inputs.h0, run, torch.device('cuda'))
    initial = state.storage.float().clone()
    q_offsets = _dense_offsets(run, torch.device('cuda'))
    finished = torch.zeros(1, device='cuda', dtype=torch.bool)
    bridge = turbomind_gated_delta_rule.NativeBridge(
        turbomind_gated_delta_rule._require_native_bridge())
    plan = bridge.plan(
        native_inputs,
        state_dtype='f32',
        mode='chunked',
        chunk_size=None,
        cp_mode='auto',
        q_offsets=q_offsets,
        num_head_groups=1,
        heads_per_block=8,
    )
    assert plan['cp']['enabled'] is True
    expected_o, expected_state = chunk_gated_delta_rule_fwd(
        inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta,
        initial_state=initial, chunk_size=64)
    state.reset(inputs.h0)
    actual_o = turbomind_gated_delta_rule.chunk_gated_delta_rule_fwd(
        native_inputs.q, native_inputs.k, native_inputs.v,
        native_inputs.g, native_inputs.beta,
        state_ptrs=state.ptrs[:, None],
        q_offsets=q_offsets,
        finished=finished,
        state_dtype='f32',
        mode='chunked',
        num_head_groups=1,
        heads_per_block=8,
        plan=plan,
    )
    torch.testing.assert_close(actual_o, expected_o, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(state.storage.float(), expected_state, rtol=8e-2, atol=8e-2)
    state.reset(inputs.h0)
    finished.fill_(True)
    finished_o = turbomind_gated_delta_rule.chunk_gated_delta_rule_fwd(
        native_inputs.q, native_inputs.k, native_inputs.v,
        native_inputs.g, native_inputs.beta,
        state_ptrs=state.ptrs[:, None],
        q_offsets=q_offsets,
        finished=finished,
        state_dtype='f32',
        mode='chunked',
        num_head_groups=1,
        heads_per_block=8,
        plan=plan,
    )
    torch.testing.assert_close(finished_o, expected_o, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(state.storage.float(), initial, rtol=0, atol=0)
