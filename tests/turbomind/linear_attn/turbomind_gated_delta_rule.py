from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .benchmark import (
    BenchmarkRequest,
    BenchmarkTask,
    apply_cp_pattern,
    base_row,
    diff_metrics,
)
from .cases import (
    Fixed,
    InputTensors,
    RunCase,
    make_packed_qkv_views,
    make_state_buffer,
)
from .reference import (
    chunk_gated_delta_rule_fwd as reference_chunk_gated_delta_rule_fwd,
)

REQUIRED_NATIVE_BRIDGE_SYMBOLS = (
    'delta_rule_plan',
    'delta_rule_run',
    'from_dlpack_with_strides',
    'delta_rule_prepare_state_tma_descs',
)


def _load_native_bridge(required_symbols=REQUIRED_NATIVE_BRIDGE_SYMBOLS):
    try:
        import _turbomind as tm
    except ImportError:
        return None
    return tm if all(hasattr(tm, symbol) for symbol in required_symbols) else None


def _require_native_bridge(required_symbols=REQUIRED_NATIVE_BRIDGE_SYMBOLS):
    tm = _load_native_bridge(required_symbols)
    if tm is None:
        missing = ', '.join(required_symbols)
        raise ImportError(f'TurboMind native delta-rule bridge is unavailable; required symbols: {missing}')
    return tm


def is_available() -> bool:
    return _load_native_bridge(REQUIRED_NATIVE_BRIDGE_SYMBOLS) is not None


@dataclass(frozen=True)
class NativeBridge:
    tm: object

    def tensor(self, x: torch.Tensor | None):
        return None if x is None else self.tm.from_dlpack_with_strides(x)

    def plan(
        self,
        tensors,
        *,
        q_offsets=None,
        state_dtype='f32',
        mode='chunked',
        chunk_size=None,
        cp_level='all',
        num_head_groups=1,
        heads_per_block=0,
    ):
        return self.tm.delta_rule_plan(
            self.tensor(tensors.q),
            self.tensor(tensors.k),
            self.tensor(tensors.v),
            self.tensor(tensors.g),
            self.tensor(tensors.beta),
            q_offsets=self.tensor(q_offsets),
            state_dtype=state_dtype,
            mode=mode,
            chunk_size=chunk_size,
            cp_level=cp_level,
            num_head_groups=num_head_groups,
            heads_per_block=heads_per_block,
        )

    def run(self, tensors, *, plan, **kwargs):
        return self.tm.delta_rule_run(
            self.tensor(tensors.q),
            self.tensor(tensors.k),
            self.tensor(tensors.v),
            self.tensor(tensors.g),
            self.tensor(tensors.beta),
            plan,
            out=self.tensor(kwargs.get('out')),
            workspace=self.tensor(kwargs.get('workspace')),
            stream_ptr=_current_stream_ptr(tensors.q),
            state_layer_offset=int(kwargs.get('state_layer_offset', 0)),
            state_ptrs=self.tensor(kwargs.get('state_ptrs')),
            state_tma_descs=self.tensor(kwargs.get('state_tma_descs')),
            q_offsets=self.tensor(kwargs.get('q_offsets')),
            finished=self.tensor(kwargs.get('finished')),
        )


def validate_benchmark_case(run: RunCase, request: BenchmarkRequest) -> None:
    if run.input.input_dtype != torch.bfloat16:
        raise ValueError('turbomind_requires_bf16_qkv')


def make_benchmark_task(run: RunCase, inputs: InputTensors, request: BenchmarkRequest,
                        device: torch.device) -> BenchmarkTask:
    validate_benchmark_case(run, request)
    tm = _load_native_bridge(REQUIRED_NATIVE_BRIDGE_SYMBOLS)
    if tm is None:
        raise RuntimeError('TurboMind native delta-rule bridge is unavailable')
    bridge = NativeBridge(tm)
    return _turbomind_task(run, inputs, request, device, bridge)


def _strided_cosize(shape, stride):
    if not shape:
        return 0
    return 1 + sum((dim - 1) * step for dim, step in zip(shape, stride))


def _current_stream_ptr(tensor: torch.Tensor) -> int:
    return int(torch.cuda.current_stream(tensor.device).cuda_stream)


def _state_dtype_arg(state_dtype: str) -> str:
    if state_dtype == 'f16':
        return 'f16'
    return 'bfloat16' if state_dtype == 'bf16' else 'f32'


def _q_offsets_for_tensors(tensors, *, recurrent=False):
    offsets = getattr(tensors, 'offsets', getattr(tensors, 'cu_seqlens', None))
    if offsets is not None:
        return offsets.to(device=tensors.q.device, dtype=torch.int32).contiguous()
    if recurrent and tensors.q.dtype == torch.bfloat16 and tensors.q.shape[1] == 1:
        return None
    batch = tensors.q.shape[0]
    seq_len = tensors.q.shape[1]
    return torch.arange(0, (batch + 1) * seq_len, seq_len, device=tensors.q.device, dtype=torch.int32)


def _sequence_num(case, q_offsets):
    return case.real_batch_size if q_offsets is None else int(q_offsets.numel() - 1)


def _padded_gate_view(x: torch.Tensor) -> torch.Tensor:
    batch, tokens, hv = x.shape
    gate_stride = max(4, hv)
    if gate_stride % 4:
        gate_stride += 4 - gate_stride % 4
    padded = x.new_zeros((batch, tokens, gate_stride))
    padded[..., :hv].copy_(x)
    return torch.as_strided(padded, x.shape, (tokens * gate_stride, gate_stride, 1))


def _native_aligned_tensors(tensors):
    if (tensors.g.stride(2) == 1 and tensors.beta.stride(2) == 1 and tensors.g.stride(1) % 4 == 0
            and tensors.beta.stride(1) % 4 == 0 and tensors.g.stride(0) % 4 == 0
            and tensors.beta.stride(0) % 4 == 0):
        return tensors
    return replace(tensors, g=_padded_gate_view(tensors.g), beta=_padded_gate_view(tensors.beta))


def _is_recurrent_run(run: RunCase, chunk_size: int | None = None) -> bool:
    if chunk_size is not None:
        return chunk_size == 1
    return (
        isinstance(run.input.layout, Fixed)
        and run.input.layout.seq_len == 1
        and run.input.input_dtype == torch.bfloat16
    )


def _state_kwargs(
    run: RunCase,
    inputs,
    native_inputs,
    device,
    bridge,
    state,
    state_dtype,
    chunk_size: int | None = None,
):
    del inputs, bridge, state_dtype
    if _is_recurrent_run(run, chunk_size):
        finished = torch.zeros(run.input.real_batch_size, device=device, dtype=torch.bool)
        return native_inputs, {
            'state_ptrs': state.ptrs[:, None],
            'finished': finished,
            'recurrent': True,
        }, True
    q_offsets = _q_offsets_for_tensors(native_inputs)
    sequence_num = _sequence_num(run.input, q_offsets)
    finished = torch.zeros(sequence_num, device=device, dtype=torch.bool)
    kwargs = {'state_ptrs': state.ptrs, 'finished': finished}
    if q_offsets is not None:
        kwargs['q_offsets'] = q_offsets
    return native_inputs, kwargs, False


def _empty_from_plan(plan, *, device, dtype):
    if 'stride' not in plan:
        return torch.empty(tuple(plan['shape']), device=device, dtype=dtype)
    shape = tuple(plan['shape'])
    stride = tuple(plan['stride'])
    storage_size = plan.get('storage_size')
    if storage_size is not None and int(storage_size) > _strided_cosize(shape, stride):
        storage = torch.empty(int(storage_size), device=device, dtype=dtype)
        return torch.as_strided(storage, shape, stride)
    return torch.empty_strided(shape, stride, device=device, dtype=dtype)


def _keep_alive(tensor: torch.Tensor | None, *refs) -> None:
    if tensor is not None:
        tensor._delta_rule_refs = refs


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    state_ptrs: torch.Tensor,
    q_offsets: torch.Tensor | None = None,
    finished: torch.Tensor,
    state_dtype: str = 'f32',
    mode: str | None = None,
    chunk_size: int | None = None,
    cp_level: str = 'all',
    state_layer_offset: int = 0,
    num_head_groups: int = 1,
    heads_per_block: int = 0,
    layer_groups: int = 1,
    layers_per_block: int = 1,
    out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    state_tma_descs: torch.Tensor | None = None,
    plan: object | None = None,
) -> torch.Tensor:
    bridge = NativeBridge(_require_native_bridge())
    if plan is not None:
        effective_chunk_size = int(plan['problem']['chunk_size'])
        recurrent = plan['kernel']['mode'] == 'recurrent'
    elif mode is not None:
        recurrent = mode == 'recurrent'
        effective_chunk_size = 1 if recurrent and chunk_size is None else chunk_size
    else:
        effective_chunk_size = chunk_size
        recurrent = effective_chunk_size == 1 or (
            effective_chunk_size is None and q_offsets is None and q.dtype == torch.bfloat16 and q.shape[1] == 1
        )
    if q_offsets is None and not recurrent:
        q_offsets = torch.arange(0, (q.shape[0] + 1) * q.shape[1], q.shape[1], device=q.device, dtype=torch.int32)

    tensors = InputTensors(q=q, k=k, v=v, g=g, beta=beta, h0=None, offsets=q_offsets)
    if plan is None:
        plan = bridge.plan(
            tensors,
            q_offsets=q_offsets,
            state_dtype=state_dtype,
            mode='recurrent' if recurrent else 'chunked',
            chunk_size=chunk_size,
            cp_level=cp_level,
            num_head_groups=num_head_groups,
            heads_per_block=heads_per_block or v.shape[2],
        )

    execution_state_ptrs = state_ptrs[:, None] if recurrent and state_ptrs.ndim == 1 else state_ptrs
    if recurrent and state_tma_descs is None:
        optimized_recurrent = plan['kernel']['arch'] != 'pre_sm90'
        prepared_descs = (
            torch.empty(
                (layer_groups, execution_state_ptrs.shape[-2], plan['problem']['num_head_groups'], 128),
                device=q.device,
                dtype=torch.uint8,
            )
            if optimized_recurrent
            else torch.empty(0, device=q.device, dtype=torch.uint8)
        )
        bridge.tm.delta_rule_prepare_state_tma_descs(
            bridge.tensor(execution_state_ptrs),
            bridge.tensor(prepared_descs),
            plan,
            layer_groups=layer_groups,
            layers_per_block=layers_per_block,
            stream_ptr=_current_stream_ptr(q),
        )
        state_tma_descs = prepared_descs if optimized_recurrent else None

    if out is None:
        out = _empty_from_plan(plan['out'], device=q.device, dtype=q.dtype)
    if workspace is None and plan['workspace_bytes']:
        workspace = torch.empty(plan['workspace_bytes'], device=q.device, dtype=torch.uint8)

    bridge.run(
        tensors,
        plan=plan,
        out=out,
        workspace=workspace,
        state_ptrs=execution_state_ptrs,
        q_offsets=q_offsets,
        finished=finished,
        state_tma_descs=state_tma_descs,
        state_layer_offset=state_layer_offset,
    )
    if workspace is not None:
        out._delta_rule_workspace = workspace
    _keep_alive(out, workspace, execution_state_ptrs, state_tma_descs, q_offsets, finished)
    return out


def _turbomind_task(run: RunCase, inputs: InputTensors, request: BenchmarkRequest, device: torch.device,
                    bridge: NativeBridge) -> BenchmarkTask:
    native_inputs = make_packed_qkv_views(_native_aligned_tensors(inputs))
    state_arg = _state_dtype_arg(request.state_dtype)
    chunk_size = request.chunk_size
    state = make_state_buffer(inputs.h0, run, device)
    run_inputs, kwargs, recurrent = _state_kwargs(
        run,
        inputs,
        native_inputs,
        device,
        bridge,
        state,
        request.state_dtype,
        chunk_size,
    )
    out = torch.empty_like(inputs.v)
    plan = bridge.plan(
        run_inputs,
        q_offsets=kwargs.get('q_offsets'),
        state_dtype=state_arg,
        mode='recurrent' if recurrent else 'chunked',
        chunk_size=chunk_size,
        cp_level=request.cp_level,
        num_head_groups=1,
        heads_per_block=run.input.heads.hv,
    )
    cp_enabled = bool(plan['cp']['enabled'])
    if request.cp_pattern != 'auto':
        if not cp_enabled:
            raise ValueError('cp_not_selected')
        segment_tokens = int(plan['cp']['segment_tokens'])
        if isinstance(run.input.layout, Fixed):
            has_handoff = run.input.layout.seq_len > segment_tokens
        else:
            has_handoff = any(
                end - begin > segment_tokens
                for begin, end in zip(run.input.layout.offsets[:-1], run.input.layout.offsets[1:])
            )
        if not has_handoff:
            raise ValueError('cp_pattern_requires_multiple_segments')
        controlled_inputs = apply_cp_pattern(
            run.input,
            inputs,
            request.cp_pattern,
            segment_tokens,
        )
        run_inputs.g.copy_(controlled_inputs.g)
        inputs = controlled_inputs
    if recurrent:
        optimized_recurrent = plan['kernel']['arch'] != 'pre_sm90'
        prepared_descs = (
            torch.empty(
                (1, run.input.real_batch_size, 1, 128),
                device=device,
                dtype=torch.uint8,
            )
            if optimized_recurrent
            else torch.empty(0, device=device, dtype=torch.uint8)
        )
        bridge.tm.delta_rule_prepare_state_tma_descs(
            bridge.tensor(kwargs['state_ptrs']),
            bridge.tensor(prepared_descs),
            plan,
            layer_groups=1,
            layers_per_block=1,
            stream_ptr=_current_stream_ptr(native_inputs.q),
        )
        if optimized_recurrent:
            state.tma_descs = prepared_descs
            kwargs['state_tma_descs'] = state.tma_descs
    planned_chunk_size = int(plan['problem']['chunk_size'])
    if recurrent:
        if planned_chunk_size != 1:
            raise RuntimeError(f'native recurrent plan selected chunk_size={planned_chunk_size}, expected 1')
    elif planned_chunk_size <= 1:
        raise RuntimeError(f'native chunked plan selected chunk_size={planned_chunk_size}, expected > 1')
    workspace = (
        torch.empty(plan['workspace_bytes'], device=device, dtype=torch.uint8)
        if plan['workspace_bytes']
        else None
    )
    planned_run = replace(run, chunk_size=planned_chunk_size)
    row = base_row(
        planned_run,
        'turbomind',
        state_dtype=request.state_dtype,
        chunk_size=planned_chunk_size,
        cp_level=request.cp_level,
        cp_pattern=request.cp_pattern,
        cp_enabled=cp_enabled,
    )
    def prepare():
        state.reset(inputs.h0)

    def execute():
        return chunk_gated_delta_rule_fwd(
            run_inputs.q,
            run_inputs.k,
            run_inputs.v,
            run_inputs.g,
            run_inputs.beta,
            state_ptrs=kwargs['state_ptrs'],
            q_offsets=kwargs.get('q_offsets'),
            finished=kwargs['finished'],
            state_dtype=state_arg,
            chunk_size=planned_chunk_size,
            cp_level=request.cp_level,
            out=out,
            workspace=workspace,
            state_tma_descs=kwargs.get('state_tma_descs'),
            plan=plan,
        )

    def validate():
        prepare()
        actual_o = execute()
        torch.cuda.synchronize(device)
        expected_o, expected_state = reference_chunk_gated_delta_rule_fwd(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.g,
            inputs.beta,
            initial_state=inputs.h0,
            cu_seqlens=inputs.offsets,
            chunk_size=planned_chunk_size,
        )
        if request.validate_outputs:
            torch.testing.assert_close(actual_o, expected_o, rtol=8e-2, atol=8e-2)
            torch.testing.assert_close(state.storage.float(), expected_state, rtol=8e-2, atol=8e-2)
        return {
            **diff_metrics('output', actual_o, expected_o),
            **diff_metrics('state', state.storage.float(), expected_state),
        }

    return BenchmarkTask(row, execute, prepare, validate)
