from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch

from tests.turbomind.linear_attn.benchmark import (
    BenchmarkRequest,
    BenchmarkTask,
    apply_cp_pattern,
    diff_metrics,
)
from tests.turbomind.linear_attn.cases import InputTensors, RunCase


class FlashQlaUnsupported(ValueError):
    pass


@dataclass(frozen=True)
class FlashQlaResult:
    out: torch.Tensor | None
    final_state: torch.Tensor | None
    cp_enabled: bool


@cache
def _imports() -> dict[str, object]:
    from flash_qla.ops.gated_delta_rule import chunk as qla_chunk
    from flash_qla.ops.gated_delta_rule.chunk.cp_context import (
        _calc_cp_seqs,
        intra_card_cp_preprocess,
    )
    from flash_qla.ops.utils import chunk_local_cumsum

    return {
        'chunk_local_cumsum': chunk_local_cumsum,
        'kkt_solve': qla_chunk.kkt_solve,
        'fused_gdr_fwd': qla_chunk.fused_gdr_fwd,
        'calc_cp_seqs': _calc_cp_seqs,
        'intra_card_cp_preprocess': intra_card_cp_preprocess,
    }


def is_available() -> bool:
    try:
        _imports()
    except (ImportError, OSError, RuntimeError, ValueError):
        return False
    return True


def all_forward(q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                g: torch.Tensor,
                beta: torch.Tensor,
                *,
                initial_state: torch.Tensor | None = None,
                cu_seqlens: torch.Tensor | None = None,
                cp_mode: str = 'auto') -> FlashQlaResult:
    imports = _imports()
    g_cumsum = imports['chunk_local_cumsum'](
        g,
        chunk_size=64,
        cu_seqlens=cu_seqlens,
    )
    a = imports['kkt_solve'](
        k=k,
        b=beta,
        chunk_size=64,
        cu_seqlens=cu_seqlens,
    )
    run_initial_state = initial_state
    run_cu_seqlens = cu_seqlens
    cp_seq_map = None
    raw_cu_seqlens = None
    cp_enabled = False

    if cp_mode == 'auto':
        run_initial_state, run_cu_seqlens, cp_seq_map, raw_cu_seqlens = imports['intra_card_cp_preprocess'](
            k=k,
            v=v,
            a=a,
            g=g_cumsum,
            b=beta,
            raw_h0=initial_state,
            raw_cu_seqlens=cu_seqlens,
            state_v_first=False,
            enable_fwd_cp_cache=False,
        )
        cp_enabled = cp_seq_map is not None
    elif cp_mode != 'off':
        raise FlashQlaUnsupported(f'unsupported_cp_mode {cp_mode}')

    out, _h, final_state = imports['fused_gdr_fwd'](
        q=q,
        k=k,
        v=v,
        a=a,
        g=g_cumsum,
        b=beta,
        initial_state=run_initial_state,
        output_final_state=True,
        output_h=False,
        output_o=True,
        cu_seqlens=run_cu_seqlens,
        cp_seq_map=cp_seq_map,
        raw_cu_seqlens=raw_cu_seqlens,
        chunk_size=64,
        state_v_first=False,
    )
    return FlashQlaResult(out=out, final_state=final_state, cp_enabled=cp_enabled)


def validate_benchmark_case(run: RunCase, request: BenchmarkRequest) -> None:
    if run.chunk_size != 64:
        raise ValueError('flashqla_requires_chunk64')


def _cp_pattern_segment_tokens(inputs: InputTensors) -> int:
    if inputs.q.shape[0] != 1:
        raise ValueError('cp_not_selected')

    raw_q_offsets = inputs.offsets
    if raw_q_offsets is None:
        raw_q_offsets = torch.tensor(
            (0, inputs.q.shape[1]),
            dtype=torch.int32,
            device=inputs.q.device,
        )
    use_cp, cp_q_offsets, sequence_starts, _, _, _ = _imports()['calc_cp_seqs'](
        raw_q_offsets,
        64,
        inputs.v.shape[2],
    )
    if not use_cp:
        raise ValueError('cp_not_selected')
    if not bool(((sequence_starts[1:] - sequence_starts[:-1]) > 1).any().item()):
        raise ValueError('cp_pattern_requires_multiple_segments')
    return int((cp_q_offsets[1:] - cp_q_offsets[:-1]).max().item())


def make_benchmark_task(run: RunCase, inputs: InputTensors, request: BenchmarkRequest,
                        device: torch.device) -> BenchmarkTask:
    from tests.turbomind.linear_attn.benchmark import base_row
    from tests.turbomind.linear_attn.reference import chunk_gated_delta_rule_fwd

    validate_benchmark_case(run, request)
    if request.cp_pattern != 'auto':
        if request.cp_mode != 'auto':
            raise ValueError('cp_not_selected')
        inputs = apply_cp_pattern(
            run.input,
            inputs,
            request.cp_pattern,
            _cp_pattern_segment_tokens(inputs),
        )
    row = base_row(
        run,
        'flashqla',
        cp_mode=request.cp_mode,
        cp_pattern=request.cp_pattern,
    )

    def execute():
        result = all_forward(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.g,
            inputs.beta,
            initial_state=inputs.h0,
            cu_seqlens=inputs.offsets,
            cp_mode=request.cp_mode,
        )
        row['cp_enabled'] = result.cp_enabled
        return result

    def validate():
        actual = execute()
        expected_o, expected_state = chunk_gated_delta_rule_fwd(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.g,
            inputs.beta,
            initial_state=inputs.h0,
            cu_seqlens=inputs.offsets,
            chunk_size=run.chunk_size,
        )
        if request.validate_outputs:
            torch.testing.assert_close(actual.out, expected_o, rtol=8e-2, atol=8e-2)
            if actual.final_state is not None or expected_state is not None:
                torch.testing.assert_close(actual.final_state, expected_state, rtol=8e-2, atol=8e-2)
        return {
            **diff_metrics('output', actual.out, expected_o),
            **diff_metrics('state', actual.final_state, expected_state),
        }

    return BenchmarkTask(row, execute, validate=validate)
