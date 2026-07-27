from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch

from .benchmark import (
    BenchmarkRequest,
    BenchmarkTask,
    apply_cp_pattern,
    diff_metrics,
)
from .cases import InputTensors, RunCase


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
    from flash_qla.ops.gated_delta_rule.chunk.cp_context import intra_card_cp_preprocess
    from flash_qla.ops.utils import chunk_local_cumsum

    return {
        'chunk_local_cumsum': chunk_local_cumsum,
        'kkt_solve': qla_chunk.kkt_solve,
        'fused_gdr_fwd': qla_chunk.fused_gdr_fwd,
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
                cp_level: str = 'all') -> FlashQlaResult:
    imports = _imports()
    a = imports['kkt_solve'](
        k=k,
        b=beta,
        cu_seqlens=cu_seqlens,
    )
    g_cumsum = imports['chunk_local_cumsum'](
        g,
        cu_seqlens=cu_seqlens,
    )
    run_initial_state = initial_state
    run_cu_seqlens = cu_seqlens
    cp_seq_map = None
    raw_cu_seqlens = None
    cp_enabled = False

    if cp_level == 'all':
        run_initial_state, run_cu_seqlens, cp_seq_map, raw_cu_seqlens = imports[
            'intra_card_cp_preprocess'
        ](
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
    elif cp_level != 'off':
        raise FlashQlaUnsupported(f'unsupported_cp_level {cp_level}')

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
        state_v_first=False,
    )
    return FlashQlaResult(out=out, final_state=final_state, cp_enabled=cp_enabled)


def _cp_pattern_segment_tokens(inputs: InputTensors) -> int:
    imports = _imports()
    a = imports['kkt_solve'](
        k=inputs.k,
        b=inputs.beta,
        cu_seqlens=inputs.offsets,
    )
    g_cumsum = imports['chunk_local_cumsum'](
        inputs.g,
        cu_seqlens=inputs.offsets,
    )
    _, cp_q_offsets, cp_seq_map, _ = imports['intra_card_cp_preprocess'](
        k=inputs.k,
        v=inputs.v,
        a=a,
        g=g_cumsum,
        b=inputs.beta,
        raw_h0=inputs.h0,
        raw_cu_seqlens=inputs.offsets,
        state_v_first=False,
        enable_fwd_cp_cache=False,
    )
    if cp_seq_map is None:
        raise ValueError('cp_not_selected')
    if cp_seq_map.unique().numel() == cp_seq_map.numel():
        raise ValueError('cp_pattern_requires_multiple_segments')
    return int((cp_q_offsets[1:] - cp_q_offsets[:-1]).max().item())


def make_benchmark_task(run: RunCase, inputs: InputTensors, request: BenchmarkRequest,
                        device: torch.device) -> BenchmarkTask:
    from .benchmark import base_row
    from .reference import chunk_gated_delta_rule_fwd

    if request.cp_pattern != 'auto':
        if request.cp_level != 'all':
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
        cp_level=request.cp_level,
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
            cp_level=request.cp_level,
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
