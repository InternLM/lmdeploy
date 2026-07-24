from __future__ import annotations

import os
from functools import cache

os.environ['FLA_INTRACARD_CP'] = '0'

import torch

from .benchmark import (
    BenchmarkRequest,
    BenchmarkTask,
    diff_metrics,
)
from .cases import InputTensors, RunCase

CHUNK_SIZE = 64
SCALE = 128 ** -0.5


@cache
def _imports():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return {
        'chunk_gated_delta_rule': chunk_gated_delta_rule,
    }


def is_available() -> bool:
    try:
        _imports()
    except ImportError:
        return False
    return True


def all_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    imports = _imports()
    return imports['chunk_gated_delta_rule'](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
    )


def validate_benchmark_case(run: RunCase, request: BenchmarkRequest) -> None:
    if run.chunk_size != 64:
        raise ValueError('fla_requires_chunk64')


def make_benchmark_task(run: RunCase, inputs: InputTensors, request: BenchmarkRequest,
                        device: torch.device) -> BenchmarkTask:
    from .benchmark import base_row
    from .reference import chunk_gated_delta_rule_fwd

    validate_benchmark_case(run, request)

    def execute():
        return all_forward(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.g,
            inputs.beta,
            initial_state=inputs.h0,
            cu_seqlens=inputs.offsets,
        )

    def validate():
        actual_o, actual_state = execute()
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
            torch.testing.assert_close(actual_o, expected_o, rtol=8e-2, atol=8e-2)
            if actual_state is not None or expected_state is not None:
                torch.testing.assert_close(actual_state, expected_state, rtol=8e-2, atol=8e-2)
        return {
            **diff_metrics('output', actual_o, expected_o),
            **diff_metrics('state', actual_state, expected_state),
        }

    return BenchmarkTask(
        base_row(
            run,
            'fla',
            cp_level=request.cp_level,
            cp_pattern=request.cp_pattern,
            cp_enabled=False,
        ),
        execute,
        validate=validate,
    )
