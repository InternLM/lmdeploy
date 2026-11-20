# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Engine-level golden gate for the W4A16 GEMM routes.

One public AWQ checkpoint, two engine runs over the same fixed prompt set:
the AWQ Triton route is the reference, the TurboMind route the candidate.
Gates: top-1 agreement, greedy agreement, truncated KL on the top-20
first-token logits.
"""

import os

import pytest
import torch

HF_AWQ_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct-AWQ'

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


class TestGoldenGate:
    """The TurboMind route tracks the AWQ Triton route on real weights."""

    def test_tm_route_matches_triton_reference(self, tmp_path):
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from w4a16_golden import PROMPTS, dump_reference, gate_against, run_engine

        reference = dump_reference(HF_AWQ_MODEL, str(tmp_path / 'awq_ref.json'))
        results = run_engine(HF_AWQ_MODEL, PROMPTS, backend='turbomind')
        gate = gate_against(reference, results)
        assert gate['top1_agreement'] >= 0.97, gate
        # Greedy compares whole 16-token continuations, so one tie flip at the
        # flat logits of a 0.5B model cascades into a mismatched sequence;
        # top1 agreement and the KL gate are the numeric anchors here.
        assert gate['greedy_agreement'] >= 0.90, gate
        assert gate['worst_truncated_kl'] <= gate['kl_threshold'], gate
