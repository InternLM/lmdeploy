# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Loader suffix-dispatch regression for the gpt-oss mxfp4 expert tensors.

The mxfp4 checkpoint names ``gate_up_proj_blocks`` / ``down_proj_blocks``,
which contain the plain-projection substrings. The suffix checks must run
first (blocks/scales route to the mxfp4 ``.blocks`` / ``.scales`` module
params through the P2 ``flatten(2)``); the plain bf16 ``down_proj`` path
must still transpose. Regression pins the crash ordering where
``'gate_up_proj' in name`` would catch ``'gate_up_proj_blocks'``.
"""

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.models.gpt_oss import GptOssForCausalLM

# The loader path moves loaded tensors with ``.cuda()``.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason='gpt-oss mxfp4 loader test requires CUDA')


class _Recorder:
    """Collect load_weight calls instead of touching real params."""

    def __init__(self):
        self.calls: list[tuple[str, tuple[int, ...], int, str]] = []

    def __call__(self, param, tensor, expert_id=0, shard_id=None):
        self.calls.append((param.param_name, tuple(tensor.shape), expert_id, shard_id))


@pytest.fixture
def model(monkeypatch):
    """A bare model object with only the loader path wired up."""
    model = object.__new__(GptOssForCausalLM)
    model.config = PretrainedConfig(num_local_experts=2)
    recorder = _Recorder()
    monkeypatch.setattr('lmdeploy.pytorch.models.gpt_oss.load_weight', recorder)
    model._recorder = recorder
    return model


def _param(name, shape):
    param = torch.nn.Parameter(torch.empty(shape, dtype=torch.uint8), requires_grad=False)
    param.param_name = name
    return param


def test_blocks_route_to_mxfp4_params_before_plain_suffix(model):
    """`gate_up_proj_blocks` must map to `gate_up.blocks` via flatten(2) —
    never to the plain `gate_up_proj` parameter."""
    num_experts, fused_out, k_half = 2, 8, 16  # fused out=2*ffn, k_half=in//2
    params = {
        'layers.0.experts.experts.gate_up.blocks': _param('layers.0.experts.experts.gate_up.blocks',
                                                          (num_experts, fused_out, k_half)),
        'layers.0.experts.experts.gate_up.weight': _param('layers.0.experts.experts.gate_up.weight', (99,)),
    }
    raw = torch.randint(0, 255, (num_experts, fused_out, k_half // 16, 16), dtype=torch.uint8)
    model._load_weight_experts_gate_up('layers.0.experts.gate_up_proj_blocks', raw, params)
    calls = model._recorder.calls
    assert calls, 'blocks weight must reach load_weight'
    assert all(name.endswith('gate_up.blocks') for name, _, _, _ in calls), calls
    # P2: the 4D grouped byte layout [E, out, in//32, 16] was flattened to
    # [E, out, in//2] before the deinterleave slices.
    for _, shape, _, shard in calls:
        assert shape == (fused_out // 2, k_half)
        assert shard in ('gate', 'up')
    assert not any('gate_up.weight' in name for name, _, _, _ in calls)


def test_down_blocks_route_and_scales(model):
    """`down_proj_blocks` flattens to the flat byte layout; `down_proj_scales`
    maps to `down.scales` untouched."""
    num_experts, out, k_half = 2, 8, 16
    params = {
        'layers.0.experts.experts.down.blocks': _param('layers.0.experts.experts.down.blocks',
                                                       (num_experts, out, k_half)),
        'layers.0.experts.experts.down.scales': _param('layers.0.experts.experts.down.scales',
                                                       (num_experts, out, k_half // 16)),
    }
    raw = torch.randint(0, 255, (num_experts, out, k_half // 16, 16), dtype=torch.uint8)
    model._load_weight_experts_down('layers.0.experts.down_proj_blocks', raw, params)
    scales = torch.randint(0, 255, (num_experts, out, k_half // 16), dtype=torch.uint8)
    model._load_weight_experts_down('layers.0.experts.down_proj_scales', scales, params)
    calls = model._recorder.calls
    assert len(calls) == 2 * num_experts
    blocks_calls = [c for c in calls if c[3] == 'down' and c[2] >= 0]
    assert all(name.endswith('down.blocks') or name.endswith('down.scales') for name, _, _, _ in calls)
    assert blocks_calls[0][1] == (out, k_half), 'blocks must arrive flattened'


def test_plain_down_proj_still_transposes(model):
    """The non-quantized path keeps its transpose; the mxfp4 suffix must not
    swallow it."""
    num_experts, out, in_features = 2, 6, 8
    params = {
        'layers.0.experts.experts.down.weight': _param('layers.0.experts.experts.down.weight',
                                                       (num_experts, in_features, out)),
    }
    raw = torch.randn(num_experts, out, in_features)
    model._load_weight_experts_down('layers.0.experts.down_proj', raw, params)
    calls = model._recorder.calls
    assert len(calls) == num_experts
    for name, shape, _, shard in calls:
        assert name == 'layers.0.experts.experts.down.weight'
        assert shape == (in_features, out), 'plain down_proj must be transposed'
        assert shard == 'down'
