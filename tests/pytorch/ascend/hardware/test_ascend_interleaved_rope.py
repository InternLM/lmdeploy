# ruff: noqa: E402
# Copyright (c) 2026, DeepLink. All rights reserved.

import pytest
import torch

torch_npu = pytest.importorskip('torch_npu')

if not torch.npu.is_available():
    pytest.skip('Ascend NPU is required', allow_module_level=True)

from lmdeploy.pytorch.kernels.ascend.ops import apply_rotary_pos_emb_interleaved


def _reference(x, cos, sin):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((x_even * cos - x_odd * sin, x_odd * cos + x_even * sin), dim=-1).flatten(-2)


@pytest.mark.parametrize('num_heads', [1, 3, 32])
def test_interleaved_rope_matches_adjacent_pair_reference(num_heads):
    torch.manual_seed(20260821)
    num_tokens = 7
    head_dim = 64
    x = torch.randn(num_tokens, num_heads, 1, head_dim, dtype=torch.bfloat16, device='npu')
    cos = torch.randn(num_tokens, 1, 1, head_dim // 2, dtype=torch.bfloat16, device='npu')
    sin = torch.randn_like(cos)
    cos_native = torch.cat((cos, cos), dim=-1)
    sin_native = torch.cat((sin, sin), dim=-1)

    expected = _reference(x, cos, sin)
    actual_native = apply_rotary_pos_emb_interleaved(x, cos_native, sin_native)
    actual_adjacent = apply_rotary_pos_emb_interleaved(x, cos_native, sin_native, return_native_layout=False)
    expected_native = torch.cat((expected[..., 0::2], expected[..., 1::2]), dim=-1)

    for actual in (actual_native, actual_adjacent):
        assert actual.shape == x.shape
        assert actual.dtype == x.dtype
    torch.testing.assert_close(actual_native.float(), expected_native.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_adjacent.float(), expected.float(), rtol=2e-2, atol=2e-2)
