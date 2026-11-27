# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Env three-state routing matrix for the TurboMind grouped W4A16 MoE.

Unit cases without a CUDA dependency. The env state and the cheap probe
checks (w_bit, group size) decide before any device touch, so the matrix
runs everywhere. The device-dependent probe outcomes (bf16 rejection,
alignment rejections) live in ``test_tm_moe_w4a16.py`` under the GPU mark.
"""

import pytest
import torch

from lmdeploy.pytorch.backends.cuda.tm_moe_modules import tm_moe_route


def test_route_env_three_states(monkeypatch):
    """The MoE route mirrors the dense env semantics per state.

    ``auto`` probes and reports, ``turbomind`` raises on an unsupported
    geometry, any other value routes straight to the triton fused_moe path.
    The unsupported geometry keeps the whole check CPU-legal: the probe
    rejects w_bit before any CUDA touch.
    """
    import lmdeploy.pytorch.envs as envs

    monkeypatch.setattr(envs, 'w4a16_gemm_backend', 'off')
    reason = tm_moe_route(group_size=128, w_bit=4, dtype=torch.float16, hidden_dim=4096, ffn_dim=2048)
    assert reason is not None and 'off' in reason

    monkeypatch.setattr(envs, 'w4a16_gemm_backend', 'auto')
    reason = tm_moe_route(group_size=128, w_bit=3, dtype=torch.float16, hidden_dim=4096, ffn_dim=2048)
    assert reason is not None and 'w_bit' in reason

    monkeypatch.setattr(envs, 'w4a16_gemm_backend', 'turbomind')
    with pytest.raises(RuntimeError, match='TurboMind grouped MoE'):
        tm_moe_route(group_size=128, w_bit=3, dtype=torch.float16, hidden_dim=4096, ffn_dim=2048)


def test_route_env_rejects_bad_group_size(monkeypatch):
    """A non-positive group size declines before the registry walk."""
    import lmdeploy.pytorch.envs as envs

    monkeypatch.setattr(envs, 'w4a16_gemm_backend', 'auto')
    reason = tm_moe_route(group_size=0, w_bit=4, dtype=torch.float16, hidden_dim=4096, ffn_dim=2048)
    assert reason is not None and 'group_size' in reason
