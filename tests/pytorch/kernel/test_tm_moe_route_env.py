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


def test_mxfp4_route_off_raises_offline_valueerror(monkeypatch):
    """Mxfp4=off: no fallback lane exists, the route demands offline
    dequantization with a ValueError (not the triton fallback the W4A16 classes
    take)."""
    import lmdeploy.pytorch.backends.cuda.tm_moe_modules as tm_mod
    import lmdeploy.pytorch.envs as envs

    monkeypatch.setattr(envs, 'tm_kernel_state', lambda cls: 'off' if cls == 'mxfp4' else 'auto')
    with pytest.raises(ValueError, match='off'):
        tm_mod.tm_mxfp4_route(torch.bfloat16, hidden_dim=4096, ffn_dim=2880)


def test_mxfp4_route_on_is_strict(monkeypatch):
    """Mxfp4=on: a probe rejection raises instead of falling back; a passing
    probe returns None (route served)."""
    import lmdeploy.pytorch.backends.cuda.tm_moe_modules as tm_mod
    import lmdeploy.pytorch.envs as envs

    monkeypatch.setattr(envs, 'tm_kernel_state', lambda cls: 'on' if cls == 'mxfp4' else 'auto')
    monkeypatch.setattr(tm_mod, 'turbomind_mxfp4_support_reason', lambda *a, **k: 'no family on this device')
    with pytest.raises(RuntimeError, match='no fallback lane'):
        tm_mod.tm_mxfp4_route(torch.bfloat16, hidden_dim=4096, ffn_dim=2880)

    monkeypatch.setattr(tm_mod, 'turbomind_mxfp4_support_reason', lambda *a, **k: None)
    assert tm_mod.tm_mxfp4_route(torch.bfloat16, hidden_dim=4096, ffn_dim=2880) is None


def test_mxfp4_route_auto_probes(monkeypatch):
    """Mxfp4=auto: same strictness, the probe decides."""
    import lmdeploy.pytorch.backends.cuda.tm_moe_modules as tm_mod
    import lmdeploy.pytorch.envs as envs

    monkeypatch.setattr(envs, 'tm_kernel_state', lambda cls: 'auto')
    monkeypatch.setattr(tm_mod, 'turbomind_mxfp4_support_reason',
                        lambda *a, **k: 'no grouped GEMM family accepts mxfp4 dtype=torch.float16')
    with pytest.raises(RuntimeError, match='dtype=torch.float16'):
        tm_mod.tm_mxfp4_route(torch.float16, hidden_dim=4096, ffn_dim=2880)
    monkeypatch.setattr(tm_mod, 'turbomind_mxfp4_support_reason', lambda *a, **k: None)
    assert tm_mod.tm_mxfp4_route(torch.bfloat16, hidden_dim=4096, ffn_dim=2880) is None


def test_other_kernel_classes_unaffected_by_mxfp4_off(monkeypatch):
    """Flipping the mxfp4 class must not disturb the other classes' states."""
    import lmdeploy.pytorch.envs as envs

    states = {'mxfp4': 'off', 'w4a16': 'on', 'moe_grouped': 'auto', 'nvfp4': 'off', 'ct_fp8': 'on'}
    monkeypatch.setattr(envs, 'tm_kernel_state', lambda cls: states[cls])
    # tm_moe_route reads the legacy w4a16 module attribute directly; the
    # assertion here is on the class map: the mxfp4 entry cannot shadow the
    # legacy-env-backed classes.
    assert 'w4a16' in envs._TM_KERNEL_CLASS_LEGACY_ENVS
    assert envs._TM_KERNEL_CLASS_LEGACY_ENVS['w4a16'] == 'LMDEPLOY_W4A16_GEMM_BACKEND'
    assert envs._TM_KERNEL_CLASS_LEGACY_ENVS['mxfp4'] is None
