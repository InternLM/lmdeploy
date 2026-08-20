# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from lmdeploy.pytorch.backends.mimo_swa import MiMoSWAAttentionMetadata
from lmdeploy.pytorch.configurations.mimo_v2_flash import MiMoV2FlashModelConfigBuilder
from lmdeploy.pytorch.models.mimo_v2_flash import (
    MiMoV2Attention,
    MiMoV2MLP,
    _all_reduce_mimo,
    _dequantize_blocked_fp8,
    _reduce_tp_output,
)


def _make_mimo_hf_config():
    """Build a minimal target-model config with the official layer pattern."""
    full_layers = {0, 5, 11, 17, 23, 29, 35, 41, 47}
    pattern = [0 if layer_id in full_layers else 1 for layer_id in range(48)]
    return SimpleNamespace(
        model_type='mimo_v2_flash',
        architectures=['MiMoV2FlashForCausalLM'],
        hidden_size=4096,
        num_hidden_layers=48,
        hybrid_layer_pattern=pattern,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=192,
        v_head_dim=128,
        swa_num_attention_heads=64,
        swa_num_key_value_heads=8,
        swa_head_dim=192,
        swa_v_head_dim=128,
        sliding_window=128,
        sliding_window_size=128,
        vocab_size=152576,
        bos_token_id=151643,
        eos_token_id=151643,
        routed_scaling_factor=None,
    )


@pytest.mark.parametrize(('is_swa', 'expected'), [(False, True), (True, False)])
def test_mimo_enables_fa3_only_for_full_attention(is_swa, expected):
    """Full Attention may select FA3 while SWA keeps its custom path."""
    config = _make_mimo_hf_config()
    config.partial_rotary_factor = 0.5
    config.attention_bias = False
    config.quantization_config = None
    captured = {}

    def fake_attention(*args, **kwargs):
        captured.update(kwargs)
        return nn.Identity()

    with patch('lmdeploy.pytorch.models.mimo_v2_flash.build_qkv_proj',
               return_value=nn.Identity()), patch(
                   'lmdeploy.pytorch.models.mimo_v2_flash.build_o_proj',
                   return_value=nn.Identity()), patch(
                       'lmdeploy.pytorch.models.mimo_v2_flash.Attention',
                       side_effect=fake_attention):
        MiMoV2Attention(config, is_swa=is_swa)

    assert captured['enable_fa3'] is expected


def test_mimo_target_uses_full_blocks_and_sequence_state_ring():
    """Target inference uses paged Full KV and fixed-size SWA state caches."""
    hf_config = _make_mimo_hf_config()
    config = MiMoV2FlashModelConfigBuilder.build(hf_config, tp=4)
    config.hf_config = hf_config
    config.post_build_func(config, 64)

    assert config.model_paradigm == 'ar'
    assert [spec.name for spec in config.block_cache_specs] == ['mimo_full_k', 'mimo_full_v']
    assert [spec.name for spec in config.state_cache_specs] == ['mimo_swa_ring_k', 'mimo_swa_ring_v']
    assert all(len(spec.layer_ids) == 39 for spec in config.state_cache_specs)
    assert config.state_cache_specs[0].shape == (128, 2, 192)
    assert config.state_cache_specs[1].shape == (128, 2, 128)


@pytest.mark.parametrize(('is_draft_model', 'spec_method'), [(True, None), (False, 'mimo_mtp')])
def test_mimo_target_pr_rejects_speculative_modes(is_draft_model, spec_method):
    """Keep speculative decoding outside the target-only integration."""
    with pytest.raises(ValueError, match='not available yet'):
        MiMoV2FlashModelConfigBuilder.build(
            _make_mimo_hf_config(),
            tp=4,
            is_draft_model=is_draft_model,
            spec_method=spec_method,
        )


def test_mimo_swa_metadata_enforces_total_window_and_sequence_slots():
    """SWA metadata retains 127 history tokens plus each current token."""
    metadata = SimpleNamespace(
        q_seqlens=torch.tensor([2, 1]),
        kv_seqlens=torch.tensor([130, 1]),
        max_q_seqlen=2,
    )
    context = SimpleNamespace(input_ids=torch.zeros((1, 3), dtype=torch.long), max_q_seqlen=2)

    result = MiMoSWAAttentionMetadata.from_step_context(
        metadata,
        context,
        state_slots=torch.tensor([3, -1]),
        num_state_slots=4,
        window_size=128,
    )

    assert result.start_positions.tolist() == [128, 0]
    assert result.history_lens.tolist() == [127, 0]
    assert result.kv_seqlens.tolist() == [129, 1]
    assert result.cu_q_seqlens.tolist() == [0, 2, 3]
    assert result.cu_kv_seqlens.tolist() == [0, 129, 130]
    assert result.max_kv_seqlen == 129


def test_mimo_blocked_fp8_dequantization_uses_serialized_scale_grid():
    """Blocked FP8 fallback derives tiles from serialized scale shapes."""
    weight = torch.ones((4, 4))
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = _dequantize_blocked_fp8(weight, scale, torch.float32)

    assert torch.equal(
        result,
        torch.tensor([
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0],
        ]),
    )


def test_mimo_blocked_fp8_dequantization_rejects_misaligned_scale_grid():
    """Reject scale grids that cannot tile the serialized weight."""
    with pytest.raises(ValueError, match='Invalid blocked-FP8'):
        _dequantize_blocked_fp8(torch.ones((5, 4)), torch.ones((2, 2)), torch.float32)


def test_mimo_dense_mlp_keeps_framework_dp_tp_reduction():
    """Dense down projection must keep framework DP+TP reduction semantics."""
    config = SimpleNamespace(hidden_size=8, intermediate_size=16, quantization_config=None)
    captured = {}
    gate_up = nn.Identity()
    down = nn.Identity()

    def fake_down(*args, **kwargs):
        captured.update(kwargs)
        return down

    with patch('lmdeploy.pytorch.models.mimo_v2_flash.build_gateup_linear',
               return_value=gate_up), patch(
                   'lmdeploy.pytorch.models.mimo_v2_flash.build_down_linear',
                   side_effect=fake_down), patch(
                       'lmdeploy.pytorch.models.mimo_v2_flash.SiluAndMul',
                       return_value=nn.Identity()):
        mlp = MiMoV2MLP(config)

    value = torch.randn(2, 3, 8)
    assert captured['all_reduce'] is True
    assert mlp(value) is value


def test_mimo_tp_reduce_helper_skips_tp1():
    """A row-parallel output must skip the collective at TP=1."""
    linear = SimpleNamespace(tp=1)
    output = torch.randn(2, 3)
    with patch('lmdeploy.pytorch.models.mimo_v2_flash._all_reduce_mimo') as reduce_mock:
        assert _reduce_tp_output(linear, output) is output
    reduce_mock.assert_not_called()


def test_mimo_tp_reduce_helper_uses_explicit_group():
    """A row-parallel output must reduce over its own TP group."""
    group = object()
    linear = SimpleNamespace(tp=4, tp_group=group)
    output = torch.randn(2, 3)
    with patch('lmdeploy.pytorch.models.mimo_v2_flash._all_reduce_mimo',
               return_value=output) as reduce_mock:
        assert _reduce_tp_output(linear, output) is output
    reduce_mock.assert_called_once_with(output, group=group)


def test_mimo_all_reduce_uses_requested_group():
    """MiMo collectives must not silently fall back to a global group."""
    group = object()
    output = torch.randn(2, 3)
    with patch('lmdeploy.pytorch.models.mimo_v2_flash.dist.all_reduce') as all_reduce:
        assert _all_reduce_mimo(output, group=group) is output
    all_reduce.assert_called_once_with(output, group=group)
