from __future__ import annotations

import pytest
import torch

from lmdeploy.turbomind.builders import ffn as ffn_mod
from lmdeploy.turbomind.linear import Linear
from lmdeploy.turbomind.weight_format import FP8Format, TrivialFormat


def _fp8_linear(inter_size: int) -> Linear:
    k_groups = 2
    out_groups = (inter_size + 127) // 128
    weight = torch.arange(2 * inter_size, dtype=torch.int64)
    weight = weight.remainder(251).to(torch.uint8).reshape(2, inter_size)
    scales = torch.arange(k_groups * out_groups, dtype=torch.float32)
    scales = scales.reshape(k_groups, out_groups).to(torch.bfloat16)
    bias = torch.arange(inter_size, dtype=torch.float32).to(torch.bfloat16)
    return Linear(
        tensors={'weight': weight, 'scales': scales, 'bias': bias},
        weight_format=FP8Format())


def _bf16_linear(inter_size: int) -> Linear:
    weight = torch.arange(2 * inter_size, dtype=torch.float32)
    weight = weight.reshape(2, inter_size).to(torch.bfloat16)
    bias = torch.arange(inter_size, dtype=torch.float32).to(torch.bfloat16)
    return Linear(
        tensors={'weight': weight, 'bias': bias},
        weight_format=TrivialFormat())


def _offset_linear(linear: Linear, offset: int) -> Linear:
    tensors = {
        kind: tensor + offset
        for kind, tensor in linear.tensors.items()
    }
    return Linear(tensors=tensors, weight_format=linear.weight_format)


def test_block_pack_self_adapts_weight_scale_and_bias_groups():
    w1 = _fp8_linear(512)
    w3 = _offset_linear(w1, 17)

    fused = ffn_mod._block_pack_w1w3(w1, w3, groups=4)

    assert fused.tensors['weight'].shape == (2, 1024)
    assert fused.tensors['scales'].shape == (2, 8)
    assert fused.tensors['bias'].shape == (1024,)

    for group in range(4):
        src_weight = slice(group * 128, (group + 1) * 128)
        dst_gate = slice(group * 256, group * 256 + 128)
        dst_up = slice(group * 256 + 128, (group + 1) * 256)
        torch.testing.assert_close(
            fused.tensors['weight'][:, dst_gate],
            w1.tensors['weight'][:, src_weight])
        torch.testing.assert_close(
            fused.tensors['weight'][:, dst_up],
            w3.tensors['weight'][:, src_weight])
        torch.testing.assert_close(
            fused.tensors['bias'][dst_gate],
            w1.tensors['bias'][src_weight])
        torch.testing.assert_close(
            fused.tensors['bias'][dst_up],
            w3.tensors['bias'][src_weight])
        torch.testing.assert_close(
            fused.tensors['scales'][:, group * 2],
            w1.tensors['scales'][:, group])
        torch.testing.assert_close(
            fused.tensors['scales'][:, group * 2 + 1],
            w3.tensors['scales'][:, group])


def test_sm90_fp8_fuse_w1w3_packs_scale_groups(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (9, 0))
    w1 = _fp8_linear(512)
    w3 = _offset_linear(w1, 17)

    fused, fused_silu = ffn_mod.fuse_w1w3(
        w1, w3, tp=1, act_type='silu', is_moe=True)

    assert fused_silu
    assert fused is not None
    assert fused.tensors['weight'].shape == (2, 1024)
    assert fused.tensors['scales'].shape == (2, 8)


def test_can_fuse_checks_tp1_and_local_tp_alignment():
    assert ffn_mod._can_fuse_w1w3(
        _fp8_linear(512), tp=1, pack_block=128)
    assert not ffn_mod._can_fuse_w1w3(
        _fp8_linear(500), tp=1, pack_block=128)
    assert ffn_mod._can_fuse_w1w3(
        _fp8_linear(512), tp=2, pack_block=128)
    assert not ffn_mod._can_fuse_w1w3(
        _fp8_linear(384), tp=2, pack_block=128)
    assert ffn_mod._can_fuse_w1w3(
        _bf16_linear(192), tp=1, pack_block=64)
    assert not ffn_mod._can_fuse_w1w3(
        _bf16_linear(192), tp=1, pack_block=128)


@pytest.mark.parametrize('is_moe', [False, True])
def test_sm90_bf16_uses_64_wide_block_packed_fused_silu(
        monkeypatch, is_moe):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (9, 0))
    w1 = _bf16_linear(256)
    w3 = _offset_linear(w1, 17)

    fused, fused_silu = ffn_mod.fuse_w1w3(
        w1, w3, tp=1, act_type='silu', is_moe=is_moe)

    assert fused_silu
    assert fused is not None
    for group in range(4):
        src = slice(group * 64, (group + 1) * 64)
        dst_gate = slice(group * 128, group * 128 + 64)
        dst_up = slice(group * 128 + 64, (group + 1) * 128)
        torch.testing.assert_close(
            fused.tensors['weight'][:, dst_gate],
            w1.tensors['weight'][:, src])
        torch.testing.assert_close(
            fused.tensors['weight'][:, dst_up],
            w3.tensors['weight'][:, src])
        torch.testing.assert_close(
            fused.tensors['bias'][dst_gate], w1.tensors['bias'][src])
        torch.testing.assert_close(
            fused.tensors['bias'][dst_up], w3.tensors['bias'][src])


def test_non_sm90_dense_bf16_keeps_chunk_layout(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (8, 0))
    w1 = _bf16_linear(128)
    w3 = _offset_linear(w1, 17)

    fused, fused_silu = ffn_mod.fuse_w1w3(
        w1, w3, tp=1, act_type='silu', is_moe=False)

    assert not fused_silu
    assert fused is not None
    torch.testing.assert_close(
        fused.tensors['weight'][:, :128], w1.tensors['weight'])
    torch.testing.assert_close(
        fused.tensors['weight'][:, 128:], w3.tensors['weight'])


def test_non_sm90_fp8_preserves_block_pack_layout(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (10, 0))
    w1 = _fp8_linear(512)
    w3 = _offset_linear(w1, 17)

    fused, fused_silu = ffn_mod.fuse_w1w3(
        w1, w3, tp=1, act_type='silu', is_moe=False)

    assert fused_silu
    assert fused is not None
    assert fused.tensors['weight'].shape == (2, 1024)
    assert fused.tensors['scales'].shape == (2, 8)
    torch.testing.assert_close(
        fused.tensors['weight'][:, :128], w1.tensors['weight'][:, :128])
    torch.testing.assert_close(
        fused.tensors['weight'][:, 128:256], w3.tensors['weight'][:, :128])
