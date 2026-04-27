# Copyright (c) OpenMMLab. All rights reserved.

import torch
from lmdeploy.turbomind.deploy.parameter import QuantWeightOnly, pack_u4_row
from lmdeploy.turbomind.deploy.source_model.qwen import Qwen3_5ReaderMixin


class _DummyQwen35Reader(Qwen3_5ReaderMixin):

    def __init__(self, params):
        self.params = params
        self.attn_layer_prefix = 'model.layers'

    def transform(self, x, kind: str):
        return x


def _reference_compressed_tensors_dequant(weight_packed: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    x = weight_packed
    unpacked = []
    for _ in range(8):
        unpacked.append((x & 0xF).to(torch.float16))
        x = x >> 4

    weight = torch.stack(unpacked, dim=-1).reshape(weight_packed.shape[0], -1)
    num_groups = weight_scale.shape[1]
    group_size = weight.shape[1] // num_groups
    return ((weight.reshape(weight.shape[0], num_groups, group_size) - 8.0) *
            weight_scale.to(torch.float16).unsqueeze(-1)).reshape(weight.shape[0], -1)


def test_quant_weight_only_synthesizes_compressed_tensor_zero_points_from_scales():
    scales = (
        torch.rand(4, 8, dtype=torch.float32),
        torch.rand(6, 16, dtype=torch.float32),
        torch.rand(4, 8, dtype=torch.float32),
    )
    weight_shapes = tuple((scale.shape[0], scale.shape[1] * 32) for scale in scales)
    weight_packed = tuple(torch.zeros(shape[0], shape[1] // 8, dtype=torch.uint8) for shape in weight_shapes)

    captured = {}

    def _capture_export(_i, tensors, kind, transform, apply_gs=None):
        if kind != 'zeros':
            return
        captured['zeros'] = tuple(transform(tensor) for tensor in tensors)
        captured['apply_gs'] = apply_gs

    values = {
        'weight_packed': weight_packed,
        'weight_scale': scales,
        'weight_shape': weight_shapes,
    }

    QuantWeightOnly(['layer.weight_packed', 'layer.weight_scale'])(_capture_export, values.__getitem__, 0)

    zeros = captured['zeros']
    assert captured['apply_gs'] == ['w2']
    assert tuple(t.shape for t in zeros) == tuple(scale.shape for scale in scales)
    assert all(t.dtype == torch.float16 for t in zeros)
    assert all(torch.all(t == 8) for t in zeros)


def test_compressed_tensors_dequant_matches_reference():
    out_features = 5
    group_size = 32
    num_groups = 3
    unpacked = torch.randint(0, 16, (out_features, group_size * num_groups), dtype=torch.uint8)
    scales = torch.rand(out_features, num_groups, dtype=torch.bfloat16)
    packed = pack_u4_row(unpacked)

    got = Qwen3_5ReaderMixin._compressed_tensors_dequant(packed, scales)
    expected = _reference_compressed_tensors_dequant(packed, scales)

    assert got.dtype == torch.float16
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_qwen35_linear_attn_dequantizes_compressed_tensors_weights():
    out_features = 4
    group_size = 32
    num_groups = 2
    unpacked = torch.randint(0, 16, (out_features, group_size * num_groups), dtype=torch.uint8)
    scales = torch.rand(out_features, num_groups, dtype=torch.bfloat16)
    packed = pack_u4_row(unpacked)

    prefix = 'model.layers.0.linear_attn.in_proj_b'
    reader = _DummyQwen35Reader({
        f'{prefix}.weight_packed': packed,
        f'{prefix}.weight_scale': scales,
    })

    weights = reader.linear_attn(0, 'weight')
    target_idx = reader._LINEAR_ATTN_KEYS.index('in_proj_b')

    assert len(weights) == len(reader._LINEAR_ATTN_KEYS)
    assert all(weight is None for idx, weight in enumerate(weights) if idx != target_idx)
    torch.testing.assert_close(weights[target_idx],
                               _reference_compressed_tensors_dequant(packed, scales),
                               atol=0,
                               rtol=0)
