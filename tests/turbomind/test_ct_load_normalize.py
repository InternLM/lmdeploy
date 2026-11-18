# Copyright (c) Shanghai AI Laboratory. All rights reserved.

import tempfile

import pytest
import torch

from lmdeploy.pytorch.weight_loader.model_weight_loader import normalize_ct_weights
from lmdeploy.turbomind import ct_repack
from tests.turbomind.tiny_ct import gen_tiny_ct, load_shard, shard_key


def test_passthrough_without_ct_keys():
    """Checkpoints with canonical names pass through byte for byte."""
    tensors = [
        ('model.layers.0.self_attn.q_proj.qweight', torch.zeros(4, 2, dtype=torch.int32)),
        ('model.layers.0.self_attn.q_proj.scales', torch.ones(1, 32)),
        ('lm_head.weight', torch.randn(8, 4)),
    ]
    out = list(normalize_ct_weights(iter(tensors)))
    assert out == tensors


def test_ct_module_maps_to_canonical_awq_names():
    """One ct module converts and keeps the module prefix."""
    n, k = 64, 256
    packed, scale, zero = ct_repack.quantize_ct(torch.randn(n, k))
    tensors = [
        ('model.layers.3.mlp.gate_proj.weight_scale', scale),
        ('model.layers.3.mlp.gate_proj.weight_packed', packed),
        ('model.layers.3.mlp.gate_proj.weight_zero_point', zero),
        ('model.layers.3.mlp.gate_proj.bias', torch.randn(n)),
    ]
    out = dict(normalize_ct_weights(iter(tensors)))
    assert set(out) == {
        'model.layers.3.mlp.gate_proj.qweight',
        'model.layers.3.mlp.gate_proj.scales',
        'model.layers.3.mlp.gate_proj.qzeros',
        'model.layers.3.mlp.gate_proj.bias',
    }
    expected_qweight, expected_scales, expected_qzeros = ct_repack.ct_to_awq(packed, scale, zero)
    assert torch.equal(out['model.layers.3.mlp.gate_proj.qweight'], expected_qweight)
    assert torch.equal(out['model.layers.3.mlp.gate_proj.scales'], expected_scales)
    assert torch.equal(out['model.layers.3.mlp.gate_proj.qzeros'], expected_qzeros)


def test_ct_shard_tolerates_staging_order_and_missing_zero_point():
    """The staged pair may arrive in any order; symmetric checkpoints may omit
    zero points."""
    packed, scale, _ = ct_repack.quantize_ct(torch.randn(32, 128))
    orders = [
        [('x.weight_packed', packed), ('x.weight_scale', scale)],
        [('x.weight_scale', scale), ('x.weight_packed', packed)],
    ]
    for tensors in orders:
        out = dict(normalize_ct_weights(iter(tensors)))
        assert set(out) == {'x.qweight', 'x.scales', 'x.qzeros'}
        assert out['x.qzeros'].dtype == torch.int32


def test_incomplete_ct_shard_raises():
    """A missing scale leaves the staging incomplete and must raise."""
    packed, scale, _ = ct_repack.quantize_ct(torch.randn(32, 128))
    with pytest.raises(ValueError, match='Incomplete compressed-tensors'):
        list(normalize_ct_weights(iter([('x.weight_packed', packed)])))
    with pytest.raises(ValueError, match='Incomplete compressed-tensors'):
        list(normalize_ct_weights(iter([('x.weight_scale', scale)])))


def test_gen_tiny_ct_shards_normalize_bit_exact():
    """Every shard of the real generator converts exactly like
    convert_shard."""
    with tempfile.TemporaryDirectory() as tmp:
        gen_tiny_ct(tmp, preset='llama3_8b', num_layers=1, only=('qkv', 'o'))
        for key in (shard_key(0, name) for name in ('qkv', 'o')):
            shard = load_shard(tmp, key)
            tensors = [(f'{key}.{suffix}', shard[suffix])
                       for suffix in ('weight_packed', 'weight_scale', 'weight_zero_point', 'bias')
                       if suffix in shard]
            out = dict(normalize_ct_weights(iter(tensors)))
            converted, _ = ct_repack.convert_shard(shard)
            assert set(out) == {f'{key}.{name}' for name in converted}
            for name, tensor in converted.items():
                assert torch.equal(out[f'{key}.{name}'], tensor)
