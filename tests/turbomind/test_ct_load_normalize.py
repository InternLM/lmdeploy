# Copyright (c) Shanghai AI Laboratory. All rights reserved.

import tempfile

import pytest
import torch

from lmdeploy.pytorch.weight_loader.model_weight_loader import (
    finalize_ct_weights,
    new_ct_normalize_state,
    normalize_ct_weights,
)
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


def test_shared_state_carries_module_across_streams():
    """A module split across shard files (or buckets) still converts exactly.

    Each stream is normalized with the same state, as the shard loop and the per-bucket update path do; the pair
    completing in the second stream must produce the identical conversion a single stream would.
    """
    n, k = 64, 256
    packed, scale, zero = ct_repack.quantize_ct(torch.randn(n, k))
    state = new_ct_normalize_state()

    shard_a = list(normalize_ct_weights(iter([
        ('model.layers.0.self_attn.q_proj.weight_packed', packed),
        ('lm_head.weight', torch.randn(8, 4)),
    ]), state))
    # The lone packed tensor stays staged; passthrough keys flow immediately.
    assert [name for name, _ in shard_a] == ['lm_head.weight']

    shard_b = dict(normalize_ct_weights(iter([
        ('model.layers.0.self_attn.q_proj.weight_scale', scale),
        ('model.layers.0.self_attn.q_proj.weight_zero_point', zero),
    ]), state))
    assert set(shard_b) == {
        'model.layers.0.self_attn.q_proj.qweight',
        'model.layers.0.self_attn.q_proj.scales',
        'model.layers.0.self_attn.q_proj.qzeros',
    }
    expected_qweight, expected_scales, expected_qzeros = ct_repack.ct_to_awq(packed, scale, zero)
    assert torch.equal(shard_b['model.layers.0.self_attn.q_proj.qweight'], expected_qweight)
    assert torch.equal(shard_b['model.layers.0.self_attn.q_proj.scales'], expected_scales)
    assert torch.equal(shard_b['model.layers.0.self_attn.q_proj.qzeros'], expected_qzeros)

    # Consumed entirely: finalize is silent.
    finalize_ct_weights(state)


def test_finalize_rejects_leftover_staged_module():
    """Finalize raises when a stream sequence ends with an incomplete
    module."""
    packed, _, _ = ct_repack.quantize_ct(torch.randn(32, 128))
    state = new_ct_normalize_state()
    list(normalize_ct_weights(iter([('x.weight_packed', packed)]), state))
    with pytest.raises(ValueError, match='Incomplete compressed-tensors'):
        finalize_ct_weights(state)


def test_shared_state_stays_authoritative_on_duplicate_across_streams():
    """A second conversion of the same module across streams is rejected."""
    packed, scale, _ = ct_repack.quantize_ct(torch.randn(32, 128))
    state = new_ct_normalize_state()
    list(normalize_ct_weights(iter([
        ('x.weight_packed', packed), ('x.weight_scale', scale),
    ]), state))
    with pytest.raises(ValueError, match='Duplicate compressed-tensors'):
        list(normalize_ct_weights(iter([('x.weight_packed', packed)]), state))


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
