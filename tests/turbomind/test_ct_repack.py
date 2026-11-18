# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""S5 bit-level roundtrip tests for ct -> AutoAWQ conversion.

Ticket contract items:

- the conversion roundtrips with exactly zero error
- the symmetric zero-point pad convention holds
- the AWQ nibble-order trap stays guarded

The trap: ``AWQ_NIBBLE_ORDER`` is not an involution. Indexing the table and
searching it give two different packings. Both look valid. The tests run on
CPU from generated random data. The tests download nothing. The tests need
no CUDA device.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from lmdeploy.turbomind import ct_repack
from lmdeploy.turbomind.weight_format import _unpack_awq_gemm

from . import tiny_ct

# Tiny cuts of the presets for the suite. The glm52 cut keeps the MLA
# oddballs (N=576/128/32) and the widest K (36864) at a bounded runtime.
SUITE_CUTS = {
    'llama3_8b': ('gate_up', 'down', 'qkv', 'o'),
    'glm4_9b': ('gate_up', 'down', 'qkv', 'o'),
    'glm52': ('down', 'shared_down', 'mla_q_a', 'mla_q_b', 'mla_kv_a', 'mla_o', 'index_k', 'index_weight'),
}


def _gen(tmp_path, preset, layers=1, seed=9):
    out = os.path.join(str(tmp_path), f'ct_{preset}')
    tiny_ct.gen_tiny_ct(out, preset=preset, num_layers=layers, seed=seed, only=list(SUITE_CUTS[preset]))
    return out


def test_awq_nibble_order_is_not_involutive():
    """The AWQ permutation applied twice is not the identity.

    This is why packing must index the order table. Searching the table yields the inverse permutation and a silently
    wrong checkpoint.
    """
    order = ct_repack.AWQ_NIBBLE_ORDER
    assert tuple(order[i] for i in order) != tuple(range(8))
    assert order != tuple(order.index(j) for j in range(8))


def test_wrong_nibble_mapping_fails_roundtrip():
    """Packing with the inverse mapping must break the exact roundtrip."""
    torch.manual_seed(0)
    n, k = 128, 256
    packed, scale, zero = ct_repack.quantize_ct(torch.randn(n, k))
    codes = ct_repack.unpack_u4_lsb(packed).reshape(n, k)
    codes_t = codes.t().contiguous().view(k, n // 8, 8)

    wrong = torch.zeros(k, n // 8, dtype=torch.int32)
    for seq_j in range(8):
        # The classic order / order.index confusion.
        nibble_pos = ct_repack.AWQ_NIBBLE_ORDER.index(seq_j)
        wrong |= codes_t[..., seq_j].to(torch.int32) << (4 * nibble_pos)

    qweight, scales, qzeros = ct_repack.ct_to_awq(packed, scale, zero)
    assert not torch.equal(qweight, wrong)
    w_ct = ct_repack.ct_dequant(packed, scale)
    assert (w_ct - ct_repack.awq_dequant(wrong, scales, qzeros)).abs().max().item() > 0
    assert (w_ct - ct_repack.awq_dequant(qweight, scales, qzeros)).abs().max().item() == 0.0


def test_symmetric_qzeros_pad_convention():
    """Qzeros carry no information.

    Every nibble pads with 8.
    """

    torch.manual_seed(1)
    n, k = 64, 128
    packed, scale, zero = ct_repack.quantize_ct(torch.randn(n, k))
    qweight, scales, qzeros = ct_repack.ct_to_awq(packed, scale, zero)
    assert qzeros.dtype == torch.int32
    assert qzeros.shape == (1, n // 8)
    assert int(qzeros[0, 0]) == ct_repack.AWQ_SYMMETRIC_QZEROS_WORD
    assert torch.all(ct_repack.unpack_u4_lsb(qzeros) == ct_repack.CT_SYMMETRIC_ZERO)


def test_asymmetric_zero_point_rejected():
    """The converter only accepts the symmetric ct scheme (all-8 zero)."""
    torch.manual_seed(2)
    n, k = 64, 128
    packed, scale, _ = ct_repack.quantize_ct(torch.randn(n, k))
    bad = torch.full_like(_, 9, dtype=torch.int8)
    with pytest.raises(ValueError, match='zero point'):
        ct_repack.ct_to_awq(packed, scale, bad)


@pytest.mark.parametrize('preset', sorted(SUITE_CUTS))
def test_roundtrip_bit_exact(tmp_path, preset):
    """Generated presets convert with zero error at every level."""
    model_dir = _gen(tmp_path, preset)
    with open(os.path.join(model_dir, 'config.json')) as f:
        layers = json.load(f)['layers']
    assert len(layers) == len(SUITE_CUTS[preset])

    for key, shape in layers.items():
        n, k = shape['out_features'], shape['in_features']
        shard = tiny_ct.load_shard(model_dir, key)
        qweight, scales, qzeros = ct_repack.ct_to_awq(shard['weight_packed'], shard['weight_scale'],
                                                      shard['weight_zero_point'])

        assert qweight.dtype == torch.int32
        assert qweight.shape == (k, n // 8)
        assert scales.dtype == torch.float16
        assert scales.shape == (k // ct_repack.GROUP_SIZE, n)
        assert qzeros.dtype == torch.int32
        assert qzeros.shape == (k // ct_repack.GROUP_SIZE, n // 8)

        # Bit-level read-back through lmdeploy's own AWQ unpack.
        codes_ct = ct_repack.unpack_u4_lsb(shard['weight_packed']).reshape(n, k)
        assert torch.equal(_unpack_awq_gemm(qweight).t().contiguous(), codes_ct)
        assert torch.equal(scales, shard['weight_scale'].t().contiguous())
        assert torch.all(ct_repack.unpack_u4_lsb(qzeros) == ct_repack.CT_SYMMETRIC_ZERO)

        # Float roundtrip: err must be exactly zero, not a tolerance.
        err = (ct_repack.ct_dequant(shard['weight_packed'], shard['weight_scale'])
               - ct_repack.awq_dequant(qweight, scales, qzeros)).abs().max().item()
        assert err == 0.0, f'{key}: roundtrip err {err}'


def test_convert_checkpoint_streaming(tmp_path):
    """The directory conversion matches per-shard conversion bit for bit."""
    src = _gen(tmp_path, 'llama3_8b', layers=2)
    dst = os.path.join(str(tmp_path), 'awq')
    count, err = ct_repack.convert_checkpoint(src, dst, verify=True)
    assert count == 2 * len(SUITE_CUTS['llama3_8b'])
    assert err == 0.0

    with open(os.path.join(dst, 'config.json')) as f:
        out_config = json.load(f)
    assert out_config['format'] == 'autoawq'
    assert out_config['layout'] == ct_repack.AWQ_LAYOUT_DOC

    src_names = sorted(name for name in os.listdir(src) if name.endswith('.pt'))
    dst_names = sorted(name for name in os.listdir(dst) if name.endswith('.pt'))
    assert src_names == dst_names

    for name in src_names:
        direct, _ = ct_repack.convert_shard(
            torch.load(os.path.join(src, name), map_location='cpu', weights_only=True))
        on_disk = torch.load(os.path.join(dst, name), map_location='cpu', weights_only=True)
        assert sorted(direct) == sorted(on_disk)
        for field in direct:
            assert torch.equal(direct[field], on_disk[field]), f'{name}.{field} differs'


def test_generator_is_deterministic(tmp_path):
    """The same seed reproduces identical shards.

    A new seed changes the random fields.
    """

    model_a = os.path.join(str(tmp_path), 'a')
    model_b = os.path.join(str(tmp_path), 'b')
    model_c = os.path.join(str(tmp_path), 'c')
    tiny_ct.gen_tiny_ct(model_a, preset='glm4_9b', num_layers=1, seed=7, only=['qkv'])
    tiny_ct.gen_tiny_ct(model_b, preset='glm4_9b', num_layers=1, seed=7, only=['qkv'])
    tiny_ct.gen_tiny_ct(model_c, preset='glm4_9b', num_layers=1, seed=8, only=['qkv'])
    shard_a = tiny_ct.load_shard(model_a, tiny_ct.shard_key(0, 'qkv'))
    shard_b = tiny_ct.load_shard(model_b, tiny_ct.shard_key(0, 'qkv'))
    shard_c = tiny_ct.load_shard(model_c, tiny_ct.shard_key(0, 'qkv'))
    for field in ('weight_packed', 'weight_scale', 'bias'):
        assert torch.equal(shard_a[field], shard_b[field]), field
        assert not torch.equal(shard_a[field], shard_c[field]), field
    # The symmetric zero point is the constant 8. It must not vary with the seed.
    assert torch.equal(shard_a['weight_zero_point'], shard_b['weight_zero_point'])
    assert torch.equal(shard_a['weight_zero_point'], shard_c['weight_zero_point'])


def test_generator_rejects_unknown_names(tmp_path):
    with pytest.raises(ValueError, match='unknown linear names'):
        tiny_ct.gen_tiny_ct(str(tmp_path), preset='glm4_9b', only=['nope'])
