# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""CPU tests for the GPTQ W4A16 to AutoAWQ conversion library.

Covers the pack geometry traps the GPTQ format is known for:

- pack axis: GPTQ packs along K, AWQ packs along N.
- nibble order: GPTQ sequential vs the AWQ [0, 4, 1, 5, 2, 6, 3, 7] shuffle.
- zero-point offset: GPTQ stores ``zero_point - 1``.
- group-size geometry beyond the g128 default (g64 here).
"""

import pytest
import torch

from lmdeploy import turbomind
from lmdeploy.turbomind import gptq_repack
from lmdeploy.turbomind.ct_repack import AWQ_NIBBLE_ORDER, awq_dequant


def _make_weight(n, k, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, generator=generator)


def _tm_dequant(gptq_format, qweight, scales, qzeros):
    """Dequant through the TM GPTQFormat normalize path, [N, K] out."""
    w = gptq_format.normalize(qweight, 'weight')
    s = gptq_format.normalize(scales, 'scales')
    z = gptq_format.normalize(qzeros, 'zeros')
    g, n = w.shape
    group = w.shape[0] // s.shape[0]
    grouped = w.float().unflatten(0, (s.shape[0], group))
    return ((grouped - z.float()[:, None]) * s.float()[:, None]).flatten(0, 1).t()


@pytest.mark.parametrize('sym', [True, False], ids=['sym', 'asym'])
@pytest.mark.parametrize('group', [128, 64], ids=['g128', 'g64'])
def test_quantize_roundtrip(sym, group):
    """The synthesized checkpoint dequants back near the source weight."""
    n, k = 64, 256
    w = _make_weight(n, k)
    qweight, scales, qzeros = gptq_repack.quantize_gptq(w, group=group, sym=sym)
    assert qweight.shape == (k // 8, n)
    assert scales.shape == (k // group, n)
    assert qzeros.shape == (k // group, n // 8)
    deq = gptq_repack.gptq_dequant(qweight, scales, qzeros)
    rel = (deq - w).abs().max() / w.abs().max()
    assert rel < 0.15, f'rel err {rel:.4f}'
    if sym:
        # Symmetric GPTQ stores the constant 7 (= zero point 8 minus one).
        assert bool((gptq_repack.unpack_u4_lsb(qzeros) == 7).all())


@pytest.mark.parametrize('sym', [True, False], ids=['sym', 'asym'])
def test_gptq_to_awq_matches_gptq_dequant(sym):
    """The converted AWQ tensors dequant to the GPTQ reference values."""
    n, k = 64, 256
    w = _make_weight(n, k, seed=1)
    qweight, scales, qzeros = gptq_repack.quantize_gptq(w, group=128, sym=sym)
    awq_qw, awq_sc, awq_qz = gptq_repack.gptq_to_awq(qweight, scales, qzeros)
    assert awq_qw.shape == (k, n // 8)
    assert awq_sc.shape == (k // 128, n)
    err = (awq_dequant(awq_qw, awq_sc, awq_qz) - gptq_repack.gptq_dequant(qweight, scales, qzeros)).abs().max()
    assert err == 0.0, f'max abs err {err:.3e}'


def test_gptq_to_awq_synthesizes_symmetric_zeros():
    """A missing qzeros produces the AWQ symmetric pad word."""
    w = _make_weight(64, 256, seed=2)
    qweight, scales, _ = gptq_repack.quantize_gptq(w, sym=True)
    _, _, qz = gptq_repack.gptq_to_awq(qweight, scales)
    from lmdeploy.turbomind.ct_repack import AWQ_SYMMETRIC_QZEROS_WORD
    assert bool((qz == AWQ_SYMMETRIC_QZEROS_WORD).all())


def test_gptq_to_awq_verify_accepts_synthesized_symmetric_zeros():
    """Verify=True must not reject the synthesized symmetric zero words.

    The synthesized pad has shape (g, n//8) — int32 words, matching real qzeros geometry — so the internal dequant
    verification passes instead of crashing on a (g, n) tensor.
    """
    w = _make_weight(64, 256, seed=2)
    qweight, scales, _ = gptq_repack.quantize_gptq(w, sym=True)
    awq_qw, awq_sc, awq_qz = gptq_repack.gptq_to_awq(qweight, scales, verify=True)
    assert awq_qw.shape == (256, 64 // 8)
    from lmdeploy.turbomind.ct_repack import AWQ_SYMMETRIC_QZEROS_WORD
    assert bool((awq_qz == AWQ_SYMMETRIC_QZEROS_WORD).all())


def test_tm_normalize_matches_gptq_dequant():
    """The TurboMind GPTQFormat normalize path agrees with this library."""
    if not turbomind.is_available():
        pytest.skip('TurboMind is not built')
    from lmdeploy.turbomind.weight_format import GPTQFormat

    w = _make_weight(64, 256, seed=3)
    qweight, scales, qzeros = gptq_repack.quantize_gptq(w, group=64, sym=False)
    ref = gptq_repack.gptq_dequant(qweight, scales, qzeros)
    got = _tm_dequant(GPTQFormat(block_in=64), qweight, scales, qzeros)
    assert torch.allclose(got, ref, atol=1e-2), (got - ref).abs().max()


def test_pack_order_trap():
    """Weights packed in the AWQ nibble order must not read as GPTQ.

    The two orders differ, so a converter that skips the shuffle produces a silently permuted weight matrix. The trap
    asserts a readable error margin, not a near-miss.
    """
    n, k = 64, 256
    w = _make_weight(n, k, seed=4)
    qweight, scales, qzeros = gptq_repack.quantize_gptq(w, sym=True)
    codes = gptq_repack.unpack_u4_lsb(qweight)  # [K//8, N, 8] sequential
    # Re-pack nibble seq_j at position AWQ_NIBBLE_ORDER[seq_j]: the shuffle
    # a naive converter forgets.
    shuffled = torch.zeros_like(qweight)
    for seq_j in range(8):
        shuffled |= codes[..., seq_j].to(torch.int32) << (4 * AWQ_NIBBLE_ORDER[seq_j])
    ref = gptq_repack.gptq_dequant(qweight, scales, qzeros)
    wrong = gptq_repack.gptq_dequant(shuffled, scales, qzeros)
    assert (ref - wrong).abs().max() > 1.0


def test_zero_offset_trap():
    """Qzeros stored without the -1 offset shift every group by one scale."""
    n, k = 64, 256
    w = _make_weight(n, k, seed=5)
    qweight, scales, qzeros = gptq_repack.quantize_gptq(w, sym=True)
    ref = gptq_repack.gptq_dequant(qweight, scales, qzeros)
    # Store zero_point instead of zero_point - 1: nibbles 7 -> 8.
    codes = gptq_repack.unpack_u4_lsb(qzeros).to(torch.int32) + 1
    packed = torch.zeros_like(qzeros)
    for j in range(8):
        packed |= codes[..., j] << (4 * j)
    wrong = gptq_repack.gptq_dequant(qweight, scales, packed)
    assert (ref - wrong).abs().max() > 0.5 * scales.float().abs().max()


def test_geometry_rejections():
    """Broken shapes raise with the geometry rule in the message."""
    w = _make_weight(64, 256, seed=6)
    qweight, scales, qzeros = gptq_repack.quantize_gptq(w, group=64)
    with pytest.raises(ValueError, match='not a multiple of 8'):
        gptq_repack.gptq_to_awq(qweight[:, :57].contiguous(), scales[:, :57].contiguous(), None)
    with pytest.raises(ValueError, match='multiple of the group count'):
        gptq_repack.gptq_to_awq(qweight, torch.randn(3, 64), qzeros)


def test_zero_overflow_rejected():
    """A stored zero of 15 (+1 = 16) has no AWQ nibble; reject loudly."""
    qweight = torch.zeros(32, 64, dtype=torch.int32)
    scales = torch.ones(4, 64)
    codes = torch.full((4, 8, 8), 15, dtype=torch.int32)
    qzeros = torch.zeros(4, 8, dtype=torch.int32)
    for j in range(8):
        qzeros |= codes[..., j] << (4 * j)
    with pytest.raises(ValueError, match='exceeds 15'):
        gptq_repack.gptq_to_awq(qweight, scales, qzeros)


# ---------------------------------------------------------------------------
# normalize_gptq_weights: load-time classification and staging.
# ---------------------------------------------------------------------------


def _gptq_shard(prefix, seed=0):
    w = _make_weight(64, 256, seed=seed)
    qweight, scales, qzeros = gptq_repack.quantize_gptq(w, sym=True)
    return [(f'{prefix}qweight', qweight), (f'{prefix}scales', scales), (f'{prefix}qzeros', qzeros)]


def _awq_shard(prefix, seed=0):
    w = _make_weight(64, 256, seed=seed)
    qweight, scales, qzeros = gptq_repack.quantize_gptq(w, sym=True)
    qw, sc, qz = gptq_repack.gptq_to_awq(qweight, scales, qzeros)
    return [(f'{prefix}qweight', qw), (f'{prefix}scales', sc), (f'{prefix}qzeros', qz)]


def test_normalizer_converts_gptq_and_passes_awq():
    """Mixed streams: GPTQ modules convert, AWQ modules pass unchanged."""
    from lmdeploy.pytorch.weight_loader.model_weight_loader import normalize_gptq_weights

    awq = _awq_shard('layers.0.mlp.', seed=0)
    gptq = _gptq_shard('layers.1.mlp.', seed=1)
    out = dict(normalize_gptq_weights(iter(awq + gptq)))
    for name, tensor in awq:
        assert out[name] is tensor, name
    # The GPTQ qweight repacks from [K//8, N] to [K, N//8]; scales keep
    # their [G, N] orientation.
    assert out['layers.1.mlp.qweight'].shape == (256, 8) and gptq[0][1].shape == (32, 64)
    assert out['layers.1.mlp.qzeros'].shape == (2, 8)


def test_normalizer_synthesizes_missing_qzeros():
    """A GPTQ module without qzeros flushes with the symmetric pad."""
    from lmdeploy.pytorch.weight_loader.model_weight_loader import normalize_gptq_weights

    shard = [(n, t) for n, t in _gptq_shard('a.', seed=7) if not n.endswith('qzeros')]
    extra = [('b.weight', torch.zeros(1))]
    out = dict(normalize_gptq_weights(iter(shard + extra)))
    assert out['a.qweight'].shape == (256, 8)
    from lmdeploy.turbomind.ct_repack import AWQ_SYMMETRIC_QZEROS_WORD
    assert bool((out['a.qzeros'] == AWQ_SYMMETRIC_QZEROS_WORD).all())


def test_normalizer_rejects_unknown_geometry():
    """Shapes matching neither geometry raise with both rules named."""
    from lmdeploy.pytorch.weight_loader.model_weight_loader import normalize_gptq_weights

    shard = [('a.qweight', torch.zeros(4, 8, dtype=torch.int32)), ('a.scales', torch.zeros(3, 9)),
             ('a.qzeros', torch.zeros(3, 1, dtype=torch.int32))]
    with pytest.raises(ValueError, match='neither the GPTQ nor the AWQ'):
        list(normalize_gptq_weights(iter(shard)))


def test_normalizer_rejects_stray_qweight():
    """A qweight without scales means a corrupt stream and raises."""
    from lmdeploy.pytorch.weight_loader.model_weight_loader import normalize_gptq_weights

    with pytest.raises(ValueError, match='no scales'):
        list(normalize_gptq_weights(iter([('a.qweight', torch.zeros(4, 8, dtype=torch.int32))])))


def test_normalizer_rejects_reordered_g_idx():
    """A reordered group mapping is real act-order and must raise."""
    from lmdeploy.pytorch.weight_loader.model_weight_loader import normalize_gptq_weights

    reordered = torch.tensor([1, 0, 3, 2], dtype=torch.int64)
    with pytest.raises(ValueError, match='desc_act'):
        list(normalize_gptq_weights(iter([('layers.0.g_idx', reordered)])))


def test_normalizer_consumes_sequential_g_idx():
    """AutoGPTQ ships identity group indices without act-order; consume them.

    Rejecting every g_idx would reject standard exports produced by this repository's own GPTQ exporter.
    """
    from lmdeploy.pytorch.weight_loader.model_weight_loader import normalize_gptq_weights

    shard = _gptq_shard('a.', seed=8)
    identity = torch.arange(2, dtype=torch.int64)  # two groups, identity order
    out = dict(normalize_gptq_weights(iter(shard + [('a.g_idx', identity)])))
    assert set(out) == {'a.qweight', 'a.scales', 'a.qzeros'}


def test_normalizer_keeps_split_module_staged_without_early_synthesis():
    """A module whose tensors interleave with another's stays staged.

    The old flush-on-next-prefix synthesized a symmetric pad for 'a' when 'b' opened; with asymmetric zeros still to
    come that pad was wrong. The stream below must convert both modules exactly, no early synthesis.
    """
    from lmdeploy.pytorch.weight_loader.model_weight_loader import normalize_gptq_weights

    a = _gptq_shard('a.', seed=11)
    b = _gptq_shard('b.', seed=12)
    interleaved = [a[0], b[0], b[1], b[2], a[1], a[2]]  # a splits around b
    reference = dict(normalize_gptq_weights(iter(a + b)))
    out = dict(normalize_gptq_weights(iter(interleaved)))
    assert set(out) == set(reference)
    for name, tensor in reference.items():
        assert torch.equal(out[name], tensor), name


def test_shared_state_carries_module_across_streams():
    """A module split across shard files still stages and converts exactly.

    The loader normalizes every shard with one shared state, so a shard boundary between a module's tensors must not end
    up flushing a synthesized pad early and then rejecting the real tensor in the next shard.
    """
    from lmdeploy.pytorch.weight_loader.model_weight_loader import (
        finalize_gptq_weights,
        new_gptq_normalize_state,
        normalize_gptq_weights,
    )

    full = dict(_gptq_shard('layers.0.mlp.', seed=3))
    reference = dict(normalize_gptq_weights(iter(_gptq_shard('layers.0.mlp.', seed=3))))
    state = new_gptq_normalize_state()

    shard_a = list(normalize_gptq_weights(iter([
        ('layers.0.mlp.qweight', full['layers.0.mlp.qweight']),
        ('layers.0.mlp.qzeros', full['layers.0.mlp.qzeros']),
        ('lm_head.weight', torch.zeros(1)),
    ]), state))
    # Nothing complete yet: the passthrough key flows, the quant pair stays
    # staged (an early flush would synthesize a wrong asymmetric pad).
    assert [name for name, _ in shard_a] == ['lm_head.weight']

    shard_b = dict(normalize_gptq_weights(iter([
        ('layers.0.mlp.scales', full['layers.0.mlp.scales']),
    ]), state))
    assert set(shard_b) == {
        'layers.0.mlp.qweight', 'layers.0.mlp.scales', 'layers.0.mlp.qzeros'
    }
    for name in ('layers.0.mlp.qweight', 'layers.0.mlp.scales', 'layers.0.mlp.qzeros'):
        assert torch.equal(shard_b[name], reference[name]), name
    assert list(finalize_gptq_weights(state)) == []


def test_update_path_clones_staged_tensors_against_buffer_reuse():
    """The update path (shared state) must clone staged tensors.

    Update buckets deliver views into a reused IPC buffer: when a module's
    scales arrive in a later bucket than its weights, the conversion would
    otherwise run on the buffer's current (corrupted) contents instead of
    the tensors as loaded.
    """
    from lmdeploy.pytorch.weight_loader.model_weight_loader import (
        finalize_gptq_weights,
        new_gptq_normalize_state,
        normalize_gptq_weights,
    )

    full = dict(_gptq_shard('layers.0.mlp.', seed=3))
    reference = dict(normalize_gptq_weights(iter(_gptq_shard('layers.0.mlp.', seed=3))))
    state = new_gptq_normalize_state()

    buf = full['layers.0.mlp.qweight']
    shard_a = list(normalize_gptq_weights(iter([
        ('layers.0.mlp.qweight', buf),
        ('layers.0.mlp.qzeros', full['layers.0.mlp.qzeros']),
    ]), state))
    assert shard_a == []

    # The IPC buffer is reused for the next bucket: corrupt it in place.
    buf.copy_(torch.zeros_like(buf))

    shard_b = dict(normalize_gptq_weights(iter([
        ('layers.0.mlp.scales', full['layers.0.mlp.scales']),
    ]), state))
    assert set(shard_b) == {
        'layers.0.mlp.qweight', 'layers.0.mlp.scales', 'layers.0.mlp.qzeros'
    }
    for name in ('layers.0.mlp.qweight', 'layers.0.mlp.scales', 'layers.0.mlp.qzeros'):
        assert torch.equal(shard_b[name], reference[name]), name
    assert list(finalize_gptq_weights(state)) == []


def test_finalize_flushes_missing_qzeros_and_rejects_strays():
    """Finalize mirrors end-of-stream: synthesize pads, reject scales-less strays."""
    from lmdeploy.pytorch.weight_loader.model_weight_loader import (
        finalize_gptq_weights,
        new_gptq_normalize_state,
        normalize_gptq_weights,
    )
    from lmdeploy.turbomind.ct_repack import AWQ_SYMMETRIC_QZEROS_WORD

    # A module that never ships qzeros flushes with the symmetric pad.
    state = new_gptq_normalize_state()
    shard = [(n, t) for n, t in _gptq_shard('a.', seed=7) if not n.endswith('qzeros')]
    list(normalize_gptq_weights(iter(shard), state))
    out = dict(finalize_gptq_weights(state))
    assert set(out) == {'a.qweight', 'a.scales', 'a.qzeros'}
    assert bool((out['a.qzeros'] == AWQ_SYMMETRIC_QZEROS_WORD).all())

    # A qweight without scales anywhere is a corrupt sequence: raise.
    state = new_gptq_normalize_state()
    list(normalize_gptq_weights(iter([('b.qweight', torch.zeros(4, 8, dtype=torch.int32))]), state))
    with pytest.raises(ValueError, match='no scales'):
        list(finalize_gptq_weights(state))


def test_gptq_quant_method_routes_through_awq_family():
    """GPTQ dense linears report the AWQ quant method; norms opt out."""
    from lmdeploy.pytorch.config import QuantizationConfig

    config = QuantizationConfig.from_config(
        type('Cfg', (), {'quantization_config': {
            'quant_method': 'gptq', 'bits': 4, 'group_size': 128
        }})())
    assert config.bits == 4
    assert config.group_size == 128
    assert config.get_quant_method('model.layers.0.mlp.gate_proj', module_kind='linear') == 'awq'
    assert config.get_quant_method('model.layers.0.input_layernorm', module_kind='norm') is None
    # GPTQ expert tensors normalize through the same suffix filter, so MoE
    # rides the AWQ family exactly like ct group-128 does.
    assert config.get_quant_method('model.layers.0.mlp.experts', module_kind='moe') == 'awq'
    # An ignore list without canonical prefixes cannot be honored — refuse
    # instead of silently quantizing ignored modules.
    config.ignored_layers = ['lm_head']
    with pytest.raises(ValueError, match='non-empty canonical module'):
        config.get_quant_method(module_kind='linear')
    assert config.get_quant_method('lm_head', module_kind='linear') is None


def test_gptq_config_rejects_desc_act():
    """desc_act fails at config parse with a clear message."""
    from lmdeploy.pytorch.config import QuantizationConfig

    with pytest.raises(ValueError, match='desc_act'):
        QuantizationConfig.from_config(
            type('Cfg', (), {'quantization_config': {
                'quant_method': 'gptq', 'bits': 4, 'group_size': 128, 'desc_act': True
            }})())


def test_convert_checkpoint_gathers_shards_and_rewrites_metadata(tmp_path):
    """The offline converter crosses shards, drops identity g_idx, and rewrites
    the config to AWQ."""
    import json
    import os

    from safetensors.torch import load_file, save_file

    from lmdeploy.pytorch.weight_loader.model_weight_loader import normalize_gptq_weights

    src = tmp_path / 'src'
    dst = tmp_path / 'dst'
    os.makedirs(src)
    a = _gptq_shard('layers.0.mlp.down_proj.', seed=13)
    save_file({a[0][0]: a[0][1], 'lm_head.weight': torch.zeros(2)},
              str(src / 'model-00001.safetensors'))
    # scales/qzeros of the same module live in the second shard; identity
    # g_idx ships alongside, as AutoGPTQ writes it.
    save_file({a[1][0]: a[1][1], a[2][0]: a[2][1],
               'layers.0.mlp.down_proj.g_idx': torch.arange(2, dtype=torch.int64)},
              str(src / 'model-00002.safetensors'))
    (src / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {}}))
    (src / 'config.json').write_text(json.dumps({
        'quantization_config': {'quant_method': 'gptq', 'bits': 4, 'desc_act': False}}))
    # Only the exact basename is rewritten; sibling configs pass through
    # byte for byte.
    (src / 'tokenizer_config.json').write_text('{"do_sample": true}')
    (src / 'generation_config.json').write_text('{"max_length": 20}')

    num_modules, err = gptq_repack.convert_checkpoint(str(src), str(dst), verify=True)
    assert num_modules == 1 and err == 0.0
    assert (dst / 'tokenizer_config.json').read_text() == '{"do_sample": true}'
    assert (dst / 'generation_config.json').read_text() == '{"max_length": 20}'

    converted = load_file(str(dst / 'model.safetensors'))
    assert 'lm_head.weight' in converted
    assert not any(name.endswith('g_idx') for name in converted)
    reference = dict(normalize_gptq_weights(iter(a)))
    for suffix in ('qweight', 'scales', 'qzeros'):
        assert torch.equal(converted[f'layers.0.mlp.down_proj.{suffix}'], reference[f'layers.0.mlp.down_proj.{suffix}'])
    config = json.loads((dst / 'config.json').read_text())
    assert config['quantization_config']['quant_method'] == 'awq'
    assert not os.path.exists(dst / 'model.safetensors.index.json')


def test_loader_flushes_leftover_through_rename_pipeline():
    """The post-loop flushed tail reaches load_weights renamed, like every
    shard tensor.

    A symmetric GPTQ export drops the tail module's qzeros, so it converts only at finalize — after all shards. Its
    tensors must go through the same rename (and dummy filter) as the in-loop stream, not bypass it.
    """
    from types import SimpleNamespace

    from lmdeploy.pytorch.weight_loader.model_weight_loader import (
        ModelWeightLoader,
        normalize_gptq_weights,
    )

    gptq = _gptq_shard('layers.0.mlp.', seed=5)
    symmetric = [(n, t) for n, t in gptq if not n.endswith('qzeros')]
    shards = {
        'shard_a': symmetric,  # tail module: converts only at finalize
        'shard_b': [('lm_head.weight', torch.zeros(1))],
    }
    loaded = []
    model = SimpleNamespace(
        rename_weight=lambda n: 'renamed.' + n,
        load_weights=lambda weights: loaded.extend(weights),
        named_modules=lambda: iter([]),
    )

    loader = ModelWeightLoader.__new__(ModelWeightLoader)
    loader._shard_paths = ['shard_b', 'shard_a']
    loader._weight_type = 'safetensors'
    loader._prefix = None
    loader._get_weights_iterator = lambda path: iter(shards[path])

    loader.load_model_weights(model, normalize_quant=True)

    names = [name for name, _ in loaded]
    # The flushed tail arrives renamed; nothing bypasses the pipeline.
    assert names == ['renamed.lm_head.weight', 'renamed.layers.0.mlp.qweight',
                     'renamed.layers.0.mlp.scales', 'renamed.layers.0.mlp.qzeros']
    by_name = dict(loaded)
    reference = dict(normalize_gptq_weights(iter(gptq)))
    for name in ('layers.0.mlp.qweight', 'layers.0.mlp.scales', 'layers.0.mlp.qzeros'):
        assert torch.equal(by_name[f'renamed.{name}'], reference[name]), name
