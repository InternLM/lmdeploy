# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Compressed-tensors W4A16 to AutoAWQ checkpoint conversion.

The module owns the bit-level conversion between two packed u4 layouts. It
serves offline checkpoint repack and the test fixtures that synthesize tiny
compressed-tensors models. Loading-path integration is out of scope here.

Checkpoint layouts
------------------

ct input shard, one ``.pt`` file per linear, groups along the input dim K:

- ``weight_packed``     int32 [N, K // 8]. Eight 4-bit codes per int32,
  LSB-first along K. Code = q + 8, q in [-8, 7].
- ``weight_scale``      float16 [N, G]. One scale per group of 128 along K.
- ``weight_zero_point`` int8 [N, G]. Constant 8. Symmetric schemes carry
  no packed zero point. Compressed-tensors stores the constant unpacked.
- ``bias``              float16 [N].

AutoAWQ output shard, same file naming:

- ``qweight`` int32 [K, N // 8]. One int32 holds the 8 output channels of
  one input row k. Sequential channel j lands at nibble position
  ``AWQ_NIBBLE_ORDER[j]``. This permutation is not an involution: packing
  with ``AWQ_NIBBLE_ORDER.index(j)`` instead still produces valid-looking
  u4 data, but the weights are wrong. The suite keeps a defensive test on
  that trap.
- ``scales`` float16 [G, N].
- ``qzeros`` int32 [G, N // 8]. Symmetric checkpoints carry no zero point.
  The AutoAWQ convention pads every nibble with 8, so each word reads
  ``0x88888888`` (int32 value -2004318072).
- ``bias``   float16 [N].

Both layouts dequantize as ``w = (code - 8) * scale`` per group along K.
``awq_dequant`` reads converted tensors back through lmdeploy's own
``weight_format._unpack_awq_gemm``, so the roundtrip shares one code
path with checkpoint loading and cannot drift from the loader.

Streaming and memory
--------------------

``convert_checkpoint`` processes one shard (one linear) at a time and never
holds two shards. Peak host memory stays near 3 bytes per weight element of
the largest shard. Breakdown: packed input 0.5, unpacked uint8 codes 1.0,
transposed copy 1.0, qweight output 0.5. The bound depends on the single
largest linear only, not on the shard count. ``verify`` adds about 8 bytes
per element for two float32 dequant copies. A 190.8 GB real checkpoint
therefore repacks in host memory bounded by its largest fused linear. A
2B-parameter shard needs about 6 GB without ``verify``. It needs about
22 GB with ``verify``.

Example
-------

    from lmdeploy.turbomind import ct_repack
    ct_repack.convert_checkpoint('/path/ct_model', '/path/awq_model')
"""

from __future__ import annotations

import json
import os

import torch

AWQ_NIBBLE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
CT_SYMMETRIC_ZERO = 8

# The literal 0x88888888 exceeds the int32 range. Materialize through int64
# and cast, or torch.full with dtype=torch.int32 raises.
AWQ_SYMMETRIC_QZEROS_WORD = int(torch.full((), 0x88888888, dtype=torch.int64).to(torch.int32).item())

GROUP_SIZE = 128
BITS = 4
WEIGHT_DTYPE = torch.float16

AWQ_LAYOUT_DOC = {
    'qweight': 'int32 [K, N//8]. AWQ nibble order [0, 4, 1, 5, 2, 6, 3, 7].',
    'scales': 'float16 [G, N].',
    'qzeros': 'int32 [G, N//8]. Symmetric pad 0x88888888.',
    'dequant': 'w = (code - 8) * scale. Groups of 128 along K.',
}


def validate_ct_geometry(lines) -> None:
    """Reject linear shapes that break the W4A16 g128 conversion constraints.

    ``lines`` is an iterable of ``(name, out_features, in_features)``.
    """
    for name, n, k in lines:
        if k % GROUP_SIZE != 0:
            raise ValueError(f'{name}: in_features {k} is not a multiple of the group size {GROUP_SIZE}')
        if n % 8 != 0:
            raise ValueError(f'{name}: out_features {n} is not a multiple of 8 (AWQ packs 8 channels per int32)')


def quantize_ct(w: torch.Tensor, group: int = GROUP_SIZE, bits: int = BITS):
    """Quantize a float weight [N, K] with ct W4A16 symmetric semantics.

    Returns ``(weight_packed int32 [N, K // 8], weight_scale float16
    [N, G], weight_zero_point int8 [N, G])``.
    """
    n, k = w.shape
    g = k // group
    wf = w.float().view(n, g, group)
    amax = wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / (2 ** (bits - 1) - 1)
    q = torch.clamp(torch.round(wf / scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    code = (q + 2 ** (bits - 1)).to(torch.int32).view(n, g, group // 8, 8)
    packed = torch.zeros(n, g, group // 8, dtype=torch.int32)
    for j in range(8):
        # pack-quantized: element j of the group occupies bits [4j, 4j + 4)
        packed |= code[..., j] << (4 * j)
    packed = packed.view(n, k // 8)
    scale = scale.squeeze(-1).to(WEIGHT_DTYPE)
    zero = torch.full((n, g), CT_SYMMETRIC_ZERO, dtype=torch.int8)
    return packed, scale, zero


def unpack_u4_lsb(packed: torch.Tensor) -> torch.Tensor:
    """Split int32 words into uint8 nibbles LSB-first: [.., M] -> [.., M, 8]."""
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 4
    return ((packed[..., None] >> shifts) & 15).to(torch.uint8)


def ct_dequant(weight_packed: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """Reference dequant of a ct shard to float32 [N, K]."""
    n, k8 = weight_packed.shape
    k = k8 * 8
    codes = unpack_u4_lsb(weight_packed).reshape(n, k).float() - CT_SYMMETRIC_ZERO
    scale = weight_scale.float().repeat_interleave(k // weight_scale.shape[1], dim=1)
    return codes * scale


def ct_to_awq(weight_packed: torch.Tensor,
              weight_scale: torch.Tensor,
              weight_zero_point: torch.Tensor | None = None):
    """Convert the ct tensors of one linear to the AutoAWQ layout.

    Returns ``(qweight int32 [K, N // 8], scales float16 [G, N], qzeros
    int32 [G, N // 8])``.
    """
    n, k8 = weight_packed.shape
    k = k8 * 8
    g = weight_scale.shape[1]
    if weight_scale.shape[0] != n or k != g * GROUP_SIZE:
        raise ValueError(
            f'shape mismatch: weight_packed {tuple(weight_packed.shape)}, weight_scale {tuple(weight_scale.shape)}')
    if weight_zero_point is not None and not bool((weight_zero_point == CT_SYMMETRIC_ZERO).all()):
        raise ValueError('ct symmetric zero point must be 8 everywhere')

    codes = unpack_u4_lsb(weight_packed).reshape(n, k)
    codes_t = codes.t().contiguous()
    del codes
    view = codes_t.view(k, n // 8, 8)
    qweight = torch.zeros(k, n // 8, dtype=torch.int32, device=weight_packed.device)
    for seq_j in range(8):
        # sequential channel seq_j lands at nibble position AWQ_NIBBLE_ORDER[seq_j]
        nibble_pos = AWQ_NIBBLE_ORDER[seq_j]
        qweight |= view[..., seq_j].to(torch.int32) << (4 * nibble_pos)
    scales = weight_scale.t().contiguous()
    qzeros = torch.full((g, n // 8), AWQ_SYMMETRIC_QZEROS_WORD, dtype=torch.int32, device=weight_packed.device)
    return qweight, scales, qzeros


def awq_dequant(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor) -> torch.Tensor:
    """Dequant AutoAWQ tensors back to float32 [N, K].

    Reads through lmdeploy's own AWQ unpack, so this path matches checkpoint loading bit for bit. The import stays
    local: the rest of the module must import without a built TurboMind extension.
    """
    from lmdeploy.turbomind.weight_format import _unpack_awq_gemm

    k, n8 = qweight.shape
    n = n8 * 8
    g, _ = scales.shape
    group = k // g
    w = _unpack_awq_gemm(qweight).view(g, group, n)
    zero = _unpack_awq_gemm(qzeros)
    return ((w.float() - zero[:, None]) * scales.float()[:, None]).reshape(-1, n).t()


def convert_shard(ct_shard: dict, verify: bool = False):
    """Convert one ct shard dict to the AutoAWQ shard dict.

    With ``verify``, run the exact roundtrip and raise on any nonzero
    error. Returns ``(shard_dict, err)``. ``err`` is ``None`` without
    ``verify``.
    """
    qweight, scales, qzeros = ct_to_awq(ct_shard['weight_packed'], ct_shard['weight_scale'],
                                        ct_shard.get('weight_zero_point'))
    out = {'qweight': qweight, 'scales': scales, 'qzeros': qzeros}
    if 'bias' in ct_shard:
        out['bias'] = ct_shard['bias']
    err = None
    if verify:
        err = (ct_dequant(ct_shard['weight_packed'], ct_shard['weight_scale'])
               - awq_dequant(qweight, scales, qzeros)).abs().max().item()
        if err != 0.0:
            raise AssertionError(f'roundtrip mismatch: max abs err {err}')
    return out, err


def convert_checkpoint(src: str, dst: str, verify: bool = False):
    """Convert a ct checkpoint directory to the AutoAWQ layout.

    A ct directory holds one ``config.json`` plus one ``.pt`` file per
    linear; see the module docstring for the shard schema. Shards stream
    one at a time, so peak host memory follows the largest shard (see the
    module docstring). Returns ``(shard_count, worst_roundtrip_err)``; the
    error is ``None`` without ``verify``.
    """
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, 'config.json')) as f:
        config = json.load(f)
    names = sorted(name for name in os.listdir(src) if name.endswith('.pt'))
    if not names:
        raise FileNotFoundError(f'no .pt shards under {src}')
    worst = None
    for name in names:
        shard = torch.load(os.path.join(src, name), map_location='cpu', weights_only=True)
        awq, err = convert_shard(shard, verify=verify)
        torch.save(awq, os.path.join(dst, name))
        if err is not None:
            worst = err if worst is None else max(worst, err)
        del shard, awq
    out_config = {**config, 'format': 'autoawq', 'layout': AWQ_LAYOUT_DOC}
    with open(os.path.join(dst, 'config.json'), 'w') as f:
        json.dump(out_config, f, indent=2)
    note = '' if worst is None else f', worst roundtrip err {worst}'
    print(f'converted {len(names)} shards to {dst}{note}')
    return len(names), worst
