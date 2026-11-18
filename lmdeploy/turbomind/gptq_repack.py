# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""GPTQ W4A16 (sym / asym) to AutoAWQ checkpoint conversion.

The pytorch engine has no native GPTQ linear modules. GPTQ dense linears
load through the AWQ family instead: this library converts one GPTQ module
of a checkpoint to the canonical AutoAWQ layout at load time, so both the
AWQ Triton kernel and the TurboMind GEMM route consume the tensors
unchanged (see ``normalize_gptq_weights``).

GPTQ checkpoint layout (AutoGPTQ, nibble LSB-first):

- ``qweight``: int32 [K // 8, N]. The word at ``[k8, n]`` packs the 8 input
  channels ``8 * k8 .. 8 * k8 + 7`` of output channel ``n``; nibble ``j``
  holds channel ``8 * k8 + j``. Packing runs along K.
- ``scales``: float [G, N]. Already in TM/AWQ orientation (no transpose).
- ``qzeros``: int32 [G, N // 8]. The word at ``[g, n8]`` packs the 8 output
  channels ``8 * n8 .. 8 * n8 + 7`` of group ``g``. The stored nibble is
  ``zero_point - 1``; consumers add 1 back (the TurboMind ``GPTQFormat``
  normalize does, and this conversion bakes it into the AWQ word).
- dequant: ``w[n, k] = (q[k, n] - (z[k // group, n] + 1)) * scale[k // group, n]``.

Differences from AWQ that this conversion absorbs:

- pack axis: GPTQ packs along K, AWQ packs along N.
- nibble order: GPTQ is sequential, AWQ uses ``[0, 4, 1, 5, 2, 6, 3, 7]``.
- zero-point offset: GPTQ stores ``zero_point - 1``, AWQ stores the raw
  zero point.

Example::

    from lmdeploy.turbomind import gptq_repack
    gptq_repack.convert_checkpoint('/path/gptq_model', '/path/awq_model')
"""

from __future__ import annotations

import os

import torch

from .ct_repack import AWQ_NIBBLE_ORDER, AWQ_SYMMETRIC_QZEROS_WORD, unpack_u4_lsb

GROUP_SIZE = 128
WEIGHT_DTYPE = torch.float16

GPTQ_LAYOUT_DOC = {
    'qweight': 'int32 [K // 8, N]. Sequential nibbles LSB-first along K.',
    'scales': 'float [G, N]. Same orientation as AWQ.',
    'qzeros': 'int32 [G, N // 8]. Sequential nibbles along N, value = zero_point - 1.',
    'dequant': 'w = (code - (zero + 1)) * scale. Groups along K.',
}


def validate_gptq_geometry(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor | None) -> None:
    """Reject GPTQ shapes that cannot map onto the AWQ layout.

    ``qzeros`` is optional: symmetric checkpoints carry it, and a missing
    tensor means the conversion synthesizes the symmetric zero word.
    """
    k8, n = qweight.shape
    g, n_scales = scales.shape
    k = k8 * 8
    if n_scales != n:
        raise ValueError(f'shape mismatch: qweight {tuple(qweight.shape)} vs scales {tuple(scales.shape)}; '
                         f'GPTQ packs along K, so qweight.shape[-1] must equal scales.shape[-1]')
    if n % 8 != 0:
        raise ValueError(f'{n}: out_features is not a multiple of 8 (AWQ packs 8 channels per int32)')
    if k % g != 0:
        raise ValueError(f'in_features {k} is not a multiple of the group count {g} '
                         f'(group size {k // g})')
    if qzeros is not None and tuple(qzeros.shape) != (g, n // 8):
        raise ValueError(f'shape mismatch: scales {tuple(scales.shape)} vs qzeros {tuple(qzeros.shape)}; '
                         f'expected ({g}, {n // 8})')


def _gptq_sym_qzeros(qweight: torch.Tensor, g: int) -> torch.Tensor:
    """Synthesize the symmetric GPTQ zero word (stored zero point 8 - 1)."""
    return torch.full((g, qweight.shape[-1]), 0x77777777, dtype=torch.int32, device=qweight.device)


def gptq_dequant(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor) -> torch.Tensor:
    """Reference dequant of a GPTQ module to float32 [N, K]."""
    validate_gptq_geometry(qweight, scales, qzeros)
    k8, n = qweight.shape
    k = k8 * 8
    g = scales.shape[0]
    group = k // g

    # Nibble j of word [k8, n] is input channel 8 * k8 + j.
    codes = unpack_u4_lsb(qweight).permute(1, 0, 2).reshape(n, k).float()
    # Stored nibble is zero_point - 1; the kernel semantics add 1 back.
    zero = unpack_u4_lsb(qzeros).reshape(g, n).float() + 1
    zero = zero.t().repeat_interleave(group, dim=1)
    scale = scales.float().t().repeat_interleave(group, dim=1)
    return (codes - zero) * scale


def quantize_gptq(w: torch.Tensor, group: int = GROUP_SIZE, sym: bool = True):
    """Quantize a float weight [N, K] with GPTQ packing semantics.

    Returns ``(qweight int32 [K // 8, N], scales float16 [G, N], qzeros
    int32 [G, N // 8])`` — the raw GPTQ checkpoint layout. The quantization
    grid matches AutoGPTQ: symmetric divides by ``amax / 7`` with a fixed
    zero point 8 (stored as 7); asymmetric uses ``(max - min) / 15`` with a
    per-group zero point clamped to ``[1, 15]`` so the stored ``zero - 1``
    stays a valid nibble.
    """
    n, k = w.shape
    if k % group != 0:
        raise ValueError(f'in_features {k} is not a multiple of the group size {group}')
    g = k // group
    wf = w.float().view(n, g, group)
    if sym:
        amax = wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = amax / 7
        q = torch.clamp(torch.round(wf / scale), -8, 7).to(torch.int32) + 8
        stored_zero = torch.full((n, g), 7, dtype=torch.int32)
    else:
        wmax = wf.amax(dim=-1, keepdim=True)
        wmin = wf.amin(dim=-1, keepdim=True)
        scale = ((wmax - wmin) / 15).clamp(min=1e-8)
        # Clamp the zero point to [1, 15] so zero - 1 fits a nibble; a
        # checkpoint zero of 16 has no AWQ representation.
        zero = torch.clamp(torch.round(-wmin / scale), 1, 15).to(torch.int32)
        q = torch.clamp(torch.round(wf / scale) + zero, 0, 15).to(torch.int32)
        stored_zero = (zero - 1).squeeze(-1)
    stored_zero = stored_zero.reshape(n, g)

    # Pack 8 consecutive input channels into one int32 word, LSB-first.
    q = q.reshape(n, k // 8, 8)
    packed = torch.zeros(n, k // 8, dtype=torch.int32)
    for j in range(8):
        packed |= q[..., j] << (4 * j)
    qweight = packed.t().contiguous()

    # Pack the stored zero points along N, LSB-first sequential nibbles.
    zero_view = stored_zero.t().contiguous().reshape(g, n // 8, 8)
    qzeros = torch.zeros(g, n // 8, dtype=torch.int32)
    for j in range(8):
        qzeros |= zero_view[..., j] << (4 * j)
    return qweight, scale.squeeze(-1).t().contiguous().to(WEIGHT_DTYPE), qzeros


def gptq_to_awq(qweight: torch.Tensor,
                scales: torch.Tensor,
                qzeros: torch.Tensor | None = None,
                verify: bool = False):
    """Convert the GPTQ tensors of one linear to the AutoAWQ layout.

    Returns ``(qweight int32 [K, N // 8], scales float [G, N], qzeros int32
    [G, N // 8])``. A missing ``qzeros`` synthesizes the symmetric zero
    word, matching ``GPTQFormat.synthesize_zeros``.
    """
    validate_gptq_geometry(qweight, scales, qzeros)
    k8, n = qweight.shape
    k = k8 * 8
    g, _ = scales.shape

    codes = unpack_u4_lsb(qweight).permute(1, 0, 2).reshape(n, k)
    codes_t = codes.t().contiguous()
    del codes
    view = codes_t.view(k, n // 8, 8)
    awq_qweight = torch.zeros(k, n // 8, dtype=torch.int32)
    for seq_j in range(8):
        # Sequential GPTQ channel seq_j lands at nibble position
        # AWQ_NIBBLE_ORDER[seq_j].
        nibble_pos = AWQ_NIBBLE_ORDER[seq_j]
        awq_qweight |= view[..., seq_j].to(torch.int32) << (4 * nibble_pos)

    if qzeros is None:
        awq_qzeros = torch.full((g, n // 8), AWQ_SYMMETRIC_QZEROS_WORD, dtype=torch.int32)
    else:
        # Bake the +1 offset into the AWQ word: the AWQ kernels subtract the
        # unpacked value directly. A zero point of 16 overflows the nibble
        # and has no AWQ representation, so reject it loudly.
        zero = unpack_u4_lsb(qzeros).reshape(g, n).to(torch.int32) + 1
        if bool((zero > 15).any()):
            raise ValueError('GPTQ zero point exceeds 15 after the +1 offset; '
                             'the AWQ 4-bit layout cannot represent it')
        zero_view = zero.reshape(g, n // 8, 8)
        awq_qzeros = torch.zeros(g, n // 8, dtype=torch.int32)
        for seq_j in range(8):
            nibble_pos = AWQ_NIBBLE_ORDER[seq_j]
            awq_qzeros |= zero_view[..., seq_j] << (4 * nibble_pos)

    if verify:
        from .ct_repack import awq_dequant
        ref = gptq_dequant(qweight, scales, qzeros if qzeros is not None else _gptq_sym_qzeros(qweight, g))
        got = awq_dequant(awq_qweight, scales.contiguous(), awq_qzeros)
        err = (ref - got).abs().max().item()
        if err > 1e-3:
            raise ValueError(f'GPTQ to AWQ conversion verification failed: max abs err {err:.3e}')
    return awq_qweight, scales.contiguous(), awq_qzeros


def convert_shard(gptq_shard: dict, verify: bool = False):
    """Convert one GPTQ shard dict to the AutoAWQ shard dict.

    ``gptq_shard`` maps the bare suffixes ``qweight`` / ``scales`` /
    ``qzeros`` to tensors. Returns ``(out_shard, max_abs_err)``; the error
    is 0.0 unless ``verify`` asks for the dequant round trip.
    """
    qweight = gptq_shard['qweight']
    scales = gptq_shard['scales']
    qzeros = gptq_shard.get('qzeros')
    out = dict(zip(('qweight', 'scales', 'qzeros'), gptq_to_awq(qweight, scales, qzeros)))
    err = 0.0
    if verify:
        from .ct_repack import awq_dequant
        ref = gptq_dequant(qweight, scales, qzeros if qzeros is not None else _gptq_sym_qzeros(qweight,
                                                                                             scales.shape[0]))
        got = awq_dequant(out['qweight'], out['scales'], out['qzeros'])
        err = (ref - got).abs().max().item()
    return out, err


def convert_checkpoint(src: str, dst: str, verify: bool = False):
    """Convert a GPTQ checkpoint directory to the AutoAWQ layout.

    Copies every file, rewriting safetensors shards; other files (config,
    tokenizer) pass through byte for byte. Returns ``(num_modules,
    max_abs_err)``.
    """
    from safetensors.torch import load_file, save_file

    os.makedirs(dst, exist_ok=True)
    err = 0.0
    names = []
    for fname in sorted(os.listdir(src)):
        spath = os.path.join(src, fname)
        dpath = os.path.join(dst, fname)
        if not fname.endswith('.safetensors'):
            with open(spath, 'rb') as f, open(dpath, 'wb') as g:
                g.write(f.read())
            continue
        tensors = load_file(spath)
        staged: dict[str, dict[str, torch.Tensor]] = {}
        out: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            suffix = next((s for s in ('qweight', 'scales', 'qzeros') if name.endswith(s)), None)
            if suffix is None:
                out[name] = tensor
                continue
            prefix = name[:-len(suffix)]
            staged.setdefault(prefix, {})[suffix] = tensor
        for prefix, shard in staged.items():
            out_shard, shard_err = convert_shard(shard, verify=verify)
            err = max(err, shard_err)
            names.append(prefix)
            for suffix, tensor in out_shard.items():
                out[f'{prefix}{suffix}'] = tensor
        save_file(out, dpath)
    return len(names), err
