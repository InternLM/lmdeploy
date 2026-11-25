# Copyright (c) OpenMMLab. All rights reserved.
"""Real GQA cache/attention/collective path versus independent dense attention.

FP8 prefill intentionally rounds only the cached prefix in DCP. The reference models that contract separately from the
all-quantized, non-DCP comparison.
"""
from dataclasses import replace
from datetime import timedelta
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from lmdeploy.messages import QuantPolicy
from tests.pytorch.distributed_test_utils import spawn_with_timeout


def _round_cache(value, dtype, scale):
    if dtype == value.dtype:
        return value
    limits = torch.finfo(dtype)
    return (value.float().div(scale).clamp(limits.min, limits.max).to(dtype).float() * scale).to(value.dtype)


def _avoid_e5m2_midpoints(value, scale):
    """Keep exact cache-ownership checks independent of compiler tie rounding.

    Triton's pre-SM89 E5M2 software conversion rounds midpoints away from zero; native conversion and PyTorch use ties-
    to-even. Move only those inputs one BF16 ULP toward zero, retaining ordinary FP8 rounding elsewhere.
    """
    magnitude = value.float().abs() / scale
    spacing = torch.exp2(torch.floor(torch.log2(magnitude)) - 2).clamp_min(2**-16)
    midpoint = (magnitude / spacing).frac() == 0.5
    return torch.where(midpoint, torch.nextafter(value, torch.zeros_like(value)), value)


def test_e5m2_fixture_agrees_with_native_and_software_rounding():
    values = (torch.arange(-128, 129) / 128).to(torch.bfloat16)
    for scale in (0.25, 0.5):
        adjusted = _avoid_e5m2_midpoints(values, scale)
        scaled = adjusted.float() / scale
        # Independent emulation of Triton's pre-SM89 FP16 -> E5M2 lowering.
        half_bits = scaled.half().view(torch.int16).int() & 0xffff
        software = (((half_bits & 0xfffe) + 0x80) >> 8).byte()
        native = scaled.to(torch.float8_e5m2).view(torch.uint8)
        torch.testing.assert_close(software, native, rtol=0, atol=0)
        changed = adjusted != values
        assert changed.any()
        torch.testing.assert_close(adjusted[changed],
                                   torch.nextafter(values[changed], torch.zeros_like(values[changed])), rtol=0, atol=0)


def _dense_reference(query, key, value, history):
    """FP32 causal attention with one KV head; no production DCP helpers."""
    positions = torch.arange(key.size(0), device=query.device)
    ends = history + torch.arange(query.size(0), device=query.device)
    scores = torch.einsum('qhd,khd->hqk', query.float(), key.expand(-1, query.size(1), -1).float())
    scores *= query.size(-1)**-0.5
    scores.masked_fill_(positions[None, None, :] > ends[None, :, None], -torch.inf)
    return torch.einsum('hqk,khd->qhd', scores.softmax(-1), value.expand(-1, query.size(1), -1).float())


def _gqa_worker(rank, rendezvous, quant_policy, world_size=2, use_fa3=False):
    from lmdeploy.pytorch.backends.cp_utils import build_dcp_prefix_chunks, get_dcp_local_seq_lens
    from lmdeploy.pytorch.backends.cuda.attention.cp import DCPAttentionImpl
    from lmdeploy.pytorch.backends.cuda.attention.default import TritonAttentionImpl, TritonAttentionMetadata
    from lmdeploy.pytorch.config import DistConfig
    from lmdeploy.pytorch.distributed import DistContext, DistGroup, get_dist_manager

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=60))
    context = DistContext.build(rank, DistConfig(tp=world_size, dcp=2))
    dcp_rank = context.dcp_group.rank
    manager = get_dist_manager()
    try:
        kwargs = dict(num_heads=2, head_size=64, num_kv_heads=1, v_head_size=64,
                      sliding_window=(-1, -1), causal=True)
        with manager.context(context):
            dcp_impl = DCPAttentionImpl(**kwargs, use_fa3=use_fa3)
            if use_fa3:
                dcp_impl._fa3_prefill = Mock(wraps=dcp_impl._fa3_prefill)
        with manager.context(DistContext(dist_config=DistConfig(tp=2), dcp_group=DistGroup())):
            reference_impl = TritonAttentionImpl(**kwargs)
        device = torch.device('cuda', rank)
        torch.manual_seed(5172 + rank // 2)
        # Four global Q heads; each rank owns two and shares the single KV head.
        query = torch.rand(2, 104, 4, 64, device=device, dtype=torch.bfloat16) * 2 - 1
        key = torch.rand(2, 104, 1, 64, device=device, dtype=torch.bfloat16) * 2 - 1
        value = torch.rand_like(key) * 2 - 1
        query = query[:, :, dcp_rank * 2:(dcp_rank + 1) * 2]
        dtype = {QuantPolicy.NONE: torch.bfloat16, QuantPolicy.FP8: torch.float8_e4m3fn,
                 QuantPolicy.FP8_E5M2: torch.float8_e5m2}[quant_policy]
        k_scale = torch.tensor([0.25], device=device)
        v_scale = torch.tensor([0.5], device=device)
        if quant_policy == QuantPolicy.FP8_E5M2:
            key = _avoid_e5m2_midpoints(key, k_scale)
            value = _avoid_e5m2_midpoints(value, v_scale)
        # Precision tradeoff bounds for this fixed, bounded workload. These are
        # separate from the tighter error against the exact mixed-precision oracle.
        quant_tol = {QuantPolicy.NONE: 0.02, QuantPolicy.FP8: 0.04, QuantPolicy.FP8_E5M2: 0.08}[quant_policy]
        block_size = 16
        # Non-contiguous physical page indices exercise the page table itself.
        blocks = torch.arange(1, 17, device=device).flip(0).reshape(2, 8).contiguous()
        full_outputs = None
        for mode in ('full', 'chunked', 'prefix_reuse', 'mixed_prefix', 'decode_one', 'decode_many'):
            cache_shape = (17, block_size, 1, 64)
            dcp_k, dcp_v = [torch.zeros(cache_shape, device=device, dtype=dtype) for _ in range(2)]
            ref_k, ref_v = [torch.zeros(cache_shape, device=device, dtype=dtype) for _ in range(2)]
            if mode == 'chunked':
                steps = [(37, 64), (71, 97)]
            elif mode == 'prefix_reuse':
                # Both prefixes end on reusable DCP logical-page boundaries.
                steps = [(32, 64), (71, 97)]
            elif mode == 'mixed_prefix':
                # One request has no history while the other reuses its prefix.
                steps = [(32, 32), (71, 97)]
            elif mode.startswith('decode'):
                steps = [(71, 97), (72, 98) if mode == 'decode_one' else (74, 100)]
            else:
                steps = [(71, 97)]
            history = (0, 0)
            for step, ends in enumerate(steps):
                decoding = mode.startswith('decode') and step == 1
                q_lens = torch.tensor([end - start for start, end in zip(history, ends)],
                                      dtype=torch.int32, device=device)
                kv_lens = torch.tensor(ends, dtype=torch.int32, device=device)
                cu_q = torch.nn.functional.pad(q_lens.cumsum(0, dtype=torch.int32), (1, 0))
                cu_k = torch.nn.functional.pad(kv_lens.cumsum(0, dtype=torch.int32), (1, 0))
                slices = [(i, start, end) for i, (start, end) in enumerate(zip(history, ends))]
                q, k, v = [torch.cat([tensor[i, start:end] for i, start, end in slices])
                           for tensor in (query, key, value)]
                meta = TritonAttentionMetadata(
                    is_decoding=decoding, block_offsets=blocks, q_start_loc=cu_q[:-1], q_seqlens=q_lens,
                    kv_start_loc=cu_k[:-1], kv_seqlens=kv_lens, quant_policy=quant_policy,
                    kv_flatten_size=sum(ends), cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_kv_seqlen=max(ends), max_q_seqlen=max(end - start for _, start, end in slices),
                    dcp_local_kv_seqlens=get_dcp_local_seq_lens(kv_lens, (2, dcp_rank)),
                    dcp_prefix_chunks=build_dcp_prefix_chunks(
                        prefix_lens=kv_lens - q_lens, prefix_limit=max(history),
                        block_size=block_size, kv_width=128, dcp_world_rank=(2, dcp_rank)))
                # A reused prefix uses the existing cache without writing prefix tokens again.
                with manager.context(context):
                    output = dcp_impl.forward(q.clone(), k, v, dcp_k, dcp_v, meta,
                                              k_scales_zeros=k_scale, v_scales_zeros=v_scale, inplace=False)
                # Standard Triton decode accepts one query token. Verify a
                # multi-token decode against its causal prefill path instead.
                ref_meta = replace(meta, is_decoding=False) if decoding and meta.max_q_seqlen > 1 else meta
                reference = reference_impl.forward(q.clone(), k, v, ref_k, ref_v, ref_meta,
                                                    k_scales_zeros=k_scale, v_scales_zeros=v_scale, inplace=False)
                oracle, quantized_oracle = [], []
                for i, start, end in slices:
                    rounded_k = _round_cache(key[i, :end], dtype, k_scale)
                    rounded_v = _round_cache(value[i, :end], dtype, v_scale)
                    quantized_oracle.append(_dense_reference(query[i, start:end], rounded_k, rounded_v, start))
                    mixed_k, mixed_v = rounded_k.clone(), rounded_v.clone()
                    if not decoding:
                        mixed_k[start:], mixed_v[start:] = key[i, start:end], value[i, start:end]
                    oracle.append(_dense_reference(query[i, start:end], mixed_k, mixed_v, start))
                    # Verify cache ownership/rounding independently of the attention result.
                    positions = torch.arange(dcp_rank, end, 2, device=device)
                    local = positions // 2
                    page_ids = blocks[i, local // block_size]
                    for cache, expected, scale in ((dcp_k, rounded_k, k_scale), (dcp_v, rounded_v, v_scale)):
                        actual = cache.float()[page_ids, local % block_size]
                        if quant_policy != QuantPolicy.NONE:
                            actual = (actual * scale).to(torch.bfloat16).float()
                        torch.testing.assert_close(actual, expected[positions].float(), rtol=0, atol=0)
                torch.testing.assert_close(output.float(), torch.cat(oracle), rtol=0.02, atol=0.02)
                torch.testing.assert_close(reference.float(), torch.cat(quantized_oracle), rtol=0.02, atol=0.02)
                torch.testing.assert_close(output.float(), reference.float(), rtol=quant_tol, atol=quant_tol)
                if rank == 0:
                    oracle_error = (output.float() - torch.cat(oracle)).abs().max().item()
                    native_error = (output.float() - reference.float()).abs().max().item()
                    print(f'GQA {quant_policy.name} {mode} step={step}: '
                          f'max_abs_oracle={oracle_error:.6g} max_abs_non_dcp={native_error:.6g}', flush=True)
                if mode == 'full':
                    full_outputs = list(output.split([71, 97]))
                elif not decoding and step == 1:
                    full_slice = torch.cat([full_outputs[i][start:end] for i, start, end in slices])
                    torch.testing.assert_close(output, full_slice, rtol=quant_tol, atol=quant_tol)
                # Reuse the first request slot for a fresh request. Every
                # scheduled request still has at least one query token.
                history = (0, ends[1]) if mode == 'mixed_prefix' and step == 0 else ends
        if use_fa3:
            calls = dcp_impl._fa3_prefill.call_args_list
            assert any(call.kwargs['return_lse'] for call in calls), 'FA3 prefix merge was not exercised'
            assert any(not call.kwargs['return_lse'] for call in calls), 'FA3 full prefill was not exercised'
    finally:
        context.close()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two idle CUDA GPUs')
@pytest.mark.parametrize('quant_policy', [QuantPolicy.NONE, QuantPolicy.FP8, QuantPolicy.FP8_E5M2])
@pytest.mark.parametrize('use_fa3', [False, True], ids=['triton', 'fa3'])
def test_gqa_dcp_cache_and_attention_reference(tmp_path, quant_policy, use_fa3):
    if quant_policy == QuantPolicy.FP8 and any(torch.cuda.get_device_capability(i)[0] < 9 for i in range(2)):
        pytest.skip('Triton float8_e4m3fn conversion requires device with cc>=9.0')
    if use_fa3:
        if any(torch.cuda.get_device_capability(i)[0] < 9 for i in range(2)):
            pytest.skip('FA3 requires device with cc>=9.0')
        pytest.importorskip('flash_attn_interface')
    spawn_with_timeout(_gqa_worker, args=(f'file://{tmp_path}/gqa', quant_policy, 2, use_fa3))


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason='requires eight CUDA GPUs')
def test_gqa_tp8_dcp2_groups(tmp_path):
    # Each DCP pair shares one KV head; different pairs use different K/V.
    spawn_with_timeout(_gqa_worker, args=(f'file://{tmp_path}/gqa_tp8', QuantPolicy.NONE, 8), nprocs=8)
