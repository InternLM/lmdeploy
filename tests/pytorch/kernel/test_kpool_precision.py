# Copyright (c) OpenMMLab. All rights reserved.
"""KPool precision, packed-cache and causal-selection contracts."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.nn.kpool import kpool_packed_cache_views


@pytest.mark.parametrize('setting', [None, '0', '1'])
def test_startup_precision_selects_real_cache_request(setting: str | None) -> None:
    script = '''
import json
import torch
from lmdeploy.pytorch.engine.cache_engine.schema import BlockCacheGeometry, BlockCacheRequestContext
from lmdeploy.pytorch.nn.kpool import KPoolIndexer
indexer = KPoolIndexer(16, 32, 128, 2048, 8, 4, device='cpu')
request, = indexer.get_block_cache_requests(BlockCacheRequestContext(BlockCacheGeometry(64, 64)))
print(json.dumps({'dtype': str(indexer.cache_dtype), 'shape': request.shape}))
assert not torch.cuda.is_initialized()
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    env.pop('LMDEPLOY_GLM53_KPOOL_BF16', None)
    if setting is not None:
        env['LMDEPLOY_GLM53_KPOOL_BF16'] = setting
    process = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, check=True)
    result = json.loads(process.stdout.splitlines()[-1])
    expected = torch.bfloat16 if setting == '1' else torch.float8_e4m3fn
    assert result == {'dtype': str(expected), 'shape': [16, 1, 128 * expected.itemsize + 4]}


@pytest.mark.parametrize('activation_dtype', [torch.float16, torch.bfloat16])
def test_model_bf16_selection_matches_cache_precision(
    monkeypatch: pytest.MonkeyPatch, activation_dtype: torch.dtype
) -> None:
    from lmdeploy.pytorch.models import glm5_next as model_module

    query = torch.randn(3, 32, 128, dtype=activation_dtype)
    head_gate = torch.randn(3, 32)
    forbidden = Mock(side_effect=AssertionError('BF16 selection must skip FP8 preparation'))
    indexer = SimpleNamespace(cache_dtype=torch.bfloat16, softmax_scale=128**-0.5,
                              project_query=lambda _: query.unsqueeze(0),
                              project_head_gate=lambda _: head_gate.unsqueeze(0), quantize_fp8=forbidden)
    expected = torch.zeros(3, 2051, dtype=torch.int32)

    def select(actual_query: torch.Tensor, weights: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        torch.testing.assert_close(actual_query, query.to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(weights, head_gate * indexer.softmax_scale, rtol=0, atol=0)
        return expected

    context = SimpleNamespace(attn_tp_group=SimpleNamespace(rank=0),
                              dist_config=SimpleNamespace(attn_tp=1))
    monkeypatch.setattr(model_module, 'get_dist_manager',
                        lambda: SimpleNamespace(current_context=lambda: context))
    monkeypatch.setattr(model_module, 'kpool_rotate_query', forbidden)
    attention = SimpleNamespace(indexer=indexer, index_topk=2048, index_kpool=4,
                                _select_kpool_indices_prefill=select)
    result = model_module.Glm5NextSparseAttention._select_kpool_indices(
        attention, torch.empty(1, 3, 16, dtype=activation_dtype), torch.empty(1, 3, 8, dtype=activation_dtype),
        torch.empty(1, 16, 1, 260, dtype=torch.uint8), SimpleNamespace(is_decoding=False))
    assert result is expected
    forbidden.assert_not_called()


@pytest.mark.parametrize('dtype', [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize('page_size', [16, 64])
def test_packed_values_and_scales_share_the_declared_layout(dtype: torch.dtype, page_size: int) -> None:
    dim = 128
    packed = torch.zeros(3, page_size, 1, dim * dtype.itemsize + 4, dtype=torch.uint8)
    values, scales = kpool_packed_cache_views(packed, dim)
    expected = ((torch.arange(values.numel()) % 31 - 15) / 8).reshape(values.shape).to(dtype)
    values.copy_(expected)
    scales.fill_(2)
    value_bytes = page_size * dim * dtype.itemsize
    assert values.data_ptr() == packed.data_ptr()
    assert scales.data_ptr() - values.data_ptr() == value_bytes
    flat = packed.flatten(1)
    torch.testing.assert_close(flat[:, :value_bytes].view(dtype).reshape(values.shape).float(), expected.float())
    torch.testing.assert_close(flat[:, value_bytes:].view(torch.float32), torch.full((3, page_size), 2.0))


@pytest.mark.parametrize('dtype', [torch.float8_e4m3fn, torch.bfloat16])
def test_scoring_rejects_mixed_cache_precision(dtype: torch.dtype) -> None:
    from lmdeploy.pytorch.backends.cuda.kpool import kpool_score_paged_cuda

    other = torch.bfloat16 if dtype == torch.float8_e4m3fn else torch.float8_e4m3fn
    with pytest.raises(ValueError, match='precision'):
        kpool_score_paged_cuda(
            torch.empty(1, 32, 128, dtype=dtype), torch.ones(1, 32),
            torch.empty(1, 16, 1, 128 * other.itemsize + 4, dtype=torch.uint8),
            torch.ones(1, dtype=torch.int32), torch.zeros(1, 1, dtype=torch.int32), page_size=16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('mode', ['extend', 'decode'])
@pytest.mark.parametrize('dtype', [torch.float8_e4m3fn, torch.bfloat16])
def test_compression_precision_and_reference(mode: str, dtype: torch.dtype) -> None:
    from lmdeploy.pytorch.backends.cuda.kpool import kpool_compress_quantize_cuda
    from lmdeploy.pytorch.nn.kpool import kpool_compress, kpool_quantize_fp8

    torch.manual_seed(135)
    keys = torch.randn(37, 4, 128, device='cuda', dtype=torch.bfloat16)
    scores = torch.randn_like(keys)
    ape = torch.randn(4, 128, device='cuda')
    values, scales = kpool_compress_quantize_cuda(
        keys, scores, ape, mode=mode, round_scale=True, out_dtype=dtype)
    assert values.dtype == dtype
    if dtype == torch.bfloat16:
        probabilities = (scores.double() + ape.double()).softmax(dim=1)
        expected = (keys.double() * probabilities).sum(dim=1)
        torch.testing.assert_close(values.float(), expected.float(), rtol=0.005, atol=0.005)
        assert torch.equal(scales, torch.ones_like(scales))
    else:
        pooled = kpool_compress(keys, scores, ape, mode=mode)
        expected_values, expected_scales = kpool_quantize_fp8(pooled, round_scale=True)
        torch.testing.assert_close(scales, expected_scales, rtol=0.02, atol=0)
        torch.testing.assert_close(values.float() * scales, expected_values.float() * expected_scales,
                                   rtol=0.15, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype', [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize('page_size', [16, 64])
@pytest.mark.parametrize('rows', [0, 5])
def test_paged_scores_match_independent_reference(dtype: torch.dtype, page_size: int, rows: int) -> None:
    from lmdeploy.pytorch.backends.cuda.kpool import kpool_score_paged_cuda
    from lmdeploy.pytorch.nn.kpool import kpool_quantize_fp8

    if dtype == torch.float8_e4m3fn and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip('FP8 dot requires Hopper or newer')
    torch.manual_seed(136)
    dim, heads = 128, 32
    packed = torch.empty(4, page_size, 1, dim * dtype.itemsize + 4, device='cuda', dtype=torch.uint8)
    values, scales = kpool_packed_cache_views(packed, dim)
    original = torch.randn(4, page_size, dim, device='cuda', dtype=torch.bfloat16)
    query = torch.randn(rows, heads, dim, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(rows, heads, device='cuda') * (heads * dim)**-0.5
    if dtype == torch.bfloat16:
        values.copy_(original)
        scales.fill_(1)
    else:
        kv, ks = kpool_quantize_fp8(original, round_scale=True)
        values.copy_(kv)
        scales.copy_(ks)
        query, query_scale = kpool_quantize_fp8(query, round_scale=True)
        weights = (weights * query_scale.squeeze(-1)).contiguous()
    table = torch.tensor([[2, 0], [0, 3], [3, 1], [1, 2], [2, 3]], device='cuda', dtype=torch.int32)[:rows]
    lengths = torch.tensor([0, 1, page_size - 1, page_size + 1, page_size * 2],
                           device='cuda', dtype=torch.int32)[:rows]
    logits = kpool_score_paged_cuda(query, weights, packed, lengths, table, page_size=page_size)
    assert logits.shape == (rows, page_size * 2)
    keys = values[table.long()].reshape(rows, 2 * page_size, dim).double()
    key_scales = scales[table.long()].reshape(rows, 2 * page_size).double()
    dots = torch.einsum('rhd,rkd->rhk', query.double(), keys).relu()
    expected = (dots * weights.double()[..., None]).sum(1) * key_scales
    mask = torch.arange(2 * page_size, device='cuda')[None] < lengths[:, None]
    actual, reference = logits[mask].double(), expected[mask]
    assert (actual - reference).square().sum() <= 5e-6 * (actual.square() + reference.square()).sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype', [torch.float8_e4m3fn, torch.bfloat16])
def test_decode_compresses_into_destination_cache(dtype: torch.dtype) -> None:
    from lmdeploy.pytorch.backends.cuda.kpool import kpool_compress_quantize_cuda, kpool_decode_update_cuda

    torch.manual_seed(137)
    keys = torch.randn(1, 1, 128, device='cuda', dtype=torch.bfloat16)
    scores = torch.randn_like(keys)
    tail_keys = torch.randn(1, 4, 128, device='cuda', dtype=torch.bfloat16)
    tail_scores = torch.randn_like(tail_keys)
    closed_keys = torch.cat((tail_keys[:, :3], keys), dim=1)
    closed_scores = torch.cat((tail_scores[:, :3], scores), dim=1)
    ape = torch.randn(4, 128, device='cuda')
    expected, expected_scales = kpool_compress_quantize_cuda(
        closed_keys, closed_scores, ape, mode='decode', round_scale=True, out_dtype=dtype)
    packed = torch.zeros(2, 16, 1, 128 * dtype.itemsize + 4, device='cuda', dtype=torch.uint8)
    kpool_decode_update_cuda(
        keys, scores, tail_keys, tail_scores, torch.tensor([0], device='cuda'),
        torch.tensor([3], device='cuda'), packed, torch.tensor([[1]], device='cuda', dtype=torch.int32), ape, 4, True)
    values, scales = kpool_packed_cache_views(packed, 128)
    torch.testing.assert_close(values[1, :1].float(), expected.float(), atol=0, rtol=0)
    torch.testing.assert_close(scales[1, :1], expected_scales, atol=0, rtol=0)
    assert not torch.any(packed[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('q_lengths', [(7, 9), (0, 17)])
def test_bf16_ragged_prefill_preserves_topk_and_causality(
    monkeypatch: pytest.MonkeyPatch, q_lengths: tuple[int, int]
) -> None:
    from lmdeploy.pytorch.backends.cuda import kpool as backend

    torch.manual_seed(138)
    kv_lengths = (2115, 2337)
    packed = torch.empty(80, 16, 1, 260, device='cuda', dtype=torch.uint8)
    values, scales = kpool_packed_cache_views(packed, 128)
    values.normal_()
    scales.fill_(1)
    blocks = torch.randperm(80, device='cuda').reshape(2, 40).int()
    rows = sum(q_lengths)
    query = torch.randn(rows, 32, 128, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(rows, 32, device='cuda') * (32 * 128)**-0.5
    # Force several chunks while exercising the real paged scorer and Top-K.
    monkeypatch.setattr(backend, '_get_max_score_rows', lambda *args, **kwargs: 7)
    result = backend.kpool_select_prefill_cuda(
        query, weights, packed, torch.tensor(q_lengths, device='cuda', dtype=torch.int32),
        torch.tensor(kv_lengths, device='cuda', dtype=torch.int32), blocks, 2560, 4, 2048)
    offset = 0
    for request, (q_len, kv_len) in enumerate(zip(q_lengths, kv_lengths)):
        positions = torch.arange(kv_len // 4, device='cuda')
        keys = values[blocks[request, positions // 16].long(), positions % 16].double()
        for local in range(q_len):
            row = offset + local
            seq_len = kv_len - q_len + local + 1
            tokens = result[row][result[row] >= 0]
            assert tokens.numel() == 2048 + seq_len % 4
            assert torch.unique(tokens).numel() == tokens.numel()
            assert torch.all((tokens >= 0) & (tokens < seq_len))
            groups = tokens[:2048].reshape(-1, 4)
            slots = torch.arange(4, device='cuda', dtype=groups.dtype).expand_as(groups)
            torch.testing.assert_close(groups % 4, slots)
            logits = (query[row].double() @ keys.T).relu()
            logits = (logits * weights[row].double()[:, None]).sum(0)[:seq_len // 4]
            cutoff = logits.topk(512).values[-1]
            assert torch.all(logits[groups[:, 0].long() // 4] >= cutoff - 1e-4)
        offset += q_len
