# Copyright (c) OpenMMLab. All rights reserved.
"""NoPE packed-cache round trips and real sparse FlashMLA numerical tests."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.backends.cuda.attention.default import TritonAttentionMetadata
from lmdeploy.pytorch.backends.cuda.attention.mla import _build_flash_mla_metadata
from lmdeploy.pytorch.backends.cuda.attention.sparse_mla import FlashMLASparseImpl
from lmdeploy.pytorch.engine.executor import _finalize_sparse_mla_cache_policy


@pytest.mark.parametrize('quant_policy', [QuantPolicy.NONE, QuantPolicy.FP8, QuantPolicy.INT8, QuantPolicy.FP8_E5M2])
@pytest.mark.parametrize('shared', [False, True])
def test_custom_indexer_cache_policy(quant_policy, shared):
    model = SimpleNamespace(mla_index_topk=None, mla_kv_cache_dtype='bfloat16', supports_mla_fp8_cache=True)
    cache = SimpleNamespace(quant_policy=quant_policy)
    models = [model, SimpleNamespace(mla_index_topk=None, mla_kv_cache_dtype=None)] if shared else [model]
    if quant_policy not in (QuantPolicy.NONE, QuantPolicy.FP8):
        with pytest.raises(ValueError, match='Sparse MLA does not support'):
            _finalize_sparse_mla_cache_policy(models, cache)
        return
    _finalize_sparse_mla_cache_policy(models, cache)
    assert cache.quant_policy == QuantPolicy.NONE
    assert model.mla_kv_cache_dtype == ('fp8_ds_mla' if quant_policy == QuantPolicy.FP8 else 'bfloat16')
    assert model.mla_index_topk is None


@pytest.mark.parametrize('draft', [False, True])
def test_glm_target_and_mtp_opt_into_packed_cache(monkeypatch, draft):
    from lmdeploy.hf_configs.configuration_glm5_next import Glm5NextConfig
    from lmdeploy.pytorch.configurations import deepseek_v2
    from lmdeploy.pytorch.configurations.glm5_next import Glm5NextModelConfigBuilder

    monkeypatch.setattr(deepseek_v2, 'flash_mla_available', lambda: True)
    model = Glm5NextModelConfigBuilder.build(Glm5NextConfig(), tp=8,
                                             is_draft_model=draft, num_spec_tokens=5,
                                             spec_method='deepseek_mtp')
    assert model.mla_kv_cache_dtype == 'bfloat16'
    cache = SimpleNamespace(quant_policy=QuantPolicy.FP8)
    _finalize_sparse_mla_cache_policy([model], cache)
    assert model.use_mla_fp8_cache
    assert model.mla_index_topk is None
    assert cache.quant_policy == QuantPolicy.NONE


@pytest.mark.parametrize('query_len', [1, 6])
@pytest.mark.parametrize('dcp_size', [1, 2])
def test_sparse_scheduler_keeps_query_heads_separate_from_query_length(monkeypatch, query_len, dcp_size):
    import sys

    from lmdeploy.pytorch import distributed

    get_metadata = Mock(return_value=(torch.zeros(1), torch.zeros(1)))
    monkeypatch.setitem(sys.modules, 'flash_mla', SimpleNamespace(get_mla_metadata=get_metadata))
    monkeypatch.setattr(distributed, 'get_dcp_world_rank', lambda: (dcp_size, 0))
    _build_flash_mla_metadata(torch.tensor([2051], dtype=torch.int32), num_attention_heads=8,
                             decoding_query_len=query_len, is_fp8_kvcache=True, index_topk=2112)
    assert get_metadata.call_args.args[1] == 8 * dcp_size * query_len
    assert get_metadata.call_args.kwargs['num_heads_q'] == 8 * dcp_size
    assert get_metadata.call_args.kwargs['topk'] == 2112


def _impl(heads=8, head_dim=512):
    return FlashMLASparseImpl(mla_index_topk=2048, mla_index_tail=3 if head_dim == 512 else 0,
                             num_heads=heads, num_kv_heads=1, head_size=head_dim,
                             v_head_size=512, scale=0.0625)


def _metadata(lengths, query_lengths, blocks, decoding=False):
    lengths = torch.tensor(lengths, device='cuda', dtype=torch.int32)
    query_lengths = torch.tensor(query_lengths, device='cuda', dtype=torch.int32)
    cu_q = torch.nn.functional.pad(query_lengths.cumsum(0, dtype=torch.int32), (1, 0))
    cu_k = torch.nn.functional.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    return TritonAttentionMetadata(is_decoding=decoding, block_offsets=blocks,
                                  q_start_loc=cu_q[:-1], q_seqlens=query_lengths,
                                  kv_start_loc=cu_k[:-1], kv_seqlens=lengths,
                                  cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                  max_q_seqlen=int(query_lengths.max()), max_kv_seqlen=int(lengths.max()),
                                  kv_flatten_size=int(lengths.sum()))


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@requires_cuda
@pytest.mark.parametrize('query_len', [1, 6, 79])
@pytest.mark.parametrize('dcp_size,dcp_rank', [(1, 0), (2, 0), (2, 1)])
def test_nope_writer_only_clears_written_slots(query_len, dcp_size, dcp_rank):
    torch.manual_seed(17)
    impl = _impl()
    impl.dcp_world_size, impl.dcp_rank = dcp_size, dcp_rank
    # Dirty pages expose stale tails after page reuse. The second row is a
    # CUDA Graph padding request and must not modify even block zero.
    raw = torch.full((8, 64, 1, 656), 0x5a, dtype=torch.uint8, device='cuda')
    cache = raw.view(torch.float8_e4m3fn)
    blocks = torch.tensor([[3, 1, 5, 2], [0, 0, 0, 0]], device='cuda', dtype=torch.int32)
    history = 61
    meta = _metadata([history + query_len, 0], [query_len, query_len], blocks, decoding=query_len <= 6)
    key = torch.randn(2 * query_len, 1, 512, dtype=torch.bfloat16, device='cuda')
    impl._fill_kv_cache_impl(key, key, cache, cache[..., :0], meta, query_len)
    written = torch.zeros((8, 64), dtype=torch.bool, device='cuda')
    for token in range(history, history + query_len):
        if token % dcp_size != dcp_rank:
            continue
        local = token // dcp_size
        page = int(blocks[0, local // 64])
        slot = local % 64
        written[page, slot] = True
        assert torch.count_nonzero(raw[page, slot, :, 528:]) == 0
        scale = cache[page, slot, :, 512:528].view(torch.float32)
        restored = (cache[page, slot, :, :512].float().reshape(1, 4, 128) * scale[..., None]).reshape(1, 512)
        torch.testing.assert_close(restored, key[token - history].float(), rtol=0.13, atol=0.13)
    assert torch.all(raw[~written] == 0x5a)


@requires_cuda
@pytest.mark.parametrize('head_dim', [512, 576])
@pytest.mark.parametrize('layout', ['shd', 'hsd'])
def test_packed_cache_flatten_logical_width(head_dim, layout):
    torch.manual_seed(23)
    impl = _impl(head_dim=head_dim)
    cache = torch.empty(4, 64, 1, 656, device='cuda', dtype=torch.float8_e4m3fn)
    blocks = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32, device='cuda')
    key = torch.randn(87, 1, head_dim, device='cuda', dtype=torch.bfloat16)
    meta = _metadata([70, 17], [70, 17], blocks)
    impl._fill_kv_cache_impl(key, key[..., :512], cache, cache[..., :0], meta, 70)
    out, value = impl._flatten_prefill_kv_cache(cache, cache[..., :0], meta, torch.bfloat16, layout)
    assert out.size(-1) == head_dim
    assert out.is_contiguous()
    assert value.size(-1) == 512
    actual = out if layout == 'shd' else out.transpose(0, 1)
    torch.testing.assert_close(actual[:87].float(), key.float(), rtol=0.13, atol=0.13)
    assert torch.count_nonzero(actual[87:]) == 0


@requires_cuda
@pytest.mark.parametrize('heads', [8, 16, 64])
@pytest.mark.parametrize('query_len', [1, 6])
@pytest.mark.parametrize('head_dim', [512, 576])
def test_nope_sparse_decode_matches_dequantized_reference(heads, query_len, head_dim):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip('656-byte sparse cache test targets SM90')
    import flash_mla

    torch.manual_seed(31)
    impl = _impl(heads, head_dim)
    # Exercise page mapping and all 2051 KPool indices, including live tail.
    length, batch = (2051 if head_dim == 512 else 2048), 2
    pages = (length + 63) // 64
    blocks = torch.randperm(batch * pages, device='cuda', dtype=torch.int32).reshape(batch, pages)
    cache = torch.empty(batch * pages, 64, 1, 656, device='cuda', dtype=torch.float8_e4m3fn)
    key = torch.randn(batch * length, 1, head_dim, device='cuda', dtype=torch.bfloat16)
    prefill = _metadata([length] * batch, [length] * batch, blocks)
    impl._fill_kv_cache_impl(key, key[..., :512], cache, cache[..., :0], prefill, length)
    flat, _ = impl._flatten_prefill_kv_cache(cache, cache[..., :0], prefill, torch.bfloat16, 'shd')
    meta = _metadata([length] * batch, [query_len] * batch, blocks, decoding=True)
    meta.tile_scheduler_metadata, meta.num_splits = flash_mla.get_mla_metadata()
    query = torch.randn(batch * query_len, heads, head_dim, device='cuda', dtype=torch.bfloat16)
    indices = torch.arange(length, dtype=torch.int32, device='cuda').repeat(batch * query_len, 1)
    # Verification rows have different causal limits.
    for row in range(batch * query_len):
        indices[row, length - query_len + row % query_len + 1:] = -1
    result, lse = impl._decoding_sparse_fp8(query, cache, indices, meta, return_lse=True)
    refs = []
    for row in range(batch * query_len):
        end = length - query_len + row % query_len + 1
        k = flat[(row // query_len) * length: (row // query_len) * length + end, 0].float()
        scores = query[row].float() @ k.T * impl.scale
        refs.append(torch.softmax(scores, dim=-1) @ k[:, :512])
        torch.testing.assert_close(lse[row], torch.logsumexp(scores, dim=-1), atol=0.02, rtol=0.01)
    torch.testing.assert_close(result.float(), torch.stack(refs), atol=0.003, rtol=0.03)
    expected_width = 2112 if head_dim == 512 else 2048
    assert meta.tile_scheduler_metadata.config.topk == expected_width
    assert impl.get_step_metadata_provider().index_topk == expected_width
    # Warmup followed by capture and replay uses the same padded geometry.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = impl._decoding_sparse_fp8(query, cache, indices, meta)
    graph.replay()
    torch.testing.assert_close(captured, result)
