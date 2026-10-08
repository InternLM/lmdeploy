from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends.cuda import kpool as backend


@pytest.mark.parametrize('local_columns', [False, True])
def test_prefill_score_chunks_preserve_compact_pages_and_legacy_columns(monkeypatch, local_columns):
    rows, heads, width = 5, 4, 128
    query = torch.zeros(rows, heads, width, dtype=torch.float8_e4m3fn)
    weights = torch.ones(rows, heads)
    cache = torch.zeros(10, 16, 1, width + 4, dtype=torch.uint8)
    query_starts = torch.tensor([0, 0, 1250, 1250, 1250], dtype=torch.int32)
    lengths = torch.tensor([1249, 1250, 1498, 1499, 1500], dtype=torch.int32)
    metadata = (torch.tensor([1250, 1500]), torch.tensor([0, 1250]),
                lengths * 4, lengths, query_starts, query_starts + lengths)
    monkeypatch.setattr(backend, 'kpool_prefill_metadata', lambda *args: metadata)
    monkeypatch.setattr(backend, 'flatten_kv_cache', lambda *args, **kwargs: (
        torch.zeros(1, 4000, width, dtype=torch.uint8), torch.zeros(1, 4000, 4, dtype=torch.uint8)))
    monkeypatch.setattr(backend, '_mqa_has_local_columns', lambda: local_columns)
    row_budget = Mock(return_value=2)
    monkeypatch.setattr(backend, '_get_max_score_rows', row_budget)
    calls = []

    def score(query_chunk, flat_kv, weight_chunk, starts, ends, **kwargs):
        first = sum(count for count, *_ in calls)
        calls.append((query_chunk.size(0), starts.clone(), kwargs))
        columns = 1600 if local_columns else 4000
        return torch.arange(first, first + query_chunk.size(0)).float()[:, None].expand(-1, columns)

    monkeypatch.setattr(backend, '_get_deep_gemm', lambda: SimpleNamespace(fp8_mqa_logits=score))

    def select(logits, chunk_lengths, *, group_topk, row_starts, max_group_length):
        assert max_group_length == 1600
        if local_columns:
            assert row_starts is None
        else:
            torch.testing.assert_close(row_starts, calls[-1][1])
        return logits[:, :group_topk].to(torch.int32)

    monkeypatch.setattr(backend, 'kpool_select_groups_cuda', select)
    result = backend.kpool_select_prefill_cuda(
        query, weights, cache, torch.tensor([2, 3]), torch.tensor([5000, 6000]),
        torch.ones(2, 100, dtype=torch.int32), 16000, 4, 2048, return_groups=True)
    torch.testing.assert_close(result, torch.arange(rows, dtype=torch.int32)[:, None].expand(-1, 512))
    assert [count for count, *_ in calls] == [2, 2, 1]
    assert row_budget.call_args.args[0] == (1600 if local_columns else 4000)
    for _, _, kwargs in calls:
        assert kwargs == ({'clean_logits': False, 'max_seqlen_k': 1600}
                          if local_columns else {'clean_logits': False})


def test_sparse_attention_forwards_shared_decode_metadata():
    from lmdeploy.pytorch.distributed import DistConfig, DistContext, get_dist_manager
    from lmdeploy.pytorch.models.glm5_next import Glm5NextSparseAttention

    layer = Glm5NextSparseAttention.__new__(Glm5NextSparseAttention)
    torch.nn.Module.__init__(layer)
    layer.num_heads = 2
    layer._qkv_proj_unabsorbed = Mock(return_value=tuple(torch.ones(2, 2, 4) for _ in range(4)))
    layer.kpool_attention = SimpleNamespace(forward=Mock(return_value=torch.ones(2, 2, 4)))
    layer.o_proj = torch.nn.Identity()
    shared = object()
    context = DistContext.build(dist_config=DistConfig(tp=1))
    with get_dist_manager().context(context):
        layer.forward(torch.ones(1, 2, 8), (), kpool_metadata=shared)
    assert layer.kpool_attention.forward.call_args.kwargs['kpool_metadata'] is shared


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('page_size', [16, 64])
def test_decode_metadata_matches_page_geometry_and_replay(page_size, monkeypatch):
    from lmdeploy.pytorch.nn.kpool import kpool_pooled_block_offsets

    fake_deepgemm = SimpleNamespace(
        get_paged_mqa_logits_metadata=lambda lengths, page_size, sms: lengths.clone(),
        get_num_sms=lambda: 132)
    monkeypatch.setattr(backend, '_get_deep_gemm', lambda: fake_deepgemm)
    query_lengths = torch.tensor([3, 3, 3], device='cuda')
    kv_lengths = torch.tensor([65, 131, 0], device='cuda')
    blocks = torch.arange(36, device='cuda').reshape(3, 12)[:, ::2].int()
    metadata = SimpleNamespace(q_seqlens=query_lengths, kv_seqlens=kv_lengths, block_offsets=blocks)

    def compute():
        return backend.kpool_decode_metadata_cuda(metadata, 9, 4, page_size=page_size)

    for _ in range(3):
        compute()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = compute()
    for delta in [0, 4, 64]:
        kv_lengths.copy_(torch.tensor([65 + delta, 131 + delta, 0], device='cuda'))
        graph.replay()
        expected_sequence = (kv_lengths[:, None] - query_lengths[:, None]
                             + torch.arange(1, 4, device='cuda')).flatten()
        torch.testing.assert_close(result.seq_lens, expected_sequence)
        torch.testing.assert_close(result.group_lengths, expected_sequence.div(4, rounding_mode='floor'))
        expected_table = kpool_pooled_block_offsets(blocks.repeat_interleave(3, dim=0), 4, page_size)
        torch.testing.assert_close(result.block_table, expected_table)
        assert (result.schedule is None) == (page_size == 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('scale_fmt', [None, 'ue8m0'])
@pytest.mark.parametrize('width', [128, 768])
def test_glm_fused_activation_quantization_preserves_valid_rows(width, scale_fmt):
    from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import _quant_fp8_launcher
    from lmdeploy.pytorch.models.glm5_next import _GLM53_COMPACT_FP8_MOE_ACT

    torch.manual_seed(20261008)
    inputs = (torch.randn(4, 16, 2 * width, device='cuda') * 12).bfloat16()
    counts = torch.tensor([0, 1, 7, 16], dtype=torch.int32, device='cuda')
    output = torch.full((4, 16, width), 2, device='cuda', dtype=torch.float8_e4m3fn)
    scales = torch.full((4, 16, width // 128), -1.0, device='cuda')

    def compute():
        _GLM53_COMPACT_FP8_MOE_ACT.masked_post_quant(inputs, output, scales, 128, counts, scale_fmt=scale_fmt)

    for _ in range(3):
        compute()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compute()
    for active_counts in ([0, 1, 7, 16], [16, 7, 1, 0]):
        counts.copy_(torch.tensor(active_counts, dtype=torch.int32, device='cuda'))
        inputs.normal_(0, 12)
        output.fill_(2)
        scales.fill_(-1)
        graph.replay()
        for expert, count in enumerate(active_counts):
            if count:
                activation = _GLM53_COMPACT_FP8_MOE_ACT(inputs[expert, :count])
                reference = torch.empty_like(output[expert, :count])
                reference_scales = torch.empty_like(scales[expert, :count])
                _quant_fp8_launcher(activation, 128, reference, reference_scales, scale_fmt=scale_fmt)
                torch.testing.assert_close(output[expert, :count].float(), reference.float(), rtol=0, atol=0)
                torch.testing.assert_close(scales[expert, :count], reference_scales, rtol=0, atol=0)
            assert (output[expert, count:].float() == 2).all()
            assert (scales[expert, count:] == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_draft_token_cache_fusion_preserves_ragged_and_rejected_history():
    from lmdeploy.pytorch.kernels.cuda.kpool import gather_kpool_token_tail, write_kpool_token_cache

    torch.manual_seed(20261008)
    cache = torch.randn(8, 64, 2, 128, device='cuda', dtype=torch.bfloat16)
    blocks = torch.tensor([[1, 3], [2, 4], [5, 6]], dtype=torch.int32, device='cuda')
    query_lengths = torch.tensor([3, 0, 5], device='cuda')
    history = torch.tensor([63, 4, 65], device='cuda')
    starts = torch.tensor([0, 3, 3, 8], device='cuda')
    keys = torch.randn(8, 128, device='cuda', dtype=torch.bfloat16)
    scores = torch.randn_like(keys)
    original = cache.clone()
    reference = cache.clone()
    for request, count in enumerate(query_lengths.tolist()):
        for offset in range(count):
            position = int(history[request]) + offset
            block = int(blocks[request, position // 64])
            source = int(starts[request]) + offset
            reference[block, position % 64, 0] = keys[source]
            reference[block, position % 64, 1] = scores[source]
    write_kpool_token_cache(cache, keys, scores, blocks, query_lengths, history + query_lengths, starts)
    torch.testing.assert_close(cache, reference, rtol=0, atol=0)
    for accepted in [0, 1, 3]:
        accepted_history = history + torch.minimum(query_lengths, torch.full_like(query_lengths, accepted))
        tails = gather_kpool_token_tail(cache, blocks, query_lengths, accepted_history + query_lengths, 4)
        for request, length in enumerate(accepted_history.tolist()):
            remainder = length % 4
            expected = torch.zeros(4, 2, 128, device='cuda', dtype=cache.dtype)
            for slot in range(remainder):
                position = length - remainder + slot
                block = int(blocks[request, position // 64])
                expected[slot] = reference[block, position % 64]
            torch.testing.assert_close(tails[0][request], expected[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(tails[1][request], expected[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(tails[2], torch.arange(3, device='cuda'))
    torch.testing.assert_close(cache[0], original[0], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_hc_fp32_preparation_preserves_bfloat16_rounding():
    from lmdeploy.pytorch.nn import HcPrePost

    torch.manual_seed(20261008)
    hidden = torch.randn(1, 32, 128, dtype=torch.bfloat16, device='cuda')
    residual = torch.randn(1, 32, 4, 128, dtype=torch.bfloat16, device='cuda')
    post = torch.randn(1, 32, 4, device='cuda')
    comb = torch.randn(1, 32, 4, 4, device='cuda')
    operator = HcPrePost(4)
    expected = operator.post_expand(hidden, residual, post, comb)
    actual, prepared = operator.post_expand_with_fp32(hidden, residual, post, comb)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(prepared, actual.float(), rtol=0, atol=0)
