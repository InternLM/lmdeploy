# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends.attention import PagedAttentionBuildSpec
from lmdeploy.pytorch.backends.cuda import attention as attention_module
from lmdeploy.pytorch.backends.cuda.attention import mla as mla_module
from lmdeploy.pytorch.backends.cuda.attention import sparse_mla as sparse_mla_module
from lmdeploy.pytorch.backends.cuda.attention.sparse_mla import (
    FlashMLAIndexMapper,
    FlashMLASparseImpl,
)
from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend


def _disable_dynamic_compile(monkeypatch):
    monkeypatch.setattr(sparse_mla_module, '_try_dynamic_compile', lambda func, *args, **kwargs: func)


def test_flash_mla_build_spec_selects_sparse_impl(monkeypatch):
    dense_output = object()
    sparse_output = object()
    dense_impl = Mock(return_value=dense_output)
    sparse_impl = Mock(return_value=sparse_output)
    monkeypatch.setattr(attention_module, '_enable_fa3', lambda *args: False)
    monkeypatch.setattr(attention_module, 'use_fa3', True)
    monkeypatch.setattr(mla_module, 'FlashMLAImpl', dense_impl)
    monkeypatch.setattr(sparse_mla_module, 'FlashMLASparseImpl', sparse_impl)
    spec_kwargs = dict(
        num_heads=64,
        head_dim=576,
        scale=None,
        num_kv_heads=1,
        v_head_dim=576,
        alibi=False,
        sliding_window=None,
        logit_softcapping=0.0,
        causal=True,
        use_flash_mla=True,
        learnable_sink=False,
        block_sparse_size=1,
    )

    assert CudaOpsBackend.build_op(
        PagedAttentionBuildSpec(mla_index_topk=None, **spec_kwargs)) is dense_output
    assert CudaOpsBackend.build_op(
        PagedAttentionBuildSpec(mla_index_topk=2048, **spec_kwargs)) is sparse_output
    assert sparse_impl.call_args.kwargs['use_fa3'] is True


def test_flash_mla_decode_index_mapping(monkeypatch):
    _disable_dynamic_compile(monkeypatch)
    mapper = FlashMLAIndexMapper()
    block_offsets = torch.tensor([[100, 101, 102], [200, 201, 202]])

    nsa_indices = torch.tensor([[0, 17, -1], [32, 1, 16], [0, 33, 47], [32, 1, 16]])
    output = mapper.map_paged_decode(nsa_indices, block_offsets, max_q_seqlen=2, block_size=16)
    expected = torch.tensor([[[1600, 1617, -1], [1632, 1601, 1616]],
                             [[3200, 3233, 3247], [3232, 3201, 3216]]])
    assert torch.equal(output, expected)

    nsa_indices = torch.tensor([[0, 17], [32, -1]])
    output = mapper.map_paged_decode(nsa_indices, block_offsets, max_q_seqlen=1, block_size=16)
    assert torch.equal(output, torch.tensor([[[1600, 1617]], [[3232, -1]]]))


def test_flash_mla_decode_index_mapping_caches_query_modes(monkeypatch):
    compile_func = Mock(side_effect=lambda func, *args, **kwargs: func)
    monkeypatch.setattr(sparse_mla_module, '_try_dynamic_compile', compile_func)
    mapper = FlashMLAIndexMapper()
    block_offsets = torch.tensor([[100, 101, 102], [200, 201, 202]])
    single_indices = torch.tensor([[0, 17], [32, -1]])
    multi_indices = torch.tensor([[0, 17], [32, 1], [0, 33], [32, 1]])

    mapper.map_paged_decode(single_indices, block_offsets, max_q_seqlen=1, block_size=16)
    mapper.map_paged_decode(single_indices, block_offsets, max_q_seqlen=1, block_size=16)
    mapper.map_paged_decode(multi_indices, block_offsets, max_q_seqlen=2, block_size=16)
    mapper.map_paged_decode(multi_indices.repeat_interleave(2, dim=0), block_offsets,
                            max_q_seqlen=4, block_size=16)

    assert compile_func.call_count == 2


def test_bf16_sparse_decode_uses_strided_cache_view(monkeypatch):
    _disable_dynamic_compile(monkeypatch)
    impl = object.__new__(FlashMLASparseImpl)
    impl.index_mapper = FlashMLAIndexMapper()
    impl._flash_mla_sparse_forward = Mock(return_value=torch.empty(4, 64, 512, dtype=torch.bfloat16))

    query = torch.empty(4, 64, 576, dtype=torch.bfloat16)
    block_size = 16
    block_elements = block_size * 576
    storage = torch.empty(3, block_elements + 128, dtype=torch.bfloat16)
    k_cache = storage[:, :block_elements].view(3, block_size, 1, 576)
    nsa_indices = torch.tensor([[0, 17], [32, -1], [0, 33], [32, 1]])
    metadata = SimpleNamespace(is_decoding=True,
                               q_seqlens=torch.tensor([2, 2]),
                               block_offsets=torch.tensor([[1, 2, 0], [2, 0, 1]]))

    impl._decoding_sparse_bf16(query, k_cache, nsa_indices, metadata)

    sparse_query, storage_k, global_indices = impl._flash_mla_sparse_forward.call_args.args
    assert sparse_query is query
    assert storage_k.untyped_storage().data_ptr() == k_cache.untyped_storage().data_ptr()
    assert storage_k.stride() == (64, 576, 1)
    expected = torch.tensor([[[146, 301]], [[0, -1]], [[292, 155]], [[146, 301]]])
    assert torch.equal(global_indices, expected)


def test_bf16_sparse_flashmla_uses_third_return_value_as_lse(monkeypatch):
    impl = object.__new__(FlashMLASparseImpl)
    impl.scale = 1.0
    output = torch.empty(2, 64, 512, dtype=torch.bfloat16)
    max_logits = torch.full((2, 64), 7.0)
    lse = torch.full((2, 64), 11.0)

    def fake_sparse_fwd(*args, **kwargs):
        return output, max_logits, lse

    monkeypatch.setattr(impl, '_get_flash_mla_sparse_fwd',
                        lambda: fake_sparse_fwd)
    actual_output, actual_lse = impl._flash_mla_sparse_forward(
        torch.empty(2, 8, 576, dtype=torch.bfloat16),
        torch.empty(4, 1, 576, dtype=torch.bfloat16),
        torch.zeros(2, 1, 4, dtype=torch.int32),
        return_lse=True)

    assert actual_output.shape == (2, 8, 512)
    assert torch.equal(actual_lse, lse[:, :8])
    assert not torch.equal(actual_lse, max_logits[:, :8])


def test_dcp_query_all_gather_preserves_contiguous_head_order(monkeypatch):
    from lmdeploy.pytorch import distributed

    impl = object.__new__(mla_module.FlashMLAImpl)
    impl.dcp_world_size = 2
    impl.dcp_rank = 0
    rank0_query = torch.tensor([[[0.0, 1.0]], [[2.0, 3.0]]])
    rank1_transposed = torch.tensor([[[10.0, 11.0], [12.0, 13.0]]])

    def fake_all_gather(output, input_tensor, group='tp', async_op=False):
        assert group == 'dcp'
        output[:1].copy_(input_tensor)
        output[1:].copy_(rank1_transposed)

    monkeypatch.setattr(distributed, 'all_gather_into_tensor', fake_all_gather)
    gathered = impl._gather_dcp_query(rank0_query)

    expected = torch.tensor([[[0.0, 1.0], [10.0, 11.0]],
                             [[2.0, 3.0], [12.0, 13.0]]])
    assert torch.equal(gathered, expected)


def test_dcp_attention_merge_normalizes_empty_local_shard(monkeypatch):
    from lmdeploy.pytorch import distributed

    impl = object.__new__(mla_module.FlashMLAImpl)
    impl.dcp_world_size = 2
    impl.dcp_rank = 0
    local_output = torch.full((1, 2, 1), 99.0, dtype=torch.bfloat16)
    local_lse = torch.tensor([[torch.nan, torch.inf]])
    remote_lse = torch.tensor([[1.5, 2.0]])
    remote_output = torch.tensor([[[3.0], [5.0]]])

    def fake_all_gather(output, input_tensor, group='tp', async_op=False):
        assert group == 'dcp'
        assert torch.all(torch.isneginf(input_tensor))
        output[:1].copy_(input_tensor)
        output[1:].copy_(remote_lse)

    def fake_reduce_scatter(output,
                            input_tensor,
                            op=None,
                            group='tp',
                            async_op=False):
        assert group == 'dcp'
        assert torch.equal(input_tensor, torch.zeros_like(input_tensor))
        # Rank 0 receives the first head slice. The remote rank contributes
        # its locally normalized output with correction factor one.
        output.copy_(remote_output.transpose(0, 1)[:1])

    monkeypatch.setattr(distributed, 'all_gather_into_tensor', fake_all_gather)
    monkeypatch.setattr(distributed, 'reduce_scatter_tensor',
                        fake_reduce_scatter)
    merged = impl._merge_dcp_attention(local_output,
                                       local_lse,
                                       valid_rows=torch.tensor([False]))

    assert merged.dtype == torch.bfloat16
    assert merged.shape == (1, 1, 1)
    assert merged.item() == 3.0
    assert torch.isfinite(merged).all()


def test_bf16_sparse_decode_strided_cache_matches_contiguous_cache(
        monkeypatch):
    pytest.importorskip('flash_mla')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip('FlashMLA BF16 sparse attention requires an SM90 GPU')

    _disable_dynamic_compile(monkeypatch)
    impl = object.__new__(FlashMLASparseImpl)
    impl.scale = 576**-0.5
    impl.flash_mla_sparse_fwd = None
    impl.index_mapper = FlashMLAIndexMapper()

    batch_size = 2
    query_len = 2
    block_size = 64
    num_blocks = 4
    block_elements = block_size * 576
    storage = torch.empty(num_blocks, block_elements + 128, dtype=torch.bfloat16, device='cuda')
    k_cache = storage[:, :block_elements].view(num_blocks, block_size, 1, 576)
    k_cache.normal_(std=0.1)
    query = torch.randn(batch_size * query_len, 64, 576, dtype=torch.bfloat16, device='cuda')
    nsa_indices = torch.arange(128, dtype=torch.int32, device='cuda').repeat(batch_size * query_len, 1)
    block_offsets = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device='cuda')
    metadata = SimpleNamespace(is_decoding=True,
                               q_seqlens=torch.full((batch_size, ), query_len, dtype=torch.int32, device='cuda'),
                               block_offsets=block_offsets)

    output = impl._decoding_sparse_bf16(query, k_cache, nsa_indices, metadata)

    contiguous_k = k_cache.flatten(0, 1)
    contiguous_indices = impl.index_mapper.map_paged_decode(nsa_indices, block_offsets, query_len, block_size)
    contiguous_indices = contiguous_indices.flatten(0, 1)[:, None]
    expected = impl._flash_mla_sparse_forward(query, contiguous_k, contiguous_indices)
    torch.testing.assert_close(output, expected)


def test_fp8_sparse_decode_pads_tp_query_heads_for_aligned_kernel():
    impl = object.__new__(FlashMLASparseImpl)
    impl.dcp_world_size = 1
    impl.dcp_rank = 0
    impl.causal = True
    impl.scale = 1.0
    impl.v_head_size = 512
    impl.index_mapper = Mock()
    impl.index_mapper.map_paged_decode.return_value = torch.zeros(2, 3, 4, dtype=torch.int32)
    impl.flash_mla_with_kvcache = Mock(
        return_value=(torch.empty(2, 3, 64, 512, dtype=torch.bfloat16), None))

    query = torch.empty(6, 8, 576, dtype=torch.bfloat16)
    k_cache = torch.empty(2, 16, 1, 656, dtype=torch.uint8)
    metadata = SimpleNamespace(
        q_seqlens=torch.tensor([3, 3]),
        kv_seqlens=torch.tensor([16, 16]),
        block_offsets=torch.zeros(2, 1, dtype=torch.int32),
        tile_scheduler_metadata=object(),
        num_splits=None,
    )

    output = impl._decoding_sparse_fp8(query, k_cache, torch.zeros(6, 4, dtype=torch.int32), metadata)

    padded_query = impl.flash_mla_with_kvcache.call_args.args[0]
    assert padded_query.shape == (2, 3, 64, 576)
    assert 'topk_length' not in impl.flash_mla_with_kvcache.call_args.kwargs
    assert output.shape == (6, 8, 512)


def test_bf16_sparse_decode_skips_fp8_flashmla_metadata():
    metadata = SimpleNamespace(block_offsets=torch.tensor([[0, 1]], dtype=torch.int64))
    model_config = SimpleNamespace(use_mla_fp8_cache=False, mla_index_topk=2048)

    CudaOpsBackend.update_meta_flashmla(metadata, model_config, decoding_query_len=5)

    assert metadata.block_offsets.dtype == torch.int32
    assert not hasattr(metadata, 'tile_scheduler_metadata')


def test_bf16_mla_flatten_uses_shared_k_latent_as_value():
    impl = object.__new__(mla_module.FlashMLAImpl)
    impl.v_head_size = 512
    flatten_k = torch.empty(3, 1, 576, dtype=torch.bfloat16)
    impl.flatten_kv_cache = Mock(
        return_value=(flatten_k, torch.empty(3, 1, 0, dtype=torch.bfloat16)))
    metadata = SimpleNamespace(
        kv_start_loc=torch.tensor([0]),
        kv_seqlens=torch.tensor([3]),
        block_offsets=torch.tensor([[0]]),
        kv_flatten_size=3,
        quant_policy=0,
    )

    _, flatten_v = impl._flatten_prefill_kv_cache(
        torch.empty(1, 4, 1, 576, dtype=torch.bfloat16),
        torch.empty(1, 4, 1, 0, dtype=torch.bfloat16),
        metadata,
        out_dtype=torch.bfloat16,
        kv_layout='hsd',
    )

    assert flatten_v.shape == (3, 1, 512)
    assert flatten_v.untyped_storage().data_ptr() == flatten_k.untyped_storage().data_ptr()


def test_sparse_mla_prefill_routes_by_kv_length(monkeypatch):
    dense_output = object()
    sparse_output = object()
    dense_prefill = Mock(return_value=dense_output)
    monkeypatch.setattr(mla_module.FlashMLAImpl, '_forward_prefill', dense_prefill)
    impl = object.__new__(FlashMLASparseImpl)
    impl.dcp_world_size = 1
    impl.dcp_rank = 0
    impl.mla_index_topk = 2048
    impl._flatten_prefill_kv_cache = Mock(return_value=(Mock(), Mock()))
    impl._prefill_sparse = Mock(return_value=sparse_output)
    query, k_cache, v_cache, nsa_indices = (Mock() for _ in range(4))

    dense = impl._forward_prefill(query,
                                  k_cache,
                                  v_cache,
                                  SimpleNamespace(max_kv_seqlen=2048),
                                  nsa_indices=None)
    sparse = impl._forward_prefill(query,
                                   k_cache,
                                   v_cache,
                                   SimpleNamespace(max_kv_seqlen=2049),
                                   nsa_indices=nsa_indices)

    assert dense is dense_output
    assert sparse is sparse_output
    assert dense_prefill.call_args.kwargs['nsa_indices'] is None


def test_dcp_sparse_prefill_maps_partition_indices():
    impl = object.__new__(FlashMLASparseImpl)
    metadata = SimpleNamespace(
        q_seqlens=torch.tensor([2, 1], dtype=torch.int32),
        q_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    indices = torch.tensor([[2, 3, 4, 5], [4, 1, -1, -1],
                            [5, 6, -1, -1]],
                           dtype=torch.int32)

    mapped = impl._map_dcp_prefill_partition(
        indices,
        metadata,
        partition_starts=torch.tensor([3, 5], dtype=torch.int32),
        partition_cu_lens=torch.tensor([0, 2, 3], dtype=torch.int32),
    )

    assert mapped.dtype == torch.int32
    assert mapped[:, 0].tolist() == [[-1, 0, 1, -1], [1, -1, -1, -1],
                                    [2, -1, -1, -1]]


def test_dcp_prefill_gathers_one_context_chunk_in_global_order(monkeypatch):
    from lmdeploy.pytorch import distributed
    from lmdeploy.pytorch.backends.cuda.attention.default import TritonAttentionMetadata

    impl = object.__new__(mla_module.FlashMLAImpl)
    impl.dcp_world_size = 2
    impl.dcp_rank = 0
    impl.v_head_size = 2
    # The first two-token context chunk contains one token per rank/request.
    local_prefix = torch.tensor([0, 10], dtype=torch.bfloat16)
    local_prefix = local_prefix[:, None, None].expand(-1, 1, 4).contiguous()
    impl._flatten_prefill_kv_cache = Mock(return_value=(local_prefix,
                                                        local_prefix[..., :2]))

    rank1_prefix = torch.tensor([1, 11], dtype=torch.bfloat16)
    rank1_prefix = rank1_prefix[:, None, None].expand(-1, 1, 4).contiguous()

    def fake_all_gather(output, input_tensor, group='tp', async_op=False):
        assert group == 'dcp'
        rows = input_tensor.size(0)
        output[:rows].copy_(input_tensor)
        output[rows:].copy_(rank1_prefix)

    monkeypatch.setattr(distributed, 'all_gather_into_tensor', fake_all_gather)

    metadata = TritonAttentionMetadata(
        is_decoding=False,
        q_seqlens=torch.tensor([1, 2], dtype=torch.int32),
        kv_seqlens=torch.tensor([4, 4], dtype=torch.int32),
        q_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        kv_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        kv_flatten_size=8,
        block_offsets=torch.zeros(2, 1, dtype=torch.int32),
        max_kv_seqlen=4,
    )

    context_k, context_cu_lens = impl._gather_dcp_prefill_context_chunk(
        torch.empty(1, 1, 1, 4),
        torch.empty(0),
        metadata,
        prefix_lens=torch.tensor([3, 2], dtype=torch.int32),
        chunk_start=0,
        chunk_size=2,
        out_dtype=torch.bfloat16,
    )

    assert context_k[:4, 0, 0].tolist() == [0, 1, 10, 11]
    assert context_cu_lens.tolist() == [0, 2, 4]


def test_dcp_attention_correction_kernel_matches_torch():
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    from lmdeploy.pytorch.kernels.cuda.dcp import correct_dcp_attention_output, prepare_dcp_lse

    generator = torch.Generator(device='cuda').manual_seed(20260902)
    local_output = torch.randn(5,
                               8,
                               16,
                               dtype=torch.bfloat16,
                               device='cuda',
                               generator=generator)
    # Model FlashMLA's head-padding result: slicing restores the logical head
    # count but leaves a larger physical row stride.
    all_lse = torch.randn(4,
                          5,
                          16,
                          dtype=torch.float32,
                          device='cuda',
                          generator=generator)[..., :8]
    assert not all_lse.is_contiguous()
    all_lse[0, 0] = torch.nan
    all_lse[:, 1] = -torch.inf
    valid_rows = torch.tensor([False, True, True, True, True], device='cuda')
    prepared = prepare_dcp_lse(all_lse[0], valid_rows)
    gathered = all_lse.clone()
    gathered[0].copy_(prepared)

    actual = correct_dcp_attention_output(local_output, gathered, dcp_rank=0)
    global_lse = torch.logsumexp(gathered, dim=0)
    scale = torch.exp(gathered[0] - global_lse)
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
    expected = (local_output.float() * scale[..., None]).transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = correct_dcp_attention_output(local_output,
                                                    gathered,
                                                    dcp_rank=0)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_scatter_dcp_prefill_kv_handles_uneven_requests():
    from lmdeploy.pytorch.backends.cp_utils import get_dcp_local_seq_lens
    from lmdeploy.pytorch.kernels.cuda.dcp import scatter_dcp_prefill_kv

    device = 'cuda'
    dcp_size = 2
    chunk_rows = 2
    prefix_lens = torch.tensor([0, 1, 2, 7, 8, 9],
                               dtype=torch.int32,
                               device=device)
    q_lens = torch.ones_like(prefix_lens)
    kv_lens = prefix_lens + q_lens
    kv_start_loc = torch.nn.functional.pad(kv_lens.cumsum(0), (1, 0))[:-1]
    local_lens = torch.stack([
        get_dcp_local_seq_lens(prefix_lens, (dcp_size, rank))
        for rank in range(dcp_size)
    ])
    max_local_total = int(local_lens.sum(dim=1).max())
    local_prefixes = torch.full((dcp_size, max_local_total, 1),
                                -1,
                                dtype=torch.int32,
                                device=device)
    expected = torch.full((int(kv_lens.sum()), 1),
                          -99,
                          dtype=torch.int32,
                          device=device)
    for rank in range(dcp_size):
        local_start = 0
        for request, prefix_len in enumerate(prefix_lens.tolist()):
            values = request * 100 + torch.arange(prefix_len, device=device)
            owned = values[rank::dcp_size]
            local_prefixes[rank, local_start:local_start + owned.numel(),
                           0] = owned
            local_start += owned.numel()
            expected[kv_start_loc[request]:kv_start_loc[request] + prefix_len,
                     0] = values

    output = torch.full_like(expected, -99)
    for chunk_start in range(0, max_local_total, chunk_rows):
        rows = min(chunk_rows, max_local_total - chunk_start)
        gathered = local_prefixes[:, chunk_start:chunk_start + rows]
        gathered = gathered.reshape(dcp_size * rows, 1).contiguous()
        scatter_dcp_prefill_kv(gathered,
                               output,
                               prefix_lens=prefix_lens,
                               kv_start_loc=kv_start_loc,
                               local_lens=local_lens,
                               chunk_start=chunk_start)
    assert torch.equal(output, expected)
