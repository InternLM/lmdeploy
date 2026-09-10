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
    TileLangSparseMLAImpl,
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


def test_builder_selects_tilelang_sparse_impl_from_env(monkeypatch):
    output = object()
    tilelang_impl = Mock(return_value=output)
    monkeypatch.setattr(attention_module._envs, 'sparse_mla_backend', 'tilelang')
    monkeypatch.setattr(sparse_mla_module, 'TileLangSparseMLAImpl', tilelang_impl)

    spec = PagedAttentionBuildSpec(
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
        mla_index_topk=2048,
        learnable_sink=False,
        block_sparse_size=1,
    )
    assert CudaOpsBackend.build_op(spec) is output
    assert tilelang_impl.call_args.kwargs['mla_index_topk'] == 2048


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_dcp_attention_merge_normalizes_empty_local_shard(monkeypatch):
    from lmdeploy.pytorch import distributed

    impl = object.__new__(mla_module.FlashMLAImpl)
    impl.dcp_world_size = 2
    impl.dcp_rank = 0
    local_output = torch.full((1, 2, 1), torch.nan, dtype=torch.bfloat16, device='cuda')
    local_lse = torch.tensor([[torch.nan, torch.inf]], device='cuda')
    remote_lse = torch.tensor([[1.5, 2.0]], device='cuda')
    remote_output = torch.tensor([[[3.0], [5.0]]], device='cuda')

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
                                       valid_rows=torch.tensor([False], device='cuda'))

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


def test_tilelang_sparse_decode_uses_flat_bf16_cache_view(monkeypatch):
    _disable_dynamic_compile(monkeypatch)
    impl = object.__new__(TileLangSparseMLAImpl)
    impl.scale = 576**-0.5
    impl.index_mapper = FlashMLAIndexMapper()
    impl._tilelang_sparse_mla_forward = Mock(
        return_value=torch.empty(2, 8, 512, dtype=torch.bfloat16))

    query = torch.empty(2, 8, 576, dtype=torch.bfloat16)
    k_cache = torch.arange(3 * 32 * 576, dtype=torch.float32)
    k_cache = k_cache.to(torch.bfloat16).view(3, 32, 1, 576)
    nsa_indices = torch.tensor([[0, 33, -1], [0, 1, 32]])
    metadata = SimpleNamespace(
        is_decoding=True,
        q_seqlens=torch.ones(2, dtype=torch.int32),
        block_offsets=torch.tensor([[1, 0], [2, 0]], dtype=torch.int32),
    )

    output = impl._decoding_sparse_bf16(query, k_cache, nsa_indices, metadata)

    args = impl._tilelang_sparse_mla_forward.call_args.args
    kernel_query, storage_k, indices, scale = args
    assert kernel_query is query
    assert storage_k.shape == (96, 1, 576)
    assert storage_k.untyped_storage().data_ptr() == k_cache.untyped_storage().data_ptr()
    expected = torch.tensor([[[32, 1, -1]], [[64, 65, 0]]])
    assert torch.equal(indices, expected)
    assert scale == impl.scale
    assert output.shape == (2, 8, 512)


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
    impl._step_meta_group = None

    query = torch.empty(6, 8, 576, dtype=torch.bfloat16)
    k_cache = torch.empty(2, 16, 1, 656, dtype=torch.uint8)
    metadata = SimpleNamespace(
        q_seqlens=torch.tensor([3, 3]),
        kv_seqlens=torch.tensor([16, 16]),
        block_offsets=torch.zeros(2, 1, dtype=torch.int32),
        tile_scheduler_metadata=object(),
        num_splits=None,
        kernel_metadata=(),
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('fp8_cache', [False, True])
@pytest.mark.parametrize('dcp_size', [2, 4])
def test_dcp_cached_prefill_matches_reference_across_chunks(monkeypatch, sparse, fp8_cache, dcp_size):
    pytest.importorskip('flash_mla')
    if sparse and torch.cuda.get_device_capability()[0] != 9:
        pytest.skip('FlashMLA BF16 sparse attention requires an SM90 GPU')

    from lmdeploy.pytorch import distributed
    from lmdeploy.pytorch.backends import cp_utils
    from lmdeploy.pytorch.backends.cuda.attention.default import (
        TritonAttentionMetadata,
        build_triton_attention_metadata,
    )
    torch.manual_seed(51)
    device = 'cuda'
    lengths = torch.tensor([259, 3], dtype=torch.int32, device=device)
    q_lengths = torch.tensor([2, 3], dtype=torch.int32, device=device)
    keys = torch.randn(262, 1, 576, dtype=torch.bfloat16, device=device)
    query = torch.randn(5, 8, 576, dtype=torch.bfloat16, device=device)
    current_key = torch.cat([keys[257:259], keys[259:]])
    blocks = torch.tensor([[0, 1, 2], [3, 0, 0]], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 259, 262], dtype=torch.int32, device=device)
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    layout = SimpleNamespace(block_offsets=blocks,
                             q_start_loc=cu_q[:-1],
                             q_seqlens=q_lengths,
                             kv_start_loc=cu_k[:-1],
                             kv_seqlens=lengths,
                             kv_flatten_size=262,
                             cu_seqlens_q=cu_q,
                             cu_seqlens_k=cu_k,
                             max_kv_seqlen=259)
    step = SimpleNamespace(is_decoding=False,
                           max_q_seqlen=3,
                           input_ids=torch.zeros(1, 5),
                           cache_config=SimpleNamespace(block_size=64),
                           model_config=SimpleNamespace(head_dim=576),
                           kv_quant_policy=0)
    monkeypatch.setattr(distributed, 'get_dcp_world_rank', lambda: (dcp_size, 0))
    # One virtual block per chunk, with an empty second request in every chunk.
    monkeypatch.setattr(cp_utils, 'get_dcp_prefill_workspace_size',
                        lambda **kwargs: 2 * 64 * (1 + 2 * dcp_size) * 576 * 2)
    metadata = build_triton_attention_metadata(TritonAttentionMetadata, step, layout)
    num_chunks = (257 + 64 * dcp_size - 1) // (64 * dcp_size)
    assert len(metadata.dcp_prefill_chunks) == num_chunks
    impl_cls = FlashMLASparseImpl if sparse else mla_module.FlashMLAImpl
    kwargs = dict(mla_index_topk=512) if sparse else {}
    impl = impl_cls(num_heads=8,
                    head_size=576,
                    num_kv_heads=1,
                    v_head_size=512,
                    scale=576**-0.5,
                    use_fa3=False,
                    **kwargs)
    cache = torch.zeros(4, 64, 1, 656 if fp8_cache else 576,
                        dtype=torch.float8_e4m3fn if fp8_cache else torch.bfloat16, device=device)
    value_cache = cache[..., :0] if fp8_cache else cache[..., :512]
    fill_metadata = TritonAttentionMetadata(
        is_decoding=False, block_offsets=blocks, q_start_loc=cu_k[:-1],
        q_seqlens=lengths, kv_seqlens=lengths, cu_seqlens_q=cu_k)
    impl._fill_kv_cache_impl(keys, keys[..., :512], cache, value_cache, fill_metadata, 259)
    reference_keys = keys.clone()
    if fp8_cache:
        # Independent token-wise UE8M0 scale / FP8 round trip. Only cached
        # history is quantized in DCP prefill; current keys remain BF16.
        latent = keys[:257, :, :512].float().unflatten(-1, (4, 128))
        scales = (latent.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 448).log2().ceil().exp2()
        latent = (latent / scales).to(torch.float8_e4m3fn).float() * scales
        reference_keys[:257, :, :512] = latent.flatten(-2).to(torch.bfloat16)
    gather_calls = 0

    def gather(output, local, group='tp'):
        nonlocal gather_calls
        assert group == 'dcp'
        chunk = metadata.dcp_prefill_chunks[gather_calls]
        prefix = reference_keys[chunk.start:min(chunk.start + chunk.size, 257)]
        torch.testing.assert_close(local[:prefix[::dcp_size].size(0)], prefix[::dcp_size], atol=0, rtol=0)
        output[:local.size(0)].copy_(local)
        for rank in range(1, dcp_size):
            remote = output[rank * local.size(0):(rank + 1) * local.size(0)]
            remote.zero_()
            remote[:prefix[rank::dcp_size].size(0)].copy_(prefix[rank::dcp_size])
        gather_calls += 1

    monkeypatch.setattr(distributed, 'all_gather_into_tensor', gather)
    if sparse:
        # Non-contiguous selections span cached chunks and current tokens.
        # The two long-query rows intentionally select different positions.
        selected_positions = [[7, 128, 256, 257], [0, 127, 200, 258], [0], [1], [0, 2]]
        indices = torch.full((5, 512), -1, dtype=torch.int32, device=device)
        for row, positions in enumerate(selected_positions):
            indices[row, :len(positions)] = torch.tensor(positions, dtype=torch.int32, device=device)
        actual = impl._prefill_sparse_dcp(query, current_key, cache, value_cache, indices, metadata)
    else:
        actual = impl._prefill_dcp_context(query, current_key, cache, value_cache, metadata)
    expected = []
    for row, (start, end) in enumerate([(0, 258), (0, 259), (259, 260), (259, 261), (259, 262)]):
        kv = reference_keys[start:end, 0].float()
        if sparse:
            kv = kv[selected_positions[row]]
        scores = query[row].float() @ kv.T * 576**-0.5
        expected.append(scores.softmax(-1) @ kv[:, :512])
    assert actual.dtype == query.dtype
    assert gather_calls == num_chunks
    torch.testing.assert_close(actual.float(), torch.stack(expected), atol=2e-2, rtol=2e-2)


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
    all_lse[0, 3, :2] = torch.tensor([torch.inf, torch.nan], device='cuda')
    # Empty shards may return non-finite outputs or even finite LSE values.
    local_output[:3] = torch.nan
    valid_rows = torch.tensor([False, True, False, True, True], device='cuda')
    prepared = prepare_dcp_lse(all_lse[0], valid_rows)
    expected_lse = torch.where(valid_rows[:, None] & torch.isfinite(all_lse[0]), all_lse[0], -torch.inf)
    torch.testing.assert_close(prepared, expected_lse, rtol=0, atol=0)
    assert prepared.is_contiguous()
    gathered = all_lse.clone()
    gathered[0].copy_(prepared)

    actual = correct_dcp_attention_output(local_output, gathered, dcp_rank=0)
    global_lse = torch.logsumexp(gathered, dim=0)
    scale = torch.exp(gathered[0] - global_lse)
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
    expected = torch.where(scale[..., None] == 0, 0,
                           local_output.float() * scale[..., None]).transpose(0, 1)
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
@pytest.mark.parametrize('dcp_size', [2, 4])
def test_scatter_dcp_prefill_kv_handles_uneven_requests(monkeypatch, dcp_size):
    from lmdeploy.pytorch.backends import cp_utils
    from lmdeploy.pytorch.kernels.cuda.dcp import scatter_dcp_prefill_kv

    device = 'cuda'
    lengths = [0, 1, 2, 7, 8, 9]
    prefix_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    # One virtual block per chunk, matching the production gather layout.
    monkeypatch.setattr(cp_utils, 'get_dcp_prefill_workspace_size',
                        lambda **kwargs: len(lengths) * (1 + 2 * dcp_size) * 2)
    chunks = cp_utils.build_dcp_prefill_chunks(
        prefix_lens=prefix_lens, prefix_limit=max(lengths), block_size=1,
        head_dim=1, dcp_world_rank=(dcp_size, 0))
    for chunk in chunks:
        values = [request * 100 + torch.arange(length, device=device)[chunk.start:chunk.start + chunk.size]
                  for request, length in enumerate(lengths)]
        local_capacity = len(lengths) * chunk.size // dcp_size
        gathered = torch.full((dcp_size, local_capacity, 1), -1, dtype=torch.int32, device=device)
        for rank in range(dcp_size):
            owned = torch.cat([value[rank::dcp_size] for value in values])
            gathered[rank, :owned.numel(), 0] = owned
        output = torch.full((len(lengths) * chunk.size, 1), -99, dtype=torch.int32, device=device)
        expected = torch.full_like(output, -99)
        valid_values = torch.cat(values)
        expected[:valid_values.numel(), 0] = valid_values
        gathered = gathered.flatten(0, 1)
        scatter_dcp_prefill_kv(gathered,
                               output,
                               prefix_lens=chunk.kv_seqlens,
                               kv_start_loc=chunk.cu_seqlens[:-1],
                               local_lens=chunk.local_kv_seqlens)
        assert torch.equal(output, expected)


def test_tilelang_sparse_mla_decode_zero_copy_matches_selected_reference(monkeypatch):
    pytest.importorskip('tilelang')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip('TileLang SparseMLA requires an SM90 GPU')
    _disable_dynamic_compile(monkeypatch)
    from lmdeploy.pytorch.kernels.cuda.tilelang_sparse_mla import tilelang_sparse_mla_forward

    impl = object.__new__(TileLangSparseMLAImpl)
    impl.scale = 576**-0.5
    impl.index_mapper = FlashMLAIndexMapper()
    impl._tilelang_sparse_mla_forward = tilelang_sparse_mla_forward

    batch_size, query_len, num_heads = 2, 2, 8
    block_size, num_blocks, topk = 64, 4, 64
    query = torch.randn(batch_size * query_len, num_heads, 576,
                        dtype=torch.bfloat16, device='cuda') * 0.1
    k_cache = torch.randn(num_blocks, block_size, 1, 576,
                          dtype=torch.bfloat16, device='cuda') * 0.1
    nsa_indices = torch.arange(topk, dtype=torch.int32, device='cuda')
    nsa_indices = nsa_indices.repeat(batch_size * query_len, 1)
    nsa_indices[1, -3:] = -1
    block_offsets = torch.tensor([[2, 0], [3, 1]],
                                 dtype=torch.int32, device='cuda')
    metadata = SimpleNamespace(
        is_decoding=True,
        q_seqlens=torch.full((batch_size, ), query_len,
                             dtype=torch.int32, device='cuda'),
        block_offsets=block_offsets,
    )

    output = impl._decoding_sparse_bf16(query, k_cache, nsa_indices, metadata)

    physical_indices = impl.index_mapper.map_paged_decode(
        nsa_indices, block_offsets, query_len, block_size).flatten(0, 1)
    flat_k = k_cache.flatten(0, 1)
    expected = []
    for row, indices in enumerate(physical_indices):
        valid = indices >= 0
        selected_kv = flat_k[indices.clamp_min(0)]
        compact = torch.arange(topk, dtype=torch.int32, device='cuda')
        compact = compact.masked_fill(~valid, -1)[None, None]
        expected.append(tilelang_sparse_mla_forward(query[row:row + 1],
                                                    selected_kv, compact,
                                                    impl.scale))
    expected = torch.cat(expected)

    assert torch.equal(output, expected)
