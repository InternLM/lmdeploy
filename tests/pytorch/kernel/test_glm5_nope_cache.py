# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize('layout', ['hsd', 'shd'])
@pytest.mark.parametrize('storage_width', [512, 576])
def test_nope_flatten_reuses_shared_value_output(layout, storage_width):
    from lmdeploy.pytorch.kernels.cuda.flatten_kv_cache import flatten_kv_cache

    torch.manual_seed(113)
    cache = torch.randn(7, 64, 1, storage_width, device='cuda', dtype=torch.bfloat16)
    lengths = torch.tensor([67, 19], device='cuda')
    blocks = torch.tensor([[3, 1], [5, 2]], device='cuda')
    keys, values = flatten_kv_cache(cache, cache[..., :512], lengths, blocks,
                                    out_size=128, flatten_kv_layout=layout)
    expected = torch.cat((cache[3], cache[1, :3], cache[5, :19]))
    expected = F.pad(expected, (0, 0, 0, 0, 0, 128 - expected.size(0)))
    if layout == 'hsd':
        expected = expected.transpose(0, 1)
    torch.testing.assert_close(keys, expected, rtol=0, atol=0)
    torch.testing.assert_close(values, expected[..., :512], rtol=0, atol=0)
    assert keys.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()


def make_nope_case(lengths, histories, heads, decoding):
    from lmdeploy.pytorch.backends.cuda.attention import TritonAttentionMetadata
    from lmdeploy.pytorch.backends.cuda.attention.mla import FlashMLAImpl
    from lmdeploy.pytorch.backends.cuda.attention.tilelang_sparse_mla import TilelangSparseMLADecode

    torch.manual_seed(127)
    q_lens = torch.tensor(lengths, device='cuda')
    kv_lens = q_lens + torch.tensor(histories, device='cuda')
    q_ends = q_lens.cumsum(0, dtype=torch.int32)
    kv_ends = kv_lens.cumsum(0, dtype=torch.int32)
    columns = (max(a + b for a, b in zip(lengths, histories)) + 63) // 64
    blocks = torch.randperm(len(lengths) * columns, device='cuda', dtype=torch.int32) + 1
    blocks = blocks.reshape(len(lengths), columns)
    metadata = TritonAttentionMetadata(
        is_decoding=decoding, block_offsets=blocks, q_start_loc=q_ends - q_lens,
        q_seqlens=q_lens, kv_start_loc=kv_ends - kv_lens, kv_seqlens=kv_lens,
        cu_seqlens_q=F.pad(q_ends, (1, 0)), cu_seqlens_k=F.pad(kv_ends, (1, 0)),
        kv_flatten_size=sum(a + b for a, b in zip(lengths, histories)),
        max_kv_seqlen=max(a + b for a, b in zip(lengths, histories)), max_q_seqlen=max(lengths))
    query = torch.randn(sum(lengths), heads, 512, device='cuda', dtype=torch.bfloat16)
    key = torch.randn(sum(lengths), 1, 512, device='cuda', dtype=torch.bfloat16)
    initial = torch.randn(len(lengths) * columns + 1, 64, 1, 512, device='cuda', dtype=torch.bfloat16)
    indices = torch.full((sum(lengths), 2048), -1, device='cuda', dtype=torch.int32)
    offset = 0
    for count, history in zip(lengths, histories):
        for i in range(count):
            seq = history + i + 1
            ids = torch.arange(max(0, seq - 2048), seq, device='cuda', dtype=torch.int32)
            indices[offset + i, :ids.numel()] = ids
        offset += count
    outputs, caches, calls = {}, {}, {}
    for width in (576, 512):
        cache = F.pad(initial, (0, width - 512)) if width == 576 else initial.clone()
        current = F.pad(key, (0, width - 512)) if width == 576 else key
        impl = FlashMLAImpl(heads, width, num_kv_heads=1, v_head_size=512)
        writer = SimpleNamespace(impl=impl, _lazy_init=lambda device: None,
                                 fill_and_flatten_latent_kv_cache=impl.fill_and_flatten_latent_kv_cache)
        backend = TilelangSparseMLADecode(2048, 4)
        if decoding:
            def call(backend=backend, current=current, cache=cache, writer=writer):
                return backend.forward(query, current, current[..., :512], cache, cache[..., :512],
                                       metadata, 192**-0.5, writer, logical_indices=indices)
        else:
            def call(backend=backend, current=current, cache=cache, writer=writer):
                return backend.forward_prefill(query, current, cache, metadata, 192**-0.5, writer, indices)
        calls[width] = call
        outputs[width] = call()
        caches[width] = cache
    return outputs, caches, calls, query, metadata


@pytest.mark.parametrize('lengths,histories,decoding', [
    ([128], [0], False), ([3, 5], [63, 126], False), ([257, 255], [4095, 8191], False),
    ([1, 1], [4095, 127], True), ([6, 6], [4095, 127], True),
])
@pytest.mark.parametrize('heads', [8, 16])
def test_nope_512_matches_padded_576_backend(lengths, histories, decoding, heads):
    outputs, caches, _, _, _ = make_nope_case(lengths, histories, heads, decoding)
    torch.testing.assert_close(outputs[512], outputs[576], rtol=0, atol=0)
    torch.testing.assert_close(caches[512], caches[576][..., :512], rtol=0, atol=0)
    assert caches[512].nbytes * 9 == caches[576].nbytes * 8


@pytest.mark.parametrize('steps', [1, 6])
def test_nope_decode_graph_replays_with_new_lengths(steps):
    outputs, caches, calls, query, metadata = make_nope_case([steps, steps], [63, 126], 16, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = calls[512]()
    metadata.kv_seqlens.add_(1)
    query.normal_()
    expected = calls[576]()
    graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(caches[512], caches[576][..., :512], rtol=0, atol=0)
