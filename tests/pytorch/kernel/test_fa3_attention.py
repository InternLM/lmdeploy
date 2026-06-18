# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.backends.cuda.attention.default import TritonAttentionMetadata
from lmdeploy.pytorch.backends.cuda.attention.fa3 import FA3Impl

_BLOCK_SIZE = 16
_PREFILL_SEQLENS = (29, 18)


def _make_prefill_metadata(q_seqlens, block_offsets):
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(q_seqlens, dim=0, dtype=torch.int32), (1, 0))
    return TritonAttentionMetadata(
        is_decoding=False,
        block_offsets=block_offsets,
        q_start_loc=cu_seqlens[:-1],
        q_seqlens=q_seqlens,
        kv_start_loc=cu_seqlens[:-1],
        kv_seqlens=q_seqlens,
        quant_policy=QuantPolicy.NONE,
        kv_flatten_size=int(q_seqlens.sum().item()),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens.clone(),
        max_kv_seqlen=int(q_seqlens.max().item()),
        max_q_seqlen=int(q_seqlens.max().item()),
    )


def _make_recycled_block_offsets(device):
    return torch.tensor([
        [0, 2, 1],
        [3, 4, 0],
    ],
                        dtype=torch.int32,
                        device=device)


def _make_prefill_seqlens(device='cpu'):
    return torch.tensor(_PREFILL_SEQLENS, dtype=torch.int32, device=device)


def _guarded_flatten_size(q_seqlens):
    kv_flatten_size = int(q_seqlens.sum().item())
    return (kv_flatten_size + _BLOCK_SIZE - 1) // _BLOCK_SIZE * _BLOCK_SIZE + _BLOCK_SIZE


def _num_cache_blocks(block_offsets):
    return int(block_offsets.max().item()) + 1


def test_fa3_prefill_uses_guarded_flatten_buffer_and_max_kv_seqlen():
    """Regression test for FA3 prefill with recycled paged KV blocks."""
    impl = FA3Impl.__new__(FA3Impl)
    impl.scale = 1.0
    impl.causal = True
    impl.sliding_window = None
    impl.logit_softcapping = 0.0

    q_seqlens = _make_prefill_seqlens()
    block_offsets = _make_recycled_block_offsets(device='cpu')
    metadata = _make_prefill_metadata(q_seqlens, block_offsets)

    query = torch.empty((int(q_seqlens.sum().item()), 2, 8), dtype=torch.float16)
    k_cache = torch.empty((_num_cache_blocks(block_offsets), _BLOCK_SIZE, 2, 8), dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)
    captured = {}

    def fake_flatten_kv_cache(k_cache_arg, v_cache_arg, seqlens, offsets, **kwargs):
        captured['flatten_out_size'] = kwargs['out_size']
        captured['flatten_start_loc'] = kwargs['start_loc']
        return (
            torch.empty((kwargs['out_size'], 2, 8), dtype=kwargs['out_dtype']),
            torch.empty((kwargs['out_size'], 2, 8), dtype=kwargs['out_dtype']),
        )

    def fake_flash_attn_varlen_func(**kwargs):
        captured['flash_max_seqlen_k'] = kwargs['max_seqlen_k']
        captured['flash_k_size'] = kwargs['k'].size(0)
        return torch.empty_like(kwargs['q'])

    impl.flatten_kv_cache = fake_flatten_kv_cache
    impl.flash_attn_varlen_func_v3 = fake_flash_attn_varlen_func

    out = impl._forward_prefill(query, k_cache, v_cache, metadata, max_q_seqlen=int(q_seqlens.max().item()))

    assert out.shape == query.shape
    assert captured['flatten_start_loc'] is metadata.kv_start_loc
    assert captured['flatten_out_size'] == _guarded_flatten_size(q_seqlens)
    assert captured['flash_k_size'] == _guarded_flatten_size(q_seqlens)
    assert captured['flash_max_seqlen_k'] == metadata.max_kv_seqlen
