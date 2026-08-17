# Copyright (c) OpenMMLab. All rights reserved.
import sys
from types import SimpleNamespace

import pytest
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


def test_fa3_normalizes_softcap_during_initialization(monkeypatch):
    fake_interface = SimpleNamespace(
        flash_attn_varlen_func=lambda *args, **kwargs: None,
        flash_attn_with_kvcache=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(
        sys.modules,
        'lmdeploy.pytorch.third_party.flash_attn_interface',
        fake_interface,
    )

    for configured, expected in ((-1.0, 0.0), (0.0, 0.0), (30.0, 30.0)):
        impl = FA3Impl(
            num_heads=2,
            head_size=8,
            logit_softcapping=configured,
        )
        assert impl.logit_softcapping == expected


def test_fa3_prefill_uses_guarded_flatten_buffer_and_max_kv_seqlen():
    """Regression test for FA3 prefill with recycled paged KV blocks."""
    impl = FA3Impl.__new__(FA3Impl)
    impl.scale = 1.0
    impl.causal = True
    impl.sliding_window = None
    # Match the state of an initialized FA3Impl.
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
        captured['flash_softcap'] = kwargs['softcap']
        return torch.empty_like(kwargs['q'])

    impl.flatten_kv_cache = fake_flatten_kv_cache
    impl.flash_attn_varlen_func_v3 = fake_flash_attn_varlen_func

    out = impl._forward_prefill(query, k_cache, v_cache, metadata, max_q_seqlen=int(q_seqlens.max().item()))

    assert out.shape == query.shape
    assert captured['flatten_start_loc'] is metadata.kv_start_loc
    assert captured['flatten_out_size'] == _guarded_flatten_size(q_seqlens)
    assert captured['flash_k_size'] == _guarded_flatten_size(q_seqlens)
    assert captured['flash_max_seqlen_k'] == metadata.max_kv_seqlen
    assert captured['flash_softcap'] == 0.0


def test_fa3_speculative_decode_uses_normalized_disabled_softcap():
    impl = FA3Impl.__new__(FA3Impl)
    impl.scale = 1.0
    impl.causal = True
    impl.sliding_window = None
    # Match the state of an initialized FA3Impl.
    impl.logit_softcapping = 0.0

    captured = {}

    def fake_flash_attn_with_kvcache(query, k_cache, v_cache, **kwargs):
        captured['softcap'] = kwargs['softcap']
        return torch.empty_like(query)

    impl.flash_attn_with_kvcache_v3 = fake_flash_attn_with_kvcache
    metadata = SimpleNamespace(
        quant_policy=QuantPolicy.NONE,
        block_offsets=torch.tensor([[0], [1]], dtype=torch.int32),
        kv_seqlens=torch.tensor([5, 7], dtype=torch.int32),
        scheduler_metadata=None,
    )
    query = torch.empty((4, 2, 8), dtype=torch.float16)
    k_cache = torch.empty((2, _BLOCK_SIZE, 2, 8), dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)

    output = impl._decoding_speculative(query, k_cache, v_cache, metadata, max_q_seqlen=2)

    assert output.shape == (2, 2, 2, 8)
    assert captured['softcap'] == 0.0


def test_flash_mla_fa3_prefill_splits_absorbed_layout(monkeypatch):
    import sys
    from types import ModuleType

    from lmdeploy.pytorch.backends.cuda.attention.mla import FlashMLAImpl

    impl = FlashMLAImpl.__new__(FlashMLAImpl)
    impl.num_kv_heads = 1
    impl.v_head_size = 512
    impl.scale = 0.125
    impl.causal = True
    impl.sliding_window = (-1, -1)
    query = torch.empty((3, 2, 576), dtype=torch.bfloat16)
    flatten_k = torch.empty((3, 1, 576), dtype=torch.bfloat16)
    metadata = _make_prefill_metadata(
        torch.tensor([2, 1], dtype=torch.int32),
        torch.tensor([[0], [1]], dtype=torch.int32),
    )
    captured = {}

    def fake_flash_attn_varlen_func(**kwargs):
        captured.update(kwargs)
        return torch.empty_like(kwargs['qv'])

    fa3_interface = ModuleType(
        'lmdeploy.pytorch.third_party.flash_attn_interface')
    fa3_interface.flash_attn_varlen_func = fake_flash_attn_varlen_func
    monkeypatch.setitem(
        sys.modules,
        'lmdeploy.pytorch.third_party.flash_attn_interface',
        fa3_interface,
    )

    output = impl._prefill_fa3(query, flatten_k, metadata)

    assert output.shape == (3, 2, 512)
    assert captured['q'].shape == (3, 2, 64)
    assert captured['qv'].shape == (3, 2, 512)
    assert captured['k'].shape == (3, 1, 64)
    assert captured['v'].shape == (3, 1, 512)
    assert captured['max_seqlen_q'] == 3
    assert captured['max_seqlen_k'] == 3
    assert captured['causal'] is True
    assert captured['window_size'] == (-1, -1)


def test_flash_mla_builder_uses_available_fa3_for_prefill(monkeypatch):
    import lmdeploy.pytorch.backends.cuda.attention as cuda_attention
    import lmdeploy.pytorch.backends.cuda.attention.mla as mla_attention

    class DummyFlashMLA:

        def __init__(self, use_fa3, **kwargs):
            self.use_fa3 = use_fa3

    monkeypatch.setattr(mla_attention, 'FlashMLAImpl', DummyFlashMLA)
    monkeypatch.setattr(cuda_attention, 'use_fa3', True)
    selected = cuda_attention.TritonAttentionBuilder.build(
        num_heads=8,
        head_size=576,
        num_kv_heads=1,
        v_head_size=512,
        use_flash_mla=True,
    )
    assert isinstance(selected, DummyFlashMLA)
    assert selected.use_fa3 is True

    monkeypatch.setattr(cuda_attention, 'use_fa3', False)
    unavailable = cuda_attention.TritonAttentionBuilder.build(
        num_heads=8,
        head_size=576,
        num_kv_heads=1,
        v_head_size=512,
        use_flash_mla=True,
    )
    assert unavailable.use_fa3 is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_flash_mla_fa3_prefill_matches_fp32_reference():
    import lmdeploy.pytorch.backends.cuda.attention as cuda_attention
    from lmdeploy.pytorch.backends.cuda.attention.mla import FlashMLAImpl

    if not cuda_attention.use_fa3:
        pytest.skip('requires FA3')

    torch.manual_seed(1)
    device = torch.device('cuda')
    dtype = torch.bfloat16
    seqlens = (7, 5)
    q_seqlens = torch.tensor(seqlens, device=device, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(q_seqlens, 0, dtype=torch.int32), (1, 0))
    num_tokens = sum(seqlens)
    query = (torch.randn((num_tokens, 2, 576),
                         device=device,
                         dtype=dtype) * 0.1).clamp(-1, 1)
    flatten_k = (torch.randn((num_tokens, 1, 576),
                             device=device,
                             dtype=dtype) * 0.1).clamp(-1, 1)
    metadata = TritonAttentionMetadata(
        is_decoding=False,
        block_offsets=torch.tensor([[0], [1]],
                                   device=device,
                                   dtype=torch.int32),
        q_start_loc=cu_seqlens[:-1],
        q_seqlens=q_seqlens,
        kv_start_loc=cu_seqlens[:-1],
        kv_seqlens=q_seqlens,
        quant_policy=QuantPolicy.NONE,
        kv_flatten_size=num_tokens,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens.clone(),
        max_kv_seqlen=max(seqlens),
        max_q_seqlen=max(seqlens),
    )
    impl = FlashMLAImpl.__new__(FlashMLAImpl)
    impl.num_kv_heads = 1
    impl.v_head_size = 512
    impl.scale = 0.125
    impl.causal = True
    impl.sliding_window = (-1, -1)

    output = impl._prefill_fa3(query, flatten_k, metadata)

    references = []
    start = 0
    for seq_len in seqlens:
        q_nope, q_rope = query[start:start + seq_len].float().split(
            [512, 64], dim=-1)
        value, k_rope = flatten_k[start:start + seq_len, 0].float().split(
            [512, 64], dim=-1)
        scores = (
            torch.einsum('qhd,kd->hqk', q_rope, k_rope)
            + torch.einsum('qhd,kd->hqk', q_nope, value)
        ) * impl.scale
        causal_mask = torch.arange(seq_len, device=device)[None, :] > \
            torch.arange(seq_len, device=device)[:, None]
        probabilities = scores.masked_fill(
            causal_mask[None], float('-inf')).softmax(-1)
        references.append(
            torch.einsum('hqk,kd->qhd', probabilities, value))
        start += seq_len
    reference = torch.cat(references).to(dtype)

    torch.testing.assert_close(output, reference, atol=1.6e-2, rtol=1.6e-2)
