# Copyright (c) OpenMMLab. All rights reserved.
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.backends.attention import PagedAttentionBuildSpec, SWAStateRingAttentionBuildSpec
from lmdeploy.pytorch.backends.cuda.attention import _build_paged_attention
from lmdeploy.pytorch.backends.cuda.attention.default import TritonAttentionImpl, TritonAttentionMetadata
from lmdeploy.pytorch.backends.cuda.attention.fa3 import FA3Impl
from lmdeploy.pytorch.backends.cuda.attention.fa3_capabilities import fa3_build_supports_operation
from lmdeploy.pytorch.backends.cuda.attention.swa_state_ring import SWAStateRingAttentionImpl
from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend

_BLOCK_SIZE = 16
_PREFILL_SEQLENS = (29, 18)


def test_attention_builder_falls_back_when_fa3_lacks_asymmetric_head_shape(monkeypatch):
    """Avoid dispatching a head shape omitted from the installed FA3 wheel."""
    import lmdeploy.pytorch.backends.cuda.attention as attention_mod

    flags = {
        'FLASHATTENTION_DISABLE_HDIM192': False,
        'FLASH_ATTENTION_DISABLE_HDIMDIFF192': True,
    }
    monkeypatch.setitem(sys.modules, 'flash_attn_config', SimpleNamespace(CONFIG={'build_flags': flags}))
    monkeypatch.setattr(attention_mod, 'use_fa3_warning', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (9, 0))
    impl = _build_paged_attention(
        PagedAttentionBuildSpec(
            num_heads=8,
            head_dim=192,
            scale=None,
            num_kv_heads=2,
            v_head_dim=128,
            alibi=False,
            sliding_window=None,
            logit_softcapping=0.0,
            causal=True,
            use_flash_mla=False,
            mla_index_topk=None,
            learnable_sink=False,
            block_sparse_size=1,
        ))

    assert type(impl) is TritonAttentionImpl


def test_fa3_operation_capability_checks_arch_and_build_features(monkeypatch):
    flags = {
        'FLASHATTENTION_DISABLE_HDIM128': False,
        'FLASHATTENTION_DISABLE_HDIM192': False,
        'FLASH_ATTENTION_DISABLE_HDIMDIFF192': False,
    }
    monkeypatch.setitem(sys.modules, 'flash_attn_config', SimpleNamespace(CONFIG={'build_flags': flags}))

    assert not fa3_build_supports_operation(
        192,
        128,
        device_capability=(8, 0),
        dtype=torch.bfloat16,
        paged_kv=True,
        varlen=True,
    )
    assert fa3_build_supports_operation(
        192,
        128,
        device_capability=(9, 0),
        dtype=torch.bfloat16,
        paged_kv=True,
        varlen=True,
    )

    requirements = {
        'FLASHATTENTION_DISABLE_PAGEDKV': dict(paged_kv=True),
        'FLASHATTENTION_DISABLE_VARLEN': dict(varlen=True),
        'FLASHATTENTION_DISABLE_LOCAL': dict(local=True),
        'FLASHATTENTION_DISABLE_SOFTCAP': dict(softcap=True),
        'FLASHATTENTION_DISABLE_FP16': dict(dtype=torch.float16),
    }
    for flag, requirement in requirements.items():
        flags[flag] = True
        assert not fa3_build_supports_operation(
            128,
            128,
            device_capability=(9, 0),
            **requirement,
        )
        flags[flag] = False


def test_legacy_spec_metadata_skips_fa3_when_fallback_is_selected(monkeypatch):
    """Triton fallback must not retain a hidden FA3 metadata dependency."""
    import lmdeploy.pytorch.backends.cuda.attention as attention_mod

    update_meta = Mock()
    monkeypatch.setattr(attention_mod, '_enable_fa3', lambda *args, **kwargs: False)
    monkeypatch.setattr(CudaOpsBackend, 'update_meta_flashattn', update_meta)
    step_context = SimpleNamespace(
        is_decoding=True,
        q_seqlens=torch.tensor([2]),
        input_ids=torch.zeros((1, 2), dtype=torch.long),
        model_config=SimpleNamespace(
            use_flash_mla=False,
            model_paradigm='ar_spec',
            head_dim=192,
            v_head_dim=128,
            sliding_window=None,
            dtype=torch.bfloat16,
            is_gated_delta=False,
        ),
    )

    metadata = object()
    assert CudaOpsBackend._legacy_update_step_context(step_context, metadata) is metadata
    update_meta.assert_not_called()


def test_swa_state_ring_uses_dedicated_backend_implementation():
    spec = SWAStateRingAttentionBuildSpec(
        num_heads=8,
        head_dim=192,
        scale=None,
        num_kv_heads=2,
        v_head_dim=128,
        sliding_window=(127, 0),
        learnable_sink=True,
    )

    impl = CudaOpsBackend.build_op(spec)

    assert type(impl) is SWAStateRingAttentionImpl
    assert impl.supports_multi_token_decode is True
    assert impl.scale == 1.0 / (192**0.5)
    assert impl.flash_attention_fwd is None
    assert impl.paged_attention_fwd is None


def test_paged_attention_build_spec_is_pure_configuration():
    from lmdeploy.pytorch.backends.attention import PagedAttentionBuildSpec
    from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
    spec = PagedAttentionBuildSpec(
        num_heads=8,
        head_dim=192,
        scale=None,
        num_kv_heads=2,
        v_head_dim=128,
        alibi=False,
        sliding_window=(127, 0),
        logit_softcapping=0.0,
        causal=True,
        use_flash_mla=False,
        mla_index_topk=None,
        learnable_sink=True,
        block_sparse_size=1,
    )

    assert not hasattr(spec, 'requires_multi_token_decode')
    assert CudaOpsBackend.build_op(spec) is not None


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


def test_fa3_multi_token_decode_uses_mode_and_normalized_softcap():
    impl = FA3Impl.__new__(FA3Impl)
    impl.scale = 1.0
    impl.causal = True
    impl.sliding_window = None
    # Match the state of an initialized FA3Impl.
    impl.logit_softcapping = 0.0
    impl._step_meta_group = None

    captured = {}

    def fake_flash_attn_with_kvcache(query, k_cache, v_cache, **kwargs):
        captured['softcap'] = kwargs['softcap']
        captured['causal'] = kwargs['causal']
        return torch.empty_like(query)

    impl.flash_attn_with_kvcache_v3 = fake_flash_attn_with_kvcache
    metadata = SimpleNamespace(
        quant_policy=QuantPolicy.NONE,
        block_offsets=torch.tensor([[0], [1]], dtype=torch.int32),
        kv_seqlens=torch.tensor([5, 7], dtype=torch.int32),
        scheduler_metadata=None,
        kernel_metadata=(),
    )
    query = torch.empty((4, 2, 8), dtype=torch.float16)
    k_cache = torch.empty((2, _BLOCK_SIZE, 2, 8), dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)

    for decode_mode, expected_causal in (('block', False), ('speculative', True)):
        output = impl._decoding_speculative(
            query,
            k_cache,
            v_cache,
            metadata,
            max_q_seqlen=2,
            decode_mode=decode_mode,
        )

        assert output.shape == (2, 2, 2, 8)
        assert captured['softcap'] == 0.0
        assert captured['causal'] is expected_causal


def test_fa3_scheduler_metadata_uses_decode_mode(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.attention import fa3 as fa3_module

    causal_values = []

    def fake_get_meta_flashattn(**kwargs):
        causal_values.append(kwargs['causal'])
        return torch.empty(1)

    monkeypatch.setattr(fa3_module, '_get_meta_flashattn', fake_get_meta_flashattn)
    step_context = SimpleNamespace(
        decode_mode='block',
        input_ids=torch.zeros((1, 2), dtype=torch.long),
        model_config=SimpleNamespace(block_size=16, dtype=torch.bfloat16),
    )
    kwargs = dict(
        batch_size=1,
        kv_seqlens=torch.tensor([2]),
        block_offsets=torch.tensor([[0]], dtype=torch.int32),
        step_context=step_context,
        num_heads_q=2,
        num_heads_kv=1,
        head_size=128,
        v_head_size=128,
        sliding_window=None,
    )

    fa3_module._build_fa3_metadata(**kwargs)
    step_context.decode_mode = 'speculative'
    fa3_module._build_fa3_metadata(**kwargs)

    assert causal_values == [False, True]
