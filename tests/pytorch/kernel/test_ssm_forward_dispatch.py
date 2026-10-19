# Copyright (c) OpenMMLab. All rights reserved.
"""Host dispatch contracts; real PCG/kernel execution is covered separately."""
import sys
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.backends.cuda import causal_conv1d as conv
from lmdeploy.pytorch.backends.cuda import gated_delta_rule as gdn
from lmdeploy.pytorch.backends.cuda import kda
from lmdeploy.pytorch.backends.cuda.graph_runner import piecewise


@pytest.fixture
def execution(monkeypatch):
    current = SimpleNamespace(value=None)
    monkeypatch.setattr(piecewise, 'get_piecewise_graph_execution', lambda: current.value)
    # Test routing and slicing without pretending this exercises GPU capture.
    monkeypatch.setattr(piecewise, 'eager_boundary', lambda **kwargs: lambda fn: fn)
    return current


@pytest.mark.parametrize('family', ['conv', 'gdn', 'kda'])
@pytest.mark.parametrize('pcg', [False, True])
@pytest.mark.parametrize('active', [False, True])
@pytest.mark.parametrize('decoding', [False, True])
def test_forward_routes_live_metadata(monkeypatch, execution, family, pcg, active, decoding):
    tensor = torch.zeros(1, 8, 4)
    if family == 'conv':
        impl = conv.CausalConv1dTilelangImpl(indexed_state=True)
        args = [tensor, tensor, None, tensor, None, 'silu']
        meta_index, token_indices = 4, [0]
    elif family == 'gdn':
        impl = gdn.CudaGatedDeltaRuleImpl(indexed_state=True)
        args = [tensor] * 8 + [None, 1, False]
        meta_index, token_indices = 8, range(5)
    else:
        impl = kda.CudaKdaImpl(indexed_state=True)
        args = [tensor] * 9 + [None, 1, 4, -5.0]
        meta_index, token_indices = 9, range(3)
    calls = []
    if family == 'conv':
        monkeypatch.setattr(impl, 'forward_indexed', lambda *args: calls.append(args) or args[0])
    else:
        monkeypatch.setattr(type(impl), 'forward', lambda self, *args: calls.append(args) or args[0])
    method = impl.forward.__func__
    if pcg:
        impl.enable_piecewise_cuda_graph()
        wrapper = impl._piecewise_forward
        impl.enable_piecewise_cuda_graph()
        assert impl._piecewise_forward is wrapper
    if pcg:
        assert impl.forward is wrapper
    else:
        assert 'forward' not in vars(impl) and impl.forward.__func__ is method
    for phase in (decoding, not decoding):
        meta = SimpleNamespace(is_decoding=phase)
        args[meta_index] = meta
        execution.value = SimpleNamespace(raw_tokens=3) if active else None
        result = impl.forward(*args)
        assert calls[-1][meta_index] is meta
        assert result.shape[1] == (3 if pcg and active else 8)
        for index in token_indices:
            assert calls[-1][index].shape[1] == result.shape[1]
    assert len(calls) == 2


@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('decoding', [False, True])
def test_conv_explicit_layout_and_local_phase(monkeypatch, indexed, decoding):
    impl = conv.CausalConv1dTilelangImpl(indexed_state=indexed)
    monkeypatch.setattr(impl, 'forward_indexed', lambda *args: 'indexed')
    monkeypatch.setattr(impl, 'decode', lambda *args: 'decode')
    monkeypatch.setattr(impl, '_prefill', lambda *args: 'prefill')
    meta = SimpleNamespace(is_decoding=decoding)
    assert impl.forward(None, None, None, None, meta, 'silu') == (
        'indexed' if indexed else 'decode' if decoding else 'prefill')
    assert 'forward' not in vars(impl)


@pytest.mark.parametrize('decoding', [False, True])
def test_gdn_nonindexed_retains_base_dispatch(monkeypatch, decoding):
    impl = gdn.CudaGatedDeltaRuleImpl(indexed_state=False)
    monkeypatch.setattr(impl, 'decode', lambda *args: 'decode')
    monkeypatch.setattr(impl, 'prefill', lambda *args: 'prefill')
    meta = SimpleNamespace(is_decoding=decoding)
    assert impl.forward(*([None] * 8), meta, 1, False) == ('decode' if decoding else 'prefill')
    assert 'forward' not in vars(impl)


@pytest.mark.parametrize('decoding', [False, True])
def test_dao_initializes_dispatch_without_tilelang_init(monkeypatch, execution, decoding):
    monkeypatch.setitem(sys.modules, 'causal_conv1d', SimpleNamespace(
        causal_conv1d_fn=lambda *args: None, causal_conv1d_update=lambda *args: None))
    def forbidden(*args, **kwargs):
        raise AssertionError('DAO must not initialize TileLang kernels')
    monkeypatch.setattr(conv.CausalConv1dTilelangImpl, '__init__', forbidden)
    impl = conv.CausalConv1dDaoImpl()
    assert impl.indexed_state is False
    monkeypatch.setattr(impl, 'decode', lambda *args: 'decode')
    monkeypatch.setattr(impl, '_prefill', lambda *args: 'prefill')
    impl.enable_piecewise_cuda_graph()
    assert impl.forward(None, None, None, None, SimpleNamespace(is_decoding=decoding), 'silu') == (
        'decode' if decoding else 'prefill')
    assert impl.forward is impl._piecewise_forward


@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('pcg', [False, True])
@pytest.mark.parametrize('active', [False, True])
def test_metadata_build_selection_and_live_context(monkeypatch, execution, indexed, pcg, active):
    context = SimpleNamespace(ssm_state_meta=None, ssm_prefill_meta=None)
    monkeypatch.setattr(gdn, 'get_step_ctx_manager', lambda: SimpleNamespace(current_context=lambda: context))
    monkeypatch.setattr(gdn, 'GatedDeltaMeta', lambda *args: ('legacy', args))
    monkeypatch.setattr(gdn, 'CudaCompactGatedDeltaMeta', lambda *args: ('indexed', args))
    impl = gdn.CudaGatedDeltaMetaImpl(indexed_state=indexed)
    if pcg:
        impl.enable_piecewise_cuda_graph()
        wrapper = impl._piecewise_forward
        impl.enable_piecewise_cuda_graph()
        assert impl._piecewise_forward is wrapper
    assert 'forward' not in vars(impl) and '_build' not in vars(impl)
    for decoding in (False, True):
        context.ssm_state_meta = object()
        context.ssm_prefill_meta = None if decoding else object()
        attn = SimpleNamespace(is_decoding=decoding)
        execution.value = SimpleNamespace(raw_tokens=3) if active else None
        kind, args = impl.forward(8, 4, None, attn)
        assert kind == ('indexed' if indexed else 'legacy')
        assert args[0] == (3 if pcg and active else 8)
        if indexed:
            assert args[1] is decoding
            assert args[2] is context.ssm_state_meta and args[3] is context.ssm_prefill_meta
        else:
            assert args[1:] == (4, None, attn)


@pytest.mark.parametrize('indexed', [False, True])
def test_kda_build_context_selects_callable_layout(monkeypatch, indexed):
    import lmdeploy.pytorch.nn.kda as module
    from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
    from lmdeploy.pytorch.backends.kda import KdaBuildSpec

    impl = CudaOpsBackend.build_op(KdaBuildSpec(indexed_state=indexed))
    assert impl.indexed_state is indexed
    assert impl.get_step_metadata_provider().indexed_state is indexed
    assert 'forward' not in vars(impl)
    for name in ('_conv', '_forward_spec', '_forward_spec_decode'):
        assert callable(getattr(impl, name))
    assert impl.causal_conv1d_fwd and impl.causal_conv1d_update and impl.chunk_kda
    assert impl.shared_conv_update and impl.fused_recurrent_kda

    def build(spec, **kwargs):
        assert spec.indexed_state is indexed
        return impl

    monkeypatch.setattr(module, 'get_backend', lambda: SimpleNamespace(build_op=build))
    monkeypatch.setattr(module, 'get_build_model_context', lambda: SimpleNamespace(
        state_cache_is_indexed=indexed, enable_deterministic=False))
    layer = module.Kda()
    assert layer.impl is impl

    if indexed:
        monkeypatch.setattr(impl, 'forward_indexed', lambda *args: 'indexed')
        assert impl.forward(*([None] * 10), 1, 128, -5.0) == 'indexed'
    else:
        def spec_decode(*args, **kwargs):
            return 'legacy-spec'
        monkeypatch.setattr(impl, '_forward_spec_decode', spec_decode)
        # Go through the retained _forward_spec rather than an isolated helper.
        meta = SimpleNamespace(num_spec_tokens=5, is_decoding=True)
        assert impl.forward(*([None] * 9), meta, 1, 128, -5.0) == 'legacy-spec'
