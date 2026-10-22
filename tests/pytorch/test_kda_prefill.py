# Copyright (c) OpenMMLab. All rights reserved.
"""Native FlashKDA dispatch and recurrent checkpoint buffer contracts."""
import builtins
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends.cuda import kda as kda_module
from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl


@pytest.fixture(autouse=True)
def clear_device_cache():
    kda_module._supports_flash_kda_device.cache_clear()
    yield
    kda_module._supports_flash_kda_device.cache_clear()


@pytest.fixture
def impl():
    # Contract tests do not load the FLA/TileLang CUDA kernels.
    impl = CudaKdaImpl.__new__(CudaKdaImpl)
    impl.chunk_kda = Mock(side_effect=AssertionError('unexpected FLA dispatch'))
    return impl


@pytest.fixture
def inputs():
    heads, dim, tokens = 2, 128, 48
    q, k, v = torch.randn(1, tokens, 3 * heads * dim, dtype=torch.bfloat16).split(heads * dim, dim=-1)
    return dict(q=q.unflatten(-1, (heads, dim)), k=k.unflatten(-1, (heads, dim)),
                v=v.unflatten(-1, (heads, dim)), g=torch.randn(1, tokens, heads, dim, dtype=torch.bfloat16),
                beta=torch.randn(1, tokens, heads, dtype=torch.bfloat16),
                A_log=torch.zeros(heads), dt_bias=torch.zeros(heads * dim),
                initial_state=torch.randn(2, heads, dim, dim),
                cu_seqlens=torch.tensor([0, 16, tokens], dtype=torch.int32), lower_bound=-5.0)


@pytest.mark.parametrize('capability,dtype,dim,state_dtype,bound,is_cuda,hip,expected', [
    ((9, 0), torch.bfloat16, 128, torch.float32, -5.0, True, None, True),
    ((9, 0), torch.bfloat16, 128, torch.float32, 0.0, True, None, True),
    ((8, 0), torch.bfloat16, 128, torch.float32, -5.0, True, None, False),
    ((10, 0), torch.bfloat16, 128, torch.float32, -5.0, True, None, False),
    ((9, 0), torch.float16, 128, torch.float32, -5.0, True, None, False),
    ((9, 0), torch.bfloat16, 64, torch.float32, -5.0, True, None, False),
    ((9, 0), torch.bfloat16, 128, torch.bfloat16, -5.0, True, None, False),
    ((9, 0), torch.bfloat16, 128, torch.float32, None, True, None, False),
    ((9, 0), torch.bfloat16, 128, torch.float32, -6.0, True, None, False),
    ((9, 0), torch.bfloat16, 128, torch.float32, float('nan'), True, None, False),
    ((9, 0), torch.bfloat16, 128, torch.float32, -5.0, False, None, False),
    ((9, 0), torch.bfloat16, 128, torch.float32, -5.0, True, 'rocm', False),
])
def test_dispatch_scope(impl, monkeypatch, capability, dtype, dim, state_dtype, bound, is_cuda, hip, expected):
    q = SimpleNamespace(is_cuda=is_cuda, device=torch.device('cuda:2'), dtype=dtype, size=lambda axis: dim)
    get_capability = Mock(return_value=capability)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', get_capability)
    monkeypatch.setattr(torch.version, 'hip', hip)
    assert impl._use_flash_kda(q, SimpleNamespace(dtype=state_dtype), bound) is expected
    if get_capability.called:
        get_capability.assert_called_once_with(q.device.index)


@pytest.mark.parametrize('capability,expected', [((9, 0), True), ((8, 0), False)])
def test_device_support_is_cached_across_impls(impl, monkeypatch, capability, expected):
    q = SimpleNamespace(is_cuda=True, device=torch.device('cuda:2'), dtype=torch.bfloat16, size=lambda axis: 128)
    state = SimpleNamespace(dtype=torch.float32)
    get_capability = Mock(return_value=capability)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', get_capability)
    monkeypatch.setattr(torch.version, 'hip', None)
    other = CudaKdaImpl.__new__(CudaKdaImpl)
    for backend in (impl, other, impl):
        assert backend._use_flash_kda(q, state, -5.0) is expected
    get_capability.assert_called_once_with(2)

    # A second device gets its own entry; neither backend instance owns the cache.
    q.device = torch.device('cuda:3')
    for backend in (other, impl):
        assert backend._use_flash_kda(q, state, -5.0) is expected
    assert get_capability.call_count == 2
    get_capability.assert_called_with(3)


@pytest.fixture
def native_api(impl, monkeypatch):
    calls = []

    def fwd(*, checkpoint_state=None, checkpoint_offsets=None, **kwargs):
        calls.append(dict(kwargs, checkpoint_state=checkpoint_state, checkpoint_offsets=checkpoint_offsets))
        kwargs['out'].copy_(kwargs['v'])
        kwargs['final_state'].copy_(kwargs['initial_state'] + 1)
        if checkpoint_state is not None:
            checkpoint_state.fill_(7)

    fwd._schema = torch._C.parse_schema('fwd(Tensor? checkpoint_state=None, Tensor? checkpoint_offsets=None) -> ()')
    wrapper = Mock(side_effect=AssertionError('unexpected Python wrapper call'))
    monkeypatch.setitem(sys.modules, 'flash_kda', SimpleNamespace(_C=object(), fwd=wrapper))
    monkeypatch.setattr(torch.ops, 'flash_kda', SimpleNamespace(fwd=SimpleNamespace(default=fwd),
                                                             get_workspace_size=Mock(return_value=257)))
    monkeypatch.setattr(impl, '_use_flash_kda', lambda *args: True)
    return calls


@pytest.mark.parametrize('capture', [False, True])
@pytest.mark.parametrize('seqlen_dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('varlen', [False, True])
def test_one_native_call_for_ordinary_and_checkpoint_prefill(impl, inputs, native_api, capture, seqlen_dtype,
                                                            varlen, monkeypatch):
    monkeypatch.setenv('FLA_FLASH_KDA', '0')
    inputs['cu_seqlens'] = inputs['cu_seqlens'].to(seqlen_dtype) if varlen else None
    if not varlen:
        for name in ('q', 'k', 'v', 'g', 'beta'):
            value = inputs[name]
            inputs[name] = value.reshape(2, 24, *value.shape[2:])
    state = torch.empty_like(inputs['initial_state']) if capture else None
    offsets = torch.tensor([16, 16], dtype=seqlen_dtype if varlen else torch.int64) if capture else None
    output, final = impl._prefill_recurrent(**inputs, checkpoint_state=state, checkpoint_offsets=offsets)
    assert len(native_api) == 1
    call = native_api[0]
    assert call['checkpoint_state'] is state
    assert call['checkpoint_offsets'] is offsets
    assert call['cu_seqlens'] is inputs['cu_seqlens']
    assert call['out'] is output and call['final_state'] is final
    assert call['dt_bias'].shape == (2, 128)
    assert call['scale'] == 128 ** -0.5
    assert call['lower_bound'] == -5.0
    torch.ops.flash_kda.get_workspace_size.assert_called_once_with(48, 2, 2)
    assert call['workspace'].dtype == torch.uint8
    assert call['workspace'].shape == (257,)
    assert call['workspace'].device == inputs['q'].device
    assert call['workspace'].is_contiguous()
    sys.modules['flash_kda'].fwd.assert_not_called()
    for name in ('q', 'k', 'v', 'g', 'beta'):
        torch.testing.assert_close(call[name], inputs[name], rtol=0, atol=0)
    for name in ('q', 'k', 'v', 'g', 'out', 'initial_state', 'final_state'):
        assert call[name].is_contiguous()
    torch.testing.assert_close(output, inputs['v'], rtol=0, atol=0)
    torch.testing.assert_close(final, inputs['initial_state'] + 1, rtol=0, atol=0)
    if capture:
        assert torch.all(state == 7)
    impl.chunk_kda.assert_not_called()


def test_outside_scope_preserves_fla_contract(impl, inputs, monkeypatch):
    monkeypatch.setattr(impl, '_use_flash_kda', lambda *args: False)
    monkeypatch.setitem(sys.modules, 'flash_kda', None)
    expected = (inputs['v'], inputs['initial_state'])
    impl.chunk_kda = Mock(return_value=expected)
    assert impl._prefill_recurrent(**inputs) is expected
    kwargs = impl.chunk_kda.call_args.kwargs
    for name in ('output_final_state', 'use_qk_l2norm_in_kernel', 'use_gate_in_kernel',
                 'use_beta_sigmoid_in_kernel', 'safe_gate', 'state_v_first'):
        assert kwargs[name] is True
    assert kwargs['cu_seqlens'] is inputs['cu_seqlens']
    assert kwargs['dt_bias'] is inputs['dt_bias']
    with pytest.raises(ValueError, match='requires SM90'):
        impl._prefill_recurrent(**inputs, checkpoint_state=torch.empty_like(inputs['initial_state']),
                               checkpoint_offsets=torch.tensor([16, 16], dtype=torch.int32))
    assert impl.chunk_kda.call_count == 1


@pytest.mark.parametrize('only_state', [False, True])
def test_checkpoint_arguments_must_be_paired(impl, inputs, native_api, only_state):
    kwargs = dict(checkpoint_state=torch.empty_like(inputs['initial_state'])) if only_state else dict(
        checkpoint_offsets=torch.tensor([16, 16], dtype=torch.int32))
    with pytest.raises(ValueError, match='provided together'):
        impl._prefill_recurrent(**inputs, **kwargs)
    assert not native_api


@pytest.fixture(params=['package', 'extension', 'operator', 'checkpoint', 'load_os', 'load_runtime'])
def incompatible_native(impl, monkeypatch, request):
    failure = request.param
    module = None if failure == 'package' else SimpleNamespace(fwd=lambda **kwargs: None)
    if failure in ('operator', 'checkpoint'):
        module._C = object()
    namespace = SimpleNamespace()
    if failure == 'checkpoint':
        namespace.fwd = SimpleNamespace(default=SimpleNamespace(_schema=torch._C.parse_schema('fwd(Tensor q) -> ()')))
    monkeypatch.setattr(torch.ops, 'flash_kda', namespace)
    monkeypatch.setitem(sys.modules, 'flash_kda', module)
    monkeypatch.setattr(impl, '_use_flash_kda', lambda *args: True)
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == 'flash_kda' and failure.startswith('load_'):
            error = OSError if failure == 'load_os' else RuntimeError
            raise error('incompatible CUDA/PyTorch extension')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', import_module)
    return RuntimeError if failure in ('operator', 'checkpoint') else ImportError


@pytest.mark.parametrize('spec', [0, 3])
def test_incompatible_dependency_fails_before_checkpoint_cache_update(impl, incompatible_native, spec):
    impl._conv = Mock(side_effect=AssertionError('cache modified before dependency check'))
    impl._forward_spec = Mock(side_effect=AssertionError('ring accessed before dependency check'))
    args = forward_inputs(spec)
    before = (args['conv_state'].clone(), args['recurrent_state'].clone())
    with pytest.raises(incompatible_native, match='FlashKDA|flash_kda.fwd'):
        impl.forward(**args, checkpoint_state=torch.empty(1, 1, 128, 128), checkpoint_offsets=torch.tensor([16]))
    impl._conv.assert_not_called()
    impl._forward_spec.assert_not_called()
    impl.chunk_kda.assert_not_called()
    torch.testing.assert_close(args['conv_state'], before[0], rtol=0, atol=0)
    torch.testing.assert_close(args['recurrent_state'], before[1], rtol=0, atol=0)


def forward_inputs(spec=0, decode=False):
    heads, dim, tokens, width = 1, 128, 48, 4
    return dict(mixed_qkv=torch.randn(1, tokens, 3 * heads * dim, dtype=torch.bfloat16),
                raw_gate=torch.randn(1, tokens, heads * dim, dtype=torch.bfloat16),
                raw_beta=torch.randn(1, tokens, heads, dtype=torch.bfloat16),
                conv_weight=torch.zeros(3 * heads * dim, width), conv_bias=None,
                a_log=torch.zeros(heads), dt_bias=torch.zeros(heads * dim),
                conv_state=torch.zeros(3, 3 * heads * dim, width + spec, dtype=torch.bfloat16),
                recurrent_state=torch.zeros((3, spec + 1, heads, dim, dim) if spec else (3, heads, dim, dim)),
                metadata=SimpleNamespace(is_decoding=decode, num_spec_tokens=spec, state_ids=torch.tensor([1]),
                                         valid_state=torch.tensor([True]), is_init=torch.tensor([False]),
                                         cache_seqlens=torch.tensor([17]), cu_seqlens=torch.tensor([0, tokens])),
                num_heads=heads, head_dim=dim, lower_bound=-5.0)


@pytest.fixture
def cpu_cache_ops(impl, monkeypatch):
    monkeypatch.setattr(kda_module, '_state_select', lambda state, ids, slots: state[ids, slots].clone())

    def scatter(state, ids, slots, values):
        state[ids, slots] = values

    monkeypatch.setattr(kda_module, '_state_scatter', scatter)

    def conv(x, weight, bias, state, metadata):
        state[metadata.state_ids] = x[:, -4:].transpose(1, 2)
        return x

    impl._conv = conv


@pytest.mark.parametrize('spec', [0, 3])
def test_ordinary_prefill_falls_back_with_incompatible_dependency(impl, incompatible_native, cpu_cache_ops, spec):
    args = forward_inputs(spec)
    before = args['recurrent_state'].clone()
    impl.chunk_kda = Mock(side_effect=lambda **kwargs: (kwargs['v'], kwargs['initial_state'] + 1))
    out = impl.forward(**args)
    torch.testing.assert_close(out, args['mixed_qkv'][..., 256:].unsqueeze(2), rtol=0, atol=0)
    kwargs = impl.chunk_kda.call_args.kwargs
    for name in ('output_final_state', 'use_qk_l2norm_in_kernel', 'use_gate_in_kernel',
                 'use_beta_sigmoid_in_kernel', 'safe_gate', 'state_v_first'):
        assert kwargs[name] is True
    assert 'checkpoint_state' not in kwargs and 'checkpoint_offsets' not in kwargs
    if spec:
        before[1, (17 + 48) % (spec + 1)] += 1
    else:
        before[1] += 1
    torch.testing.assert_close(args['recurrent_state'], before, rtol=0, atol=0)
    # A failed optional lookup is cached instead of retried at every layer call.
    assert impl.__dict__['_optional_flash_kda_fwd'] is None


def test_native_execution_error_does_not_retry_fla(impl, inputs, native_api):
    fwd = Mock(side_effect=RuntimeError('native kernel failed'))
    impl._flash_kda_fwd = fwd
    with pytest.raises(RuntimeError, match='native kernel failed'):
        impl._prefill_recurrent(**inputs)
    impl.chunk_kda.assert_not_called()
    fwd.assert_called_once()


@pytest.mark.parametrize('spec', [0, 1, 3, 5])
@pytest.mark.parametrize('capture', [False, True])
def test_forward_and_mtp_prefill_use_native_api(impl, native_api, cpu_cache_ops, spec, capture):
    args = forward_inputs(spec)
    before = args['recurrent_state'].clone()
    # The checkpoint owner supplies the destination slot; runtime ring writes
    # must continue to address the full-prefill end, not the checkpoint offset.
    checkpoint_ring = torch.full((1, spec + 1, 1, 128, 128), -1.0)
    checkpoint = checkpoint_ring[:, (17 + 16) % (spec + 1)] if capture else None
    offsets = torch.tensor([16]) if capture else None
    out = impl.forward(**args, checkpoint_state=checkpoint, checkpoint_offsets=offsets)
    assert out.shape == (1, 48, 1, 128)
    assert len(native_api) == 1
    assert native_api[0]['checkpoint_state'] is checkpoint
    if capture:
        assert torch.all(checkpoint == 7)
        for slot in range(spec + 1):
            if slot != (17 + 16) % (spec + 1):
                assert torch.all(checkpoint_ring[:, slot] == -1)
    if spec:
        before[1, (17 + 48) % (spec + 1)] += 1
    else:
        before[1] += 1
    torch.testing.assert_close(args['recurrent_state'], before, rtol=0, atol=0)


@pytest.mark.parametrize('spec', [0, 3])
@pytest.mark.parametrize('invalid,match', [
    ('state_dtype', 'checkpoint_state.*float32'),
    ('state_shape', 'checkpoint_state.*shape'),
    ('state_device', 'checkpoint_state.*device'),
    ('state_ring_view', 'checkpoint_state.*contiguous'),
    ('offset_dtype', 'checkpoint_offsets.*dtype'),
    ('offset_shape', 'checkpoint_offsets.*shape'),
    ('offset_device', 'checkpoint_offsets.*device'),
    ('offset_strides', 'checkpoint_offsets.*contiguous'),
    ('cu_dtype', 'cu_seqlens'),
    ('cu_shape', 'cu_seqlens'),
    ('cu_device', 'cu_seqlens'),
    ('cu_strides', 'cu_seqlens'),
])
def test_invalid_checkpoint_rejected_before_cache_update(impl, native_api, spec, invalid, match):
    args = forward_inputs(spec)
    meta = args['metadata']
    meta.state_ids = torch.tensor([1, 2])
    meta.cu_seqlens = torch.tensor([0, 16, 48])
    checkpoint = torch.empty(2, 1, 128, 128)
    offsets = torch.tensor([16, 16])
    if invalid == 'state_dtype':
        checkpoint = checkpoint.bfloat16()
    elif invalid == 'state_shape':
        checkpoint = checkpoint[:1]
    elif invalid == 'state_device':
        checkpoint = checkpoint.to('meta')
    elif invalid == 'state_ring_view':
        # A common slot across multiple requests in [N, Ring, H, V, K] is strided.
        checkpoint = torch.empty(2, 4, 1, 128, 128)[:, 1]
    elif invalid == 'offset_dtype':
        offsets = offsets.int()
    elif invalid == 'offset_shape':
        offsets = offsets.unsqueeze(0)
    elif invalid == 'offset_device':
        offsets = offsets.to('meta')
    elif invalid == 'offset_strides':
        offsets = torch.empty(4, dtype=torch.int64)[::2]
    elif invalid == 'cu_dtype':
        meta.cu_seqlens = meta.cu_seqlens.float()
    elif invalid == 'cu_shape':
        meta.cu_seqlens = meta.cu_seqlens.unsqueeze(0)
    elif invalid == 'cu_device':
        meta.cu_seqlens = meta.cu_seqlens.to('meta')
    else:
        meta.cu_seqlens = torch.empty(6, dtype=torch.int64)[::2]
    before = (args['conv_state'].clone(), args['recurrent_state'].clone())
    impl._conv = Mock(side_effect=AssertionError('cache modified before validation'))
    impl._forward_spec = Mock(side_effect=AssertionError('ring accessed before validation'))
    with pytest.raises(ValueError, match=match):
        impl.forward(**args, checkpoint_state=checkpoint, checkpoint_offsets=offsets)
    impl._conv.assert_not_called()
    impl._forward_spec.assert_not_called()
    assert not native_api
    torch.testing.assert_close(args['conv_state'], before[0], rtol=0, atol=0)
    torch.testing.assert_close(args['recurrent_state'], before[1], rtol=0, atol=0)


def test_fixed_batch_checkpoint_requires_int64_offsets(impl, inputs, native_api):
    inputs['cu_seqlens'] = None
    with pytest.raises(ValueError, match='int64 for fixed-batch'):
        impl._prefill_recurrent(**inputs, checkpoint_state=torch.empty(1, 2, 128, 128),
                               checkpoint_offsets=torch.tensor([16], dtype=torch.int32))
    assert not native_api


@pytest.mark.parametrize('spec', [0, 1, 3, 5])
def test_decode_rejects_prefill_checkpoint_before_dispatch(impl, monkeypatch, spec):
    impl._conv = Mock(side_effect=AssertionError('unexpected convolution'))
    impl._forward_spec_decode = Mock(side_effect=AssertionError('unexpected verification'))
    monkeypatch.setitem(sys.modules, 'flash_kda', None)
    with pytest.raises(ValueError, match='only supported during prefill'):
        impl.forward(**forward_inputs(spec, decode=True), checkpoint_state=torch.empty(1))
    impl._conv.assert_not_called()
    impl._forward_spec_decode.assert_not_called()


@pytest.mark.parametrize('spec', [0, 3])
def test_decode_does_not_require_flash_kda(impl, monkeypatch, spec):
    monkeypatch.setitem(sys.modules, 'flash_kda', None)
    args = forward_inputs(spec, decode=True)
    for name in ('mixed_qkv', 'raw_gate', 'raw_beta'):
        args[name] = args[name][:, :1]
    expected = torch.zeros(1, 1, 1, 128, dtype=torch.bfloat16)
    impl._forward_spec_decode = Mock(return_value=expected)
    impl._conv = lambda x, *args: x
    monkeypatch.setattr(kda_module, '_select_state', lambda state, meta: state[meta.state_ids])
    monkeypatch.setattr(kda_module, '_store_state', Mock())
    impl.fused_recurrent_kda = Mock(return_value=(expected, args['recurrent_state'][1:2]))
    torch.testing.assert_close(impl.forward(**args), expected)
    assert impl._forward_spec_decode.call_count == bool(spec)
    assert impl.fused_recurrent_kda.call_count == (not spec)
    impl.chunk_kda.assert_not_called()


def test_build_spec_uses_default_backend(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
    from lmdeploy.pytorch.backends.kda import KdaBuildSpec

    constructor = Mock()
    monkeypatch.setattr(kda_module, 'CudaKdaImpl', constructor)
    CudaOpsBackend.build_op(KdaBuildSpec())
    constructor.assert_called_once_with()


def test_nn_forwards_checkpoint_buffers():
    from lmdeploy.pytorch.nn.kda import Kda

    layer = Kda.__new__(Kda)
    torch.nn.Module.__init__(layer)
    layer.impl = SimpleNamespace(forward=Mock(return_value=torch.empty(0)))
    state, offsets = torch.empty(1, 1, 128, 128), torch.tensor([16])
    layer(**forward_inputs(), checkpoint_state=state, checkpoint_offsets=offsets)
    assert layer.impl.forward.call_args.kwargs['checkpoint_state'] is state
    assert layer.impl.forward.call_args.kwargs['checkpoint_offsets'] is offsets
