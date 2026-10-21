# Copyright (c) OpenMMLab. All rights reserved.
"""Public FlashKDA dispatch and recurrent checkpoint buffer contracts."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends.cuda import kda as kda_module
from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl


@pytest.fixture
def impl():
    # Contract tests do not load the FLA/TileLang CUDA kernels.
    impl = CudaKdaImpl.__new__(CudaKdaImpl)
    impl.model_type = 'glm5_next'
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


@pytest.mark.parametrize('model_type,capability,dtype,dim,state_dtype,bound,is_cuda,hip,expected', [
    ('glm5_next', (9, 0), torch.bfloat16, 128, torch.float32, -5.0, True, None, True),
    ('glm5_next', (9, 0), torch.bfloat16, 128, torch.float32, 0.0, True, None, True),
    (None, (9, 0), torch.bfloat16, 128, torch.float32, -5.0, True, None, False),
    ('other', (9, 0), torch.bfloat16, 128, torch.float32, -5.0, True, None, False),
    ('glm5_next', (8, 0), torch.bfloat16, 128, torch.float32, -5.0, True, None, False),
    ('glm5_next', (10, 0), torch.bfloat16, 128, torch.float32, -5.0, True, None, False),
    ('glm5_next', (9, 0), torch.float16, 128, torch.float32, -5.0, True, None, False),
    ('glm5_next', (9, 0), torch.bfloat16, 64, torch.float32, -5.0, True, None, False),
    ('glm5_next', (9, 0), torch.bfloat16, 128, torch.bfloat16, -5.0, True, None, False),
    ('glm5_next', (9, 0), torch.bfloat16, 128, torch.float32, None, True, None, False),
    ('glm5_next', (9, 0), torch.bfloat16, 128, torch.float32, -6.0, True, None, False),
    ('glm5_next', (9, 0), torch.bfloat16, 128, torch.float32, float('nan'), True, None, False),
    ('glm5_next', (9, 0), torch.bfloat16, 128, torch.float32, -5.0, False, None, False),
    ('glm5_next', (9, 0), torch.bfloat16, 128, torch.float32, -5.0, True, 'rocm', False),
])
def test_dispatch_scope(impl, monkeypatch, model_type, capability, dtype, dim, state_dtype, bound, is_cuda, hip,
                        expected):
    impl.model_type = model_type
    q = SimpleNamespace(is_cuda=is_cuda, device=torch.device('cuda:2'), dtype=dtype, size=lambda axis: dim)
    get_capability = Mock(return_value=capability)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', get_capability)
    monkeypatch.setattr(torch.version, 'hip', hip)
    assert impl._use_flash_kda(q, SimpleNamespace(dtype=state_dtype), bound) is expected
    if get_capability.called:
        get_capability.assert_called_once_with(q.device)


@pytest.fixture
def public_api(impl, monkeypatch):
    calls = []

    def fwd(*, checkpoint_state=None, checkpoint_offsets=None, **kwargs):
        calls.append(dict(kwargs, checkpoint_state=checkpoint_state, checkpoint_offsets=checkpoint_offsets))
        kwargs['out'].copy_(kwargs['v'])
        kwargs['final_state'].copy_(kwargs['initial_state'] + 1)
        if checkpoint_state is not None:
            checkpoint_state.fill_(7)

    monkeypatch.setitem(sys.modules, 'flash_kda', SimpleNamespace(fwd=fwd))
    monkeypatch.setattr(impl, '_use_flash_kda', lambda *args: True)
    monkeypatch.setattr(torch.ops.flash_kda, 'fwd', Mock(side_effect=AssertionError('private operator bypass')),
                        raising=False)
    return calls


@pytest.mark.parametrize('capture', [False, True])
@pytest.mark.parametrize('seqlen_dtype', [torch.int32, torch.int64])
def test_one_public_call_for_ordinary_and_checkpoint_prefill(impl, inputs, public_api, capture, seqlen_dtype,
                                                            monkeypatch):
    monkeypatch.setenv('FLA_FLASH_KDA', '0')
    inputs['cu_seqlens'] = inputs['cu_seqlens'].to(seqlen_dtype)
    state = torch.empty_like(inputs['initial_state']) if capture else None
    offsets = torch.tensor([16, 16], dtype=seqlen_dtype) if capture else None
    output, final = impl._prefill_recurrent(**inputs, checkpoint_state=state, checkpoint_offsets=offsets)
    assert len(public_api) == 1
    call = public_api[0]
    assert call['checkpoint_state'] is state
    assert call['checkpoint_offsets'] is offsets
    assert call['cu_seqlens'] is inputs['cu_seqlens']
    assert call['out'] is output and call['final_state'] is final
    assert call['dt_bias'].shape == (2, 128)
    assert call['scale'] == 128 ** -0.5
    assert call['lower_bound'] == -5.0
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
    with pytest.raises(ValueError, match='requires GLM-5.3'):
        impl._prefill_recurrent(**inputs, checkpoint_state=torch.empty_like(inputs['initial_state']),
                               checkpoint_offsets=torch.tensor([16, 16], dtype=torch.int32))
    assert impl.chunk_kda.call_count == 1


@pytest.mark.parametrize('only_state', [False, True])
def test_checkpoint_arguments_must_be_paired(impl, inputs, public_api, only_state):
    kwargs = dict(checkpoint_state=torch.empty_like(inputs['initial_state'])) if only_state else dict(
        checkpoint_offsets=torch.tensor([16, 16], dtype=torch.int32))
    with pytest.raises(ValueError, match='provided together'):
        impl._prefill_recurrent(**inputs, **kwargs)
    assert not public_api


@pytest.mark.parametrize('missing', [False, True])
def test_incompatible_dependency_fails_before_cache_update(impl, monkeypatch, missing):
    module = None if missing else SimpleNamespace(fwd=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, 'flash_kda', module)
    monkeypatch.setattr(impl, '_use_flash_kda', lambda *args: True)
    impl._conv = Mock(side_effect=AssertionError('cache modified before dependency check'))
    with pytest.raises(ImportError if missing else RuntimeError, match='FlashKDA|flash_kda.fwd'):
        impl.forward(**forward_inputs())
    impl._conv.assert_not_called()
    impl.chunk_kda.assert_not_called()


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


@pytest.mark.parametrize('spec', [0, 1, 3, 5])
@pytest.mark.parametrize('capture', [False, True])
def test_forward_and_mtp_prefill_use_public_api(impl, public_api, monkeypatch, spec, capture):
    args = forward_inputs(spec)
    before = args['recurrent_state'].clone()
    monkeypatch.setattr(kda_module, '_state_select', lambda state, ids, slots: state[ids, slots].clone())

    def scatter(state, ids, slots, values):
        state[ids, slots] = values

    monkeypatch.setattr(kda_module, '_state_scatter', scatter)

    def conv(x, weight, bias, state, metadata):
        state[metadata.state_ids] = x[:, -4:].transpose(1, 2)
        return x

    impl._conv = conv
    # The checkpoint owner supplies the destination slot; runtime ring writes
    # must continue to address the full-prefill end, not the checkpoint offset.
    checkpoint_ring = torch.full((1, spec + 1, 1, 128, 128), -1.0)
    checkpoint = checkpoint_ring[:, (17 + 16) % (spec + 1)] if capture else None
    offsets = torch.tensor([16]) if capture else None
    out = impl.forward(**args, checkpoint_state=checkpoint, checkpoint_offsets=offsets)
    assert out.shape == (1, 48, 1, 128)
    assert len(public_api) == 1
    assert public_api[0]['checkpoint_state'] is checkpoint
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


def test_build_spec_passes_model_scope(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
    from lmdeploy.pytorch.backends.kda import KdaBuildSpec

    constructor = Mock()
    monkeypatch.setattr(kda_module, 'CudaKdaImpl', constructor)
    CudaOpsBackend.build_op(KdaBuildSpec(model_type='glm5_next'))
    constructor.assert_called_once_with(model_type='glm5_next')
    assert KdaBuildSpec().model_type is None


def test_nn_forwards_checkpoint_buffers():
    from lmdeploy.pytorch.nn.kda import Kda

    layer = Kda.__new__(Kda)
    torch.nn.Module.__init__(layer)
    layer.impl = SimpleNamespace(forward=Mock(return_value=torch.empty(0)))
    state, offsets = torch.empty(1, 1, 128, 128), torch.tensor([16])
    layer(**forward_inputs(), checkpoint_state=state, checkpoint_offsets=offsets)
    assert layer.impl.forward.call_args.kwargs['checkpoint_state'] is state
    assert layer.impl.forward.call_args.kwargs['checkpoint_offsets'] is offsets
