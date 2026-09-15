# Copyright (c) OpenMMLab. All rights reserved.
import runpy
from dataclasses import replace

import pytest
import torch

from lmdeploy.pytorch import envs


@pytest.mark.parametrize('value,expected', [(None, 'cute'), ('auto', 'auto'), ('triton', 'triton'),
                                            ('cute', 'cute'), (' CUTE ', 'cute')])
def test_w4a16_moe_backend_choices(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv('LMDEPLOY_W4A16_MOE_BACKEND', raising=False)
    else:
        monkeypatch.setenv('LMDEPLOY_W4A16_MOE_BACKEND', value)
    # Evaluate the actual declaration without replacing the process's envs module.
    assert runpy.run_path(envs.__file__)['w4a16_moe_backend'] == expected


@pytest.mark.parametrize('value', ['cutew4a8', 'cutew4a16', 'cutew', 'w4a8', 'unknown'])
def test_w4a16_moe_backend_rejects_invalid_names(monkeypatch, value):
    monkeypatch.setenv('LMDEPLOY_W4A16_MOE_BACKEND', value)
    with pytest.raises(ValueError, match='LMDEPLOY_W4A16_MOE_BACKEND'):
        runpy.run_path(envs.__file__)


@pytest.fixture
def build_spec():
    from lmdeploy.pytorch.backends.moe import FusedMoEW4A16BuildSpec

    return FusedMoEW4A16BuildSpec(top_k=2, num_experts=8, renormalize=False, num_bits=4, group_size=32,
                                  hidden_dim=128, ep_size=1, ep_group=None, output_dtype=torch.bfloat16,
                                  num_max_dispatch_tokens_per_rank=128, layer_idx=0)


@pytest.mark.parametrize('compatible', [False, True])
@pytest.mark.parametrize('ep_size', [1, 2])
def test_auto_selects_compatible_provider(monkeypatch, build_spec, compatible, ep_size):
    from lmdeploy.pytorch.backends.cuda.moe import compressed_tensors as backend

    spec = replace(build_spec, ep_size=ep_size)
    selected = object()
    monkeypatch.setattr(envs, 'w4a16_moe_backend', 'auto')
    monkeypatch.setattr(backend, '_supports_cute', lambda spec: compatible)

    def build_cute(actual_spec):
        assert compatible and actual_spec is spec
        return selected

    def build_triton(**kwargs):
        assert not compatible
        assert kwargs['top_k'] == spec.top_k
        if ep_size > 1:
            assert kwargs['ep_size'] == ep_size
            assert kwargs.get('local_backend', 'triton') == 'triton'
        return selected

    monkeypatch.setattr(backend, '_build_fused_moe_cute', build_cute)
    monkeypatch.setattr(backend, 'TritonFusedMoEW4A16Impl', build_triton)
    monkeypatch.setattr(backend, 'DeepEPFusedMoEW4A16Impl', build_triton)
    assert backend._build_fused_moe_w4a16(spec) is selected


@pytest.mark.parametrize('provider', ['cute', 'triton'])
def test_explicit_provider_does_not_probe_auto(monkeypatch, build_spec, provider):
    from lmdeploy.pytorch.backends.cuda.moe import compressed_tensors as backend

    monkeypatch.setattr(envs, 'w4a16_moe_backend', provider)

    def unexpected_probe(spec):
        raise AssertionError('explicit provider must not probe automatic compatibility')

    monkeypatch.setattr(backend, '_supports_cute', unexpected_probe)
    if provider == 'triton':
        assert isinstance(backend._build_fused_moe_w4a16(build_spec), backend.TritonFusedMoEW4A16Impl)
    else:
        def unavailable_cute(spec):
            raise ImportError('optional dependency unavailable')

        monkeypatch.setattr(backend, '_build_fused_moe_cute', unavailable_cute)
        with pytest.raises(ImportError, match='optional dependency unavailable'):
            backend._build_fused_moe_w4a16(build_spec)


@pytest.mark.parametrize('overrides', [dict(num_bits=8), dict(group_size=128), dict(output_dtype=torch.float16),
                                       dict(hidden_dim=0), dict(hidden_dim=1), dict(hidden_dim=48)])
def test_cute_rejects_incompatible_build_shapes(monkeypatch, build_spec, overrides):
    from lmdeploy.pytorch.backends.cuda.moe import compressed_tensors as backend

    def unexpected_cuda_probe():
        raise AssertionError('reject incompatible formats before probing CUDA')

    monkeypatch.setattr(torch.cuda, 'is_available', unexpected_cuda_probe)
    assert not backend._supports_cute(replace(build_spec, **overrides))


@pytest.mark.parametrize('cuda_available,major', [(False, 9), (True, 8), (True, 10)])
def test_cute_rejects_incompatible_devices(monkeypatch, build_spec, cuda_available, major):
    from lmdeploy.pytorch.backends.cuda.moe import compressed_tensors as backend

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (major, 0))
    assert not backend._supports_cute(build_spec)


@pytest.mark.parametrize('provider', ['auto', 'cute'])
@pytest.mark.parametrize('cuda_available,major', [(False, 9), (True, 8), (True, 10)])
def test_builder_handles_unsupported_devices(monkeypatch, build_spec, provider, cuda_available, major):
    from lmdeploy.pytorch.backends.cuda.moe import compressed_tensors as backend

    monkeypatch.setattr(envs, 'w4a16_moe_backend', provider)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (major, 0))
    if provider == 'auto':
        assert isinstance(backend._build_fused_moe_w4a16(build_spec), backend.TritonFusedMoEW4A16Impl)
    else:
        with pytest.raises(RuntimeError, match='Hopper SM90'):
            backend._build_fused_moe_w4a16(build_spec)


@pytest.mark.parametrize('available', [False, True])
def test_cute_probes_optional_kernel_import(monkeypatch, build_spec, available):
    import builtins
    from types import SimpleNamespace

    from lmdeploy.pytorch.backends.cuda.moe import compressed_tensors as backend

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == 'lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16_cute':
            if not available:
                raise ImportError('optional CuTe dependency missing')
            return SimpleNamespace(fused_moe_w4a16_cute=object())
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (9, 0))
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    assert backend._supports_cute(build_spec) is available


def test_auto_does_not_hide_builder_errors(monkeypatch, build_spec):
    from lmdeploy.pytorch.backends.cuda.moe import compressed_tensors as backend

    monkeypatch.setattr(envs, 'w4a16_moe_backend', 'auto')
    monkeypatch.setattr(backend, '_supports_cute', lambda spec: True)

    def invalid_spec(spec):
        raise ValueError('invalid EP configuration')

    monkeypatch.setattr(backend, '_build_fused_moe_cute', invalid_spec)
    with pytest.raises(ValueError, match='invalid EP configuration'):
        backend._build_fused_moe_w4a16(build_spec)


def test_builder_rejects_unknown_provider(monkeypatch, build_spec):
    from lmdeploy.pytorch.backends.cuda.moe import compressed_tensors as backend

    monkeypatch.setattr(envs, 'w4a16_moe_backend', 'unknown')
    with pytest.raises(ValueError, match='Unsupported compressed-tensors W4A16 MoE provider'):
        backend._build_fused_moe_w4a16(build_spec)
