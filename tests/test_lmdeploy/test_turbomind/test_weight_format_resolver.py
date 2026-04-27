# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for WeightFormatResolver dispatch logic.

Uses a lightweight fake WeightFormat subclass that stubs out
``make_data_format`` so the resolver can be exercised without the real
``_turbomind`` extension.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

import pytest
import torch

# ---------------------------------------------------------------------------
# _turbomind stub (same pattern as test_transform_tensors.py)
# ---------------------------------------------------------------------------

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def _setup_fake_tm():
    """Ensure ``_turbomind`` in sys.modules has every attribute the resolver
    and weight_format class bodies touch.

    Idempotent: augments whatever is
    already there so running after test_transform_tensors.py (which sets up
    a minimal stub) still leaves a usable module.
    """
    tm = sys.modules.get('_turbomind')
    if tm is None:
        tm = types.ModuleType('_turbomind')
        sys.modules['_turbomind'] = tm

    dt = getattr(tm, 'DataType', None)
    if dt is None:
        class DataType:
            pass
        dt = DataType
        tm.DataType = dt

    # Class-body references in weight_format.py and builder/_base.py
    # (_STR_TO_DTYPE, _TORCH_TO_CPP) need these specific names present at
    # module load time.
    for name, val in (('TYPE_FP32', 0), ('TYPE_FP16', 1), ('TYPE_BF16', 2),
                      ('TYPE_INVALID', 3), ('TYPE_INT32', 4),
                      ('TYPE_INT64', 5), ('TYPE_INT8', 6),
                      ('TYPE_UINT8', 7), ('TYPE_UINT4', 10),
                      ('TYPE_FP8_E4M3', 11), ('TYPE_FP4_E2M1', 12)):
        if not hasattr(dt, name):
            setattr(dt, name, val)

    if not hasattr(tm, 'ResolveLinearWeightFormat'):
        tm.ResolveLinearWeightFormat = lambda d, w, bi, bo: ('DataFormat', d, w, bi, bo)


_setup_fake_tm()

# Register package stubs.
import lmdeploy  # noqa: F401

for _pkg in ('lmdeploy.turbomind',):
    if _pkg not in sys.modules:
        mod = types.ModuleType(_pkg)
        mod.__path__ = [os.path.join(_repo_root, *_pkg.split('.'))]
        mod.__package__ = _pkg
        sys.modules[_pkg] = mod


def _load(mod_name, file_rel_path):
    path = os.path.join(_repo_root, *file_rel_path.split('/'))
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_linear_mod = _load('lmdeploy.turbomind.linear',
                    'lmdeploy/turbomind/linear.py')
_wf_mod = _load('lmdeploy.turbomind.weight_format',
                'lmdeploy/turbomind/weight_format.py')

Linear = _linear_mod.Linear
WeightFormat = _wf_mod.WeightFormat
WeightFormatResolver = _wf_mod.WeightFormatResolver


# ---------------------------------------------------------------------------
# Fake format used by the tests
# ---------------------------------------------------------------------------


class _FakeQuant(WeightFormat):
    """Accepts when a ``.qfoo`` tensor is present.

    ``normalize`` is identity.
    """
    name = 'fakeq'
    suffix_map = {'.qfoo': 'weight', '.scales': 'scales', '.bias': 'bias'}
    weight_dtype = 0  # TYPE_FP32 from our stub
    has_zero_point = False

    def __init__(self, *, block_in=None, block_out=None):
        super().__init__(block_in=block_in, block_out=block_out)

    def accepts(self, available):
        return '.qfoo' in available

    def normalize(self, x, kind):
        return x


class _FakeQuantWithZeros(_FakeQuant):
    name = 'fakeqz'
    suffix_map = {'.qfoo': 'weight', '.scales': 'scales',
                  '.qzeros': 'zeros', '.bias': 'bias'}
    has_zero_point = True

    def synthesize_zeros(self, scales):
        return torch.zeros_like(scales)


class _FakeTrivial(WeightFormat):
    name = 'faketr'
    suffix_map = {'.weight': 'weight', '.bias': 'bias'}
    weight_dtype = None
    has_zero_point = False

    def accepts(self, available):
        return available.keys() <= {'.weight', '.bias'} and '.weight' in available

    def normalize(self, x, kind):
        return x

    def dequant(self, tensors, data_type):
        return tensors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolveQuantized:

    def _make_resolver(self):
        return WeightFormatResolver(
            data_type=0,
            formats=[_FakeQuant(), _FakeTrivial()])

    def test_quant_prefix_picks_quant_format(self):
        params = {
            'layer.qfoo':   torch.randn(4, 4),
            'layer.scales': torch.randn(1, 4),
        }
        lin = self._make_resolver().resolve(params, 'layer')
        assert isinstance(lin.weight_format, _FakeQuant)
        assert set(lin.tensors) == {'weight', 'scales'}

    def test_trivial_prefix_falls_through(self):
        params = {'layer.weight': torch.randn(4, 4)}
        lin = self._make_resolver().resolve(params, 'layer')
        assert isinstance(lin.weight_format, _FakeTrivial)


class TestResolveFailureModes:

    def _make_resolver(self):
        return WeightFormatResolver(
            data_type=0,
            formats=[_FakeQuant(), _FakeTrivial()])

    def test_missing_prefix_default_raises_key_error(self):
        with pytest.raises(KeyError, match='no checkpoint tensors found'):
            self._make_resolver().resolve({}, 'missing.prefix')

    def test_missing_prefix_optional_returns_none(self):
        assert self._make_resolver().resolve(
            {}, 'missing.prefix', optional=True) is None

    def test_tensors_present_no_match_raises_value_error(self):
        class _PickyTrivial(_FakeTrivial):
            def accepts(self, available):
                return False

        resolver = WeightFormatResolver(
            data_type=0,
            formats=[_FakeQuant(), _PickyTrivial()])
        params = {'layer.weight': torch.randn(4, 4)}
        with pytest.raises(ValueError, match='no weight format accepts'):
            resolver.resolve(params, 'layer')


class TestIndexedProbe:

    def test_index_slices_available_tensors(self):
        resolver = WeightFormatResolver(
            data_type=0, formats=[_FakeTrivial()])
        params = {'experts.weight': torch.arange(24).reshape(3, 4, 2).float()}
        lin = resolver.resolve(params, 'experts', index=1)
        assert lin.tensors['weight'].shape == (4, 2)
        torch.testing.assert_close(
            lin.tensors['weight'],
            torch.arange(8, 16).reshape(4, 2).float())


class TestZerosSynthesis:

    def test_synthesize_zeros_called_when_missing(self):
        params = {
            'layer.qfoo':   torch.randn(4, 4),
            'layer.scales': torch.ones(1, 4),
        }
        resolver = WeightFormatResolver(
            data_type=0, formats=[_FakeQuantWithZeros()])
        lin = resolver.resolve(params, 'layer')
        assert 'zeros' in lin.tensors
        torch.testing.assert_close(
            lin.tensors['zeros'], torch.zeros(1, 4))

    def test_synthesize_zeros_skipped_when_present(self):
        scales   = torch.ones(1, 4)
        supplied = torch.full_like(scales, 5.0)
        params = {
            'layer.qfoo':   torch.randn(4, 4),
            'layer.scales': scales,
            'layer.qzeros': supplied,
        }
        resolver = WeightFormatResolver(
            data_type=0, formats=[_FakeQuantWithZeros()])
        lin = resolver.resolve(params, 'layer')
        torch.testing.assert_close(lin.tensors['zeros'], supplied)


class TestEquality:

    def test_same_class_same_blocks_equal(self):
        a = _FakeQuant(block_in=128)
        b = _FakeQuant(block_in=128)
        assert a == b
        assert hash(a) == hash(b)
        assert {a, b} == {a}

    def test_different_blocks_unequal(self):
        assert _FakeQuant(block_in=128) != _FakeQuant(block_in=64)

    def test_different_classes_unequal(self):
        assert _FakeQuant() != _FakeTrivial()
