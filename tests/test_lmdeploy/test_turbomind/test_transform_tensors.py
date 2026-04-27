# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for the @transform_output_dim decorator."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

import torch

# ---------------------------------------------------------------------------
# Bootstrap: make _turbomind available as a lightweight stub so that
# ``lmdeploy.turbomind.linear`` and ``_base`` can be imported without
# the real C extension.
# ---------------------------------------------------------------------------

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def _setup_fake_tm():
    """Register a minimal ``_turbomind`` stub in ``sys.modules``."""
    if '_turbomind' in sys.modules:
        return

    tm = types.ModuleType('_turbomind')

    class DataType:
        TYPE_FP32 = 0
        TYPE_FP16 = 1
        TYPE_BF16 = 2
        TYPE_INVALID = 3
        TYPE_INT32 = 4
        TYPE_INT64 = 5
        TYPE_INT8 = 6
        TYPE_UINT8 = 7
        TYPE_UINT4 = 8
        TYPE_FP8_E4M3 = 9
        TYPE_FP4_E2M1 = 10

    tm.DataType = DataType

    # Stub functions / classes referenced throughout turbomind/
    tm.create_module = lambda cfg: None
    tm.LinearConfig = type('LinearConfig', (), {})()
    tm.ResolveLinearWeightFormat = lambda *a, **kw: None

    sys.modules['_turbomind'] = tm


_setup_fake_tm()

# ---------------------------------------------------------------------------
# Import modules under test by loading their files directly so we avoid
# triggering the ``lmdeploy.turbomind`` package __init__ (which drags in
# the real TurboMind runtime).
# ---------------------------------------------------------------------------

# Ensure ``lmdeploy`` top-level is importable.
import lmdeploy  # noqa: F401

# Register the sub-package stubs so that relative imports resolve.
_turbomind_pkg = sys.modules.get('lmdeploy.turbomind')
if _turbomind_pkg is None:
    _turbomind_pkg = types.ModuleType('lmdeploy.turbomind')
    _turbomind_pkg.__path__ = [os.path.join(_repo_root, 'lmdeploy', 'turbomind')]
    _turbomind_pkg.__package__ = 'lmdeploy.turbomind'
    sys.modules['lmdeploy.turbomind'] = _turbomind_pkg

# (No longer need 'lmdeploy.turbomind.deploy' stub -- deploy/ was promoted.)


def _load_module_from_file(mod_name: str, file_path: str):
    """Load a Python module from *file_path* and register it as *mod_name*."""
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load linear.py first — weight_format.py imports from .linear at module level.
_linear_path = os.path.join(_repo_root, 'lmdeploy', 'turbomind', 'linear.py')
_linear_mod = _load_module_from_file('lmdeploy.turbomind.linear', _linear_path)
Linear = _linear_mod.Linear

# Load weight_format (needed by _base for TrivialFormat)
_wf_path = os.path.join(_repo_root, 'lmdeploy', 'turbomind', 'weight_format.py')
_load_module_from_file('lmdeploy.turbomind.weight_format', _wf_path)

# Load builder/_base.py
_base_path = os.path.join(_repo_root, 'lmdeploy', 'turbomind', 'builders', '_base.py')
_base_mod = _load_module_from_file('lmdeploy.turbomind.builders._base', _base_path)
transform_output_dim = _base_mod.transform_output_dim
transform_input_dim = _base_mod.transform_input_dim

# Register builder sub-package
_builder_pkg = sys.modules.get('lmdeploy.turbomind.builders')
if _builder_pkg is None:
    _builder_pkg = types.ModuleType('lmdeploy.turbomind.builders')
    _builder_pkg.__path__ = [os.path.join(_repo_root, 'lmdeploy', 'turbomind', 'builders')]
    _builder_pkg.__package__ = 'lmdeploy.turbomind.builders'
    sys.modules['lmdeploy.turbomind.builders'] = _builder_pkg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear(out_dim: int, in_dim: int | None = None,
                 has_bias: bool = False) -> Linear:
    """Create a trivial Linear for testing.

    If *in_dim* is given the weight is 2-D (in_dim, out_dim); otherwise it is 1-D (out_dim,) -- simulating a bias-only
    tensor.
    """
    tensors: dict[str, torch.Tensor] = {}
    if in_dim is not None:
        tensors['weight'] = torch.randn(in_dim, out_dim)
    else:
        tensors['weight'] = torch.randn(out_dim)
    if has_bias:
        tensors['bias'] = torch.randn(out_dim)
    return Linear(tensors=tensors,
                  weight_format='placeholder')


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTransformTensors:

    # -- 1-in / 1-out -------------------------------------------------------

    def test_1in_1out_2d_weight_only(self):
        """1-in/1-out with a 2-D weight tensor."""

        @transform_output_dim
        def double(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        lin = _make_linear(out_dim=8, in_dim=4)
        result = double(lin)
        assert isinstance(result, Linear)
        assert set(result.tensors) == {'weight'}
        assert result.tensors['weight'].shape == (4, 8)
        assert torch.allclose(result.tensors['weight'],
                              lin.tensors['weight'] * 2)

    def test_1in_1out_1d_bias_only(self):
        """1-in/1-out with a 1-D tensor (bias-only shape)."""

        @transform_output_dim
        def add_one(x: torch.Tensor) -> torch.Tensor:
            return x + 1.0

        lin = _make_linear(out_dim=6)  # 1-D weight
        result = add_one(lin)
        assert isinstance(result, Linear)
        assert result.tensors['weight'].shape == (6,)
        assert torch.allclose(result.tensors['weight'],
                              lin.tensors['weight'] + 1.0)

    def test_1in_1out_mixed_dims(self):
        """1-in/1-out with 2-D weight + 1-D bias."""

        @transform_output_dim
        def negate(x: torch.Tensor) -> torch.Tensor:
            return -x

        lin = _make_linear(out_dim=5, in_dim=3, has_bias=True)
        result = negate(lin)
        assert isinstance(result, Linear)
        assert set(result.tensors) == {'weight', 'bias'}
        # weight stays 2-D
        assert result.tensors['weight'].shape == (3, 5)
        assert torch.allclose(result.tensors['weight'],
                              -lin.tensors['weight'])
        # bias stays 1-D
        assert result.tensors['bias'].shape == (5,)
        assert torch.allclose(result.tensors['bias'],
                              -lin.tensors['bias'])

    # -- 1-in / 2-out (split) -----------------------------------------------

    def test_1in_2out_split(self):
        """1-in/2-out: split one Linear into two along last dim."""

        @transform_output_dim
        def split_in_half(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mid = x.shape[-1] // 2
            return x[..., :mid], x[..., mid:]

        lin = _make_linear(out_dim=8, in_dim=4, has_bias=True)
        a, b = split_in_half(lin)
        assert isinstance(a, Linear)
        assert isinstance(b, Linear)
        assert a.tensors['weight'].shape == (4, 4)
        assert b.tensors['weight'].shape == (4, 4)
        assert a.tensors['bias'].shape == (4,)
        assert b.tensors['bias'].shape == (4,)

    def test_1in_2out_1d_only(self):
        """1-in/2-out with 1-D tensors."""

        @transform_output_dim
        def split_1d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mid = x.shape[-1] // 2
            return x[..., :mid], x[..., mid:]

        lin = _make_linear(out_dim=6)  # 1-D
        a, b = split_1d(lin)
        assert a.tensors['weight'].shape == (3,)
        assert b.tensors['weight'].shape == (3,)

    # -- multi-in / 1-out (concat) ------------------------------------------

    def test_multi_in_1out_concat(self):
        """Multi-in/1-out: concatenate three Linears along last dim."""

        @transform_output_dim
        def concat3(a: torch.Tensor, b: torch.Tensor,
                    c: torch.Tensor) -> torch.Tensor:
            return torch.cat([a, b, c], dim=-1)

        la = _make_linear(out_dim=4, in_dim=3, has_bias=True)
        lb = _make_linear(out_dim=4, in_dim=3, has_bias=True)
        lc = _make_linear(out_dim=4, in_dim=3, has_bias=True)
        result = concat3(la, lb, lc)
        assert isinstance(result, Linear)
        assert result.tensors['weight'].shape == (3, 12)
        assert result.tensors['bias'].shape == (12,)

    # -- optional tensor arg -------------------------------------------------

    def test_optional_tensor_none(self):
        """Optional tensor arg passed as None -> inner fn receives None."""

        @transform_output_dim
        def maybe_add(x: torch.Tensor,
                      y: torch.Tensor | None) -> torch.Tensor:
            if y is None:
                return x
            return x + y

        lin = _make_linear(out_dim=4, in_dim=3)
        result = maybe_add(lin, None)
        assert isinstance(result, Linear)
        assert torch.allclose(result.tensors['weight'],
                              lin.tensors['weight'])

    def test_optional_tensor_provided(self):
        """Optional tensor arg provided -> inner fn receives the tensor."""

        @transform_output_dim
        def maybe_add(x: torch.Tensor,
                      y: torch.Tensor | None) -> torch.Tensor:
            return x + y

        la = _make_linear(out_dim=4, in_dim=3)
        lb = _make_linear(out_dim=4, in_dim=3)
        result = maybe_add(la, lb)
        assert isinstance(result, Linear)
        assert torch.allclose(result.tensors['weight'],
                              la.tensors['weight'] + lb.tensors['weight'])

    # -- format propagation --------------------------------------------------

    def test_format_propagation(self):
        """Output inherits weight_format from first input."""

        @transform_output_dim
        def identity(x: torch.Tensor) -> torch.Tensor:
            return x

        lin = _make_linear(out_dim=4, in_dim=3)
        object.__setattr__(lin, 'weight_format', 'fake_fmt')
        result = identity(lin)
        assert result.weight_format == 'fake_fmt'

    # -- kwargs passthrough ---------------------------------------------------

    def test_kwargs_passthrough(self):
        """Non-tensor kwargs are forwarded unchanged."""

        @transform_output_dim
        def scale(x: torch.Tensor, factor: float) -> torch.Tensor:
            return x * factor

        lin = _make_linear(out_dim=4, in_dim=3)
        result = scale(lin, factor=3.0)
        assert isinstance(result, Linear)
        assert torch.allclose(result.tensors['weight'],
                              lin.tensors['weight'] * 3.0)


class TestTransformInputDim:

    def test_2d_transformed(self):
        """2-D tensors are passed through the inner function."""

        @transform_input_dim
        def pad_first_dim(tensor: torch.Tensor,
                          *, target: int) -> torch.Tensor:
            return torch.nn.functional.pad(
                tensor, [0, 0, 0, target - tensor.size(0)])

        lin = _make_linear(out_dim=4, in_dim=2)
        result = pad_first_dim(lin, target=6)
        assert isinstance(result, Linear)
        assert result.tensors['weight'].shape == (6, 4)

    def test_1d_passthrough(self):
        """1-D tensors (bias) pass through unchanged."""

        @transform_input_dim
        def pad_first_dim(tensor: torch.Tensor,
                          *, target: int) -> torch.Tensor:
            return torch.nn.functional.pad(
                tensor, [0, 0, 0, target - tensor.size(0)])

        lin = _make_linear(out_dim=4)  # 1-D weight
        result = pad_first_dim(lin, target=6)
        assert isinstance(result, Linear)
        assert result.tensors['weight'].shape == (4,)  # unchanged

    def test_mixed_dims_2d_transformed_1d_passthrough(self):
        """2-D weight is transformed; 1-D bias passes through."""

        @transform_input_dim
        def double_input_dim(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.repeat(2, 1)

        lin = _make_linear(out_dim=4, in_dim=3, has_bias=True)
        result = double_input_dim(lin)
        assert isinstance(result, Linear)
        assert set(result.tensors) == {'weight', 'bias'}
        assert result.tensors['weight'].shape == (6, 4)  # doubled
        assert result.tensors['bias'].shape == (4,)  # unchanged

    def test_1in_2out_distributes_1d(self):
        """Multi-output: 1-D tensors duplicated into all output buckets."""

        @transform_input_dim
        def split_input(tensor: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
            mid = tensor.size(0) // 2
            return tensor[:mid], tensor[mid:]

        lin = _make_linear(out_dim=4, in_dim=6, has_bias=True)
        a, b = split_input(lin)
        assert isinstance(a, Linear)
        assert isinstance(b, Linear)
        assert a.tensors['weight'].shape == (3, 4)
        assert b.tensors['weight'].shape == (3, 4)
        assert a.tensors['bias'].shape == (4,)  # duplicated
        assert b.tensors['bias'].shape == (4,)  # duplicated
