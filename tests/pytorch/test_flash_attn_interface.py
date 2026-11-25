# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch


def _load_adapter(monkeypatch, varlen):
    """Load the real adapter against a strict FA3 Python signature."""
    fa3 = ModuleType('flash_attn_interface')
    fa3.flash_attn_varlen_func = varlen
    fa3.flash_attn_with_kvcache = Mock()
    monkeypatch.setitem(sys.modules, 'flash_attn_interface', fa3)
    path = Path(__file__).parents[2] / 'lmdeploy/pytorch/third_party/flash_attn_interface.py'
    spec = importlib.util.spec_from_file_location('_test_fa3_adapter', path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('return_lse', [False, True])
def test_varlen_interface_and_lse(monkeypatch, legacy, return_lse):
    output = torch.randn(3, 2, 8)
    lse = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    calls = []

    def old_varlen(q, k, v, *, cu_seqlens_q, cu_seqlens_k):
        calls.append(None)
        return output, lse

    def new_varlen(q, k, v, *, cu_seqlens_q, cu_seqlens_k, return_attn_probs=False):
        calls.append(return_attn_probs)
        return (output, lse) if return_attn_probs else output

    adapter = _load_adapter(monkeypatch, old_varlen if legacy else new_varlen)
    dcp = ModuleType('lmdeploy.pytorch.kernels.cuda.dcp')
    dcp.sanitize_dcp_lse = Mock(side_effect=lambda lse, counts: lse.masked_fill(counts[:, None] == 0, -torch.inf))
    monkeypatch.setitem(sys.modules, dcp.__name__, dcp)
    result = adapter.flash_attn_varlen_func(
        output, output, output, cu_seqlens_q=torch.tensor([0, 1, 3], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 0, 4], dtype=torch.int32), return_lse=return_lse)
    assert calls == [None if legacy else return_lse]
    if return_lse:
        assert result[0] is output
        expected = lse.T.clone()
        expected[0] = -torch.inf
        torch.testing.assert_close(result[1], expected)
        dcp.sanitize_dcp_lse.assert_called_once()
        torch.testing.assert_close(dcp.sanitize_dcp_lse.call_args.args[1], torch.tensor([0, 4, 4], dtype=torch.int32))
    else:
        assert result is output
        dcp.sanitize_dcp_lse.assert_not_called()


def test_varlen_missing_lse_is_explicit(monkeypatch):
    def varlen(q, k, v):
        # A tensor with leading dimension two must not be unpacked as (out, LSE).
        return torch.zeros(2, 2, 8)

    adapter = _load_adapter(monkeypatch, varlen)
    with pytest.raises(RuntimeError, match='FA3 must return'):
        adapter.flash_attn_varlen_func(None, None, None, return_lse=True)


def test_varlen_does_not_retry_internal_type_error(monkeypatch):
    calls = []

    def varlen(q, k, v, return_attn_probs=False):
        calls.append(return_attn_probs)
        raise TypeError('internal FA3 failure')

    adapter = _load_adapter(monkeypatch, varlen)
    with pytest.raises(TypeError, match='internal FA3 failure'):
        adapter.flash_attn_varlen_func(None, None, None, return_lse=True)
    assert calls == [True]
