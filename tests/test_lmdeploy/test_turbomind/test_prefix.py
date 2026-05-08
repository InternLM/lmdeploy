# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for Prefix path arithmetic and tensor access."""
from __future__ import annotations

import pytest
import torch

from lmdeploy.turbomind.checkpoint import Prefix


class _FakeCheckpoint:
    """In-memory Checkpoint stand-in for unit tests."""

    def __init__(self, data: dict[str, torch.Tensor]):
        self._data = data

    def get(self, key: str, index=None) -> torch.Tensor:
        t = self._data[key]
        if index is not None:
            t = t[index]
        return t

    def pop(self, key: str, index=None) -> torch.Tensor:
        t = self._data.pop(key)
        if index is not None:
            t = t[index]
        return t

    def has(self, key: str) -> bool:
        return key in self._data



# ---------------------------------------------------------------------------
# Path arithmetic
# ---------------------------------------------------------------------------


class TestPrefixArithmetic:

    def test_default_prefix_is_empty(self):
        ckpt = _FakeCheckpoint({})
        p = Prefix(ckpt)
        assert p.prefix == ''
        assert p.ckpt is ckpt

    def test_add_string_inserts_dot(self):
        p = Prefix(_FakeCheckpoint({})) + 'model'
        assert p.prefix == 'model'

    def test_add_chained(self):
        p = Prefix(_FakeCheckpoint({})) + 'model' + 'layers' + 'self_attn'
        assert p.prefix == 'model.layers.self_attn'

    def test_add_int_inserts_dot(self):
        p = Prefix(_FakeCheckpoint({})) + 'model' + 'layers' + 0
        assert p.prefix == 'model.layers.0'

    def test_add_to_non_empty_inserts_dot(self):
        ckpt = _FakeCheckpoint({})
        p = Prefix(ckpt, 'model.layers') + 5
        assert p.prefix == 'model.layers.5'

    def test_append_default_separator_dot(self):
        ckpt = _FakeCheckpoint({})
        p = Prefix(ckpt, 'model').append('embed_tokens')
        assert p.prefix == 'model.embed_tokens'

    def test_append_with_empty_separator(self):
        ckpt = _FakeCheckpoint({})
        p = Prefix(ckpt, 'gate_up_proj').append('_bias', sep='')
        assert p.prefix == 'gate_up_proj_bias'

    def test_repr_shows_prefix_string(self):
        p = Prefix(_FakeCheckpoint({}), 'model.layers.0')
        assert repr(p) == "Prefix('model.layers.0')"


# ---------------------------------------------------------------------------
# Tensor access
# ---------------------------------------------------------------------------


class TestPrefixGetHas:

    def test_get_with_default_sep_inserts_dot(self):
        t = torch.zeros(2)
        ckpt = _FakeCheckpoint({'model.layers.0.input_layernorm.weight': t})
        p = Prefix(ckpt, 'model.layers.0')
        assert p.get('input_layernorm.weight') is t

    def test_get_no_name_reads_exact_prefix(self):
        t = torch.zeros(2)
        ckpt = _FakeCheckpoint({'model.norm.weight': t})
        p = Prefix(ckpt, 'model.norm.weight')
        assert p.get() is t

    def test_get_with_empty_sep_does_raw_concat(self):
        t = torch.zeros(2)
        ckpt = _FakeCheckpoint({'experts.gate_up_proj.qweight': t})
        p = Prefix(ckpt, 'experts.gate_up_proj')
        assert p.get('.qweight', sep='') is t

    def test_get_with_empty_sep_for_underscore_suffix(self):
        t = torch.zeros(2)
        ckpt = _FakeCheckpoint({'experts.gate_up_proj_bias': t})
        p = Prefix(ckpt, 'experts.gate_up_proj')
        assert p.get('_bias', sep='') is t

    def test_get_missing_key_raises_key_error(self):
        ckpt = _FakeCheckpoint({})
        p = Prefix(ckpt, 'model')
        with pytest.raises(KeyError):
            p.get('embed_tokens.weight')

    def test_has_present(self):
        ckpt = _FakeCheckpoint({'model.embed_tokens.weight': torch.zeros(2)})
        p = Prefix(ckpt, 'model.embed_tokens')
        assert p.has('weight') is True

    def test_has_missing(self):
        p = Prefix(_FakeCheckpoint({}), 'model.embed_tokens')
        assert p.has('weight') is False

    def test_has_with_empty_sep(self):
        ckpt = _FakeCheckpoint({'experts.gate_up_proj_blocks': torch.zeros(2)})
        p = Prefix(ckpt, 'experts.gate_up_proj')
        assert p.has('_blocks', sep='') is True
        assert p.has('_blocks') is False


# ---------------------------------------------------------------------------
# slices() — deterministic index iteration
# ---------------------------------------------------------------------------


class TestPrefixSlices:

    def test_yields_index_prefix_pairs(self):
        ckpt = _FakeCheckpoint({})
        p = Prefix(ckpt, 'model.layers')
        out = list(p.slices(0, 3))
        assert [(i, pf.prefix) for i, pf in out] == [
            (0, 'model.layers.0'),
            (1, 'model.layers.1'),
            (2, 'model.layers.2'),
        ]

    def test_empty_when_begin_equals_end(self):
        ckpt = _FakeCheckpoint({})
        p = Prefix(ckpt, 'model.layers')
        assert list(p.slices(3, 3)) == []

    def test_empty_when_begin_exceeds_end(self):
        ckpt = _FakeCheckpoint({})
        p = Prefix(ckpt, 'model.layers')
        assert list(p.slices(5, 3)) == []

    def test_respects_half_open_interval(self):
        """slices(2, 4) yields indices 2, 3 — drafter layer 4 is excluded."""
        ckpt = _FakeCheckpoint({})
        p = Prefix(ckpt, 'model.layers')
        out = list(p.slices(2, 4))
        assert [(i, pf.prefix) for i, pf in out] == [
            (2, 'model.layers.2'),
            (3, 'model.layers.3'),
        ]

    def test_no_checkpoint_key_scan(self):
        """slices never reads checkpoint keys — it's purely deterministic."""

        class _NoKeyCheckpoint:
            def keys(self):
                raise AssertionError('slices must not call keys()')
            def get(self, key, index=None):
                raise AssertionError('slices must not call get()')
            def pop(self, key, index=None):
                raise AssertionError('slices must not call pop()')
            def has(self, key):
                raise AssertionError('slices must not call has()')

        p = Prefix(_NoKeyCheckpoint(), 'model.layers')
        out = list(p.slices(0, 2))
        assert [(i, pf.prefix) for i, pf in out] == [
            (0, 'model.layers.0'),
            (1, 'model.layers.1'),
        ]

    def test_yields_tqdm_iterator(self):
        """slices wraps range in tqdm for progress display."""
        ckpt = _FakeCheckpoint({})
        p = Prefix(ckpt, 'model.layers')
        it = p.slices(0, 2)
        # slices is a generator; tqdm wraps the range internally
        out = list(it)
        assert [(i, pf.prefix) for i, pf in out] == [
            (0, 'model.layers.0'),
            (1, 'model.layers.1'),
        ]
        # Verify tqdm is used in the function source
        import inspect
        assert 'tqdm' in inspect.getsource(Prefix.slices)
