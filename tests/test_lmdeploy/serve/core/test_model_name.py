# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

from lmdeploy.serve.core.model_name import (
    model_matches,
    model_or_adapter_matches,
    name_matches_any,
    split_model_name,
)
from lmdeploy.serve.openai.endpoints.common import validate_request


class TestSplitModelName:
    """Unit tests for served-model name parsing."""

    def test_single_name(self):
        primary, aliases = split_model_name('Qwen3-32B')
        assert primary == 'Qwen3-32B'
        assert aliases == ['Qwen3-32B']

    def test_comma_separated_aliases(self):
        primary, aliases = split_model_name('Qwen3-32B,qwen3-32b,default')
        assert primary == 'Qwen3-32B'
        assert aliases == ['Qwen3-32B', 'qwen3-32b', 'default']

    def test_strips_whitespace_and_empty_entries(self):
        primary, aliases = split_model_name('  model-a , model-b ,,  ')
        assert primary == 'model-a'
        assert aliases == ['model-a', 'model-b']


class TestNameMatching:
    """Unit tests for case-insensitive model-name matching."""

    def test_case_insensitive_match(self):
        assert name_matches_any('qwen3-32b', ['Qwen3-32B'])
        assert name_matches_any('QWEN3-32B', ['Qwen3-32B'])

    def test_alias_match(self):
        assert name_matches_any('default', ['Qwen3-32B', 'default'])

    def test_no_match(self):
        assert not name_matches_any('other-model', ['Qwen3-32B', 'default'])

    def test_none_requested_name(self):
        assert not name_matches_any(None, ['Qwen3-32B'])

    def test_engine_without_alias_attribute(self):
        engine = SimpleNamespace(model_name='model-a')
        assert model_matches('MODEL-A', engine)
        assert not model_matches('model-b', engine)

    def test_engine_with_aliases(self):
        engine = SimpleNamespace(
            model_name='Qwen3-32B',
            model_aliases=['Qwen3-32B', 'qwen3-32b', 'default'],
        )
        assert model_matches('qwen3-32b', engine)
        assert model_matches('DEFAULT', engine)
        assert not model_matches('other', engine)

    def test_model_or_adapter_exact_adapter_match(self):
        engine = SimpleNamespace(
            model_name='fake-model',
            model_aliases=['fake-model', 'default'],
        )
        assert model_or_adapter_matches('default', engine, ['MyAdapter'])
        assert model_or_adapter_matches('MyAdapter', engine, ['MyAdapter'])
        # adapter matching stays exact: casing variants are not accepted
        assert not model_or_adapter_matches('myadapter', engine, ['MyAdapter'])
        assert not model_or_adapter_matches('other', engine, ['MyAdapter'])


class TestValidateRequestModelMatching:
    """Request validation accepts case variants and aliases of served names."""

    @staticmethod
    def _make_context(engine, adapters=None):
        context = SimpleNamespace(
            async_engine=engine,
            engine_config=SimpleNamespace(adapters=adapters or []),
        )
        return context

    @staticmethod
    def _no_error(request, server_context, **kwargs):
        return None

    def test_accepts_case_variant(self):
        engine = SimpleNamespace(
            model_name='fake-model',
            model_aliases=['fake-model', 'default'],
        )
        context = self._make_context(engine)
        request = SimpleNamespace(model='FAKE-MODEL')
        assert validate_request(request, context, self._no_error) is None

    def test_accepts_alias(self):
        engine = SimpleNamespace(
            model_name='fake-model',
            model_aliases=['fake-model', 'default'],
        )
        context = self._make_context(engine)
        request = SimpleNamespace(model='default')
        assert validate_request(request, context, self._no_error) is None

    def test_rejects_unknown_model(self):
        engine = SimpleNamespace(
            model_name='fake-model',
            model_aliases=['fake-model', 'default'],
        )
        context = self._make_context(engine)
        request = SimpleNamespace(model='other-model')
        response = validate_request(request, context, self._no_error)
        assert response is not None
        assert response.status_code == 404

    def test_adapter_matching_stays_exact(self):
        engine = SimpleNamespace(
            model_name='fake-model',
            model_aliases=['fake-model', 'default'],
        )
        context = self._make_context(engine, adapters=['MyAdapter'])
        accepted = validate_request(SimpleNamespace(model='MyAdapter'), context, self._no_error)
        assert accepted is None
        rejected = validate_request(SimpleNamespace(model='myadapter'), context, self._no_error)
        assert rejected is not None
        assert rejected.status_code == 404
