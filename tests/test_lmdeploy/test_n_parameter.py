# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for the `n` parameter in /v1/chat/completions and
/v1/completions.

These tests mock the async engine so they run without a GPU or real model.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, CompletionRequest
from lmdeploy.serve.openai.serving_chat_completion import check_request as chat_check_request
from lmdeploy.serve.openai.serving_completion import check_request as completion_check_request

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gen_out(response='hello', finish_reason=None, input_token_len=5, generate_token_len=3):
    out = MagicMock()
    out.response = response
    out.finish_reason = finish_reason
    out.token_ids = [1, 2, 3]
    out.logprobs = None
    out.input_token_len = input_token_len
    out.generate_token_len = generate_token_len
    out.cache_block_ids = None
    return out


async def _async_gen(*items):
    """Async generator yielding items."""
    for item in items:
        yield item


def _make_session(session_id=42):
    sess = MagicMock()
    sess.session_id = session_id
    sess.async_abort = AsyncMock()
    return sess


# ---------------------------------------------------------------------------
# Protocol-level validation
# ---------------------------------------------------------------------------


class TestNParameterValidation:

    def _make_server_context(self, session_occupied=False):
        ctx = MagicMock()
        ctx.get_engine_config.return_value = MagicMock(spec=[])  # no logprobs_mode attr
        mgr = MagicMock()
        mgr.has.return_value = session_occupied
        ctx.get_session_manager.return_value = mgr
        return ctx

    def test_chat_n_default_is_1(self):
        req = ChatCompletionRequest(model='m', messages='hi')
        assert req.n == 1

    def test_completion_n_default_is_1(self):
        req = CompletionRequest(model='m', prompt='hi')
        assert req.n == 1

    def test_chat_n_valid_values(self):
        ctx = self._make_server_context()
        for n in [1, 2, 5]:
            req = ChatCompletionRequest(model='m', messages='hi', n=n)
            assert chat_check_request(req, ctx) == ''

    def test_completion_n_valid_values(self):
        ctx = self._make_server_context()
        for n in [1, 2, 5]:
            req = CompletionRequest(model='m', prompt='hi', n=n)
            assert completion_check_request(req, ctx) == ''

    def test_chat_n_zero_rejected(self):
        ctx = self._make_server_context()
        req = ChatCompletionRequest(model='m', messages='hi', n=0)
        assert chat_check_request(req, ctx) != ''

    def test_completion_n_zero_rejected(self):
        ctx = self._make_server_context()
        req = CompletionRequest(model='m', prompt='hi', n=0)
        assert completion_check_request(req, ctx) != ''

    def test_chat_n_negative_rejected(self):
        ctx = self._make_server_context()
        req = ChatCompletionRequest(model='m', messages='hi', n=-1)
        assert chat_check_request(req, ctx) != ''

    def test_completion_n_negative_rejected(self):
        ctx = self._make_server_context()
        req = CompletionRequest(model='m', prompt='hi', n=-1)
        assert completion_check_request(req, ctx) != ''


# ---------------------------------------------------------------------------
# API handler tests (mocking VariableInterface and raw_request)
# ---------------------------------------------------------------------------


def _make_raw_request(disconnected=False):
    raw = MagicMock()
    raw.json = AsyncMock(return_value={})
    raw.is_disconnected = AsyncMock(return_value=disconnected)
    return raw


def _setup_variable_interface(mock_vi, n_sessions=1, gen_outputs=None):
    """Configure the mocked VariableInterface.

    gen_outputs: list of lists – one list of GenOut per generator call.
    """
    engine = MagicMock()
    engine.model_name = 'test-model'
    engine.arch = 'LlamaForCausalLM'
    engine.tokenizer = MagicMock()

    sessions = [_make_session(i + 10) for i in range(max(n_sessions, 1))]
    _session_iter = iter(sessions)

    def _get_session(sid):
        try:
            return next(_session_iter)
        except StopIteration:
            return _make_session(99)

    mock_vi.get_session.side_effect = _get_session
    mock_vi.async_engine = engine
    mock_vi.tool_parser = None
    mock_vi.reasoning_parser = None
    mock_vi.allow_terminate_by_client = False
    mock_vi.enable_abort_handling = False

    # Each call to engine.generate returns a different async generator
    gen_outputs = gen_outputs or [[_make_gen_out('hello', 'stop')]]

    _gen_iter = iter(gen_outputs)

    def _generate(*args, **kwargs):
        items = next(_gen_iter, [_make_gen_out('hello', 'stop')])
        return _async_gen(*items)

    engine.generate.side_effect = _generate
    return sessions


class TestChatCompletionsN:

    @pytest.mark.asyncio
    async def test_n1_returns_one_choice(self):
        from lmdeploy.serve.openai import api_server

        request = ChatCompletionRequest(model='test-model', messages=[{'role': 'user', 'content': 'hi'}], n=1)
        raw_request = _make_raw_request()

        with patch.object(api_server, 'VariableInterface') as mock_vi, \
             patch.object(api_server, 'check_request', return_value=None):
            _setup_variable_interface(mock_vi, n_sessions=1, gen_outputs=[[_make_gen_out('ans1', 'stop')]])
            response = await api_server.chat_completions_v1(request, raw_request)

        assert isinstance(response, dict)
        assert len(response['choices']) == 1
        assert response['choices'][0]['index'] == 0
        assert response['choices'][0]['message']['content'] == 'ans1'

    @pytest.mark.asyncio
    async def test_n3_returns_three_choices(self):
        from lmdeploy.serve.openai import api_server

        request = ChatCompletionRequest(model='test-model', messages=[{'role': 'user', 'content': 'hi'}], n=3)
        raw_request = _make_raw_request()

        outputs = [
            [_make_gen_out('ans0', 'stop')],
            [_make_gen_out('ans1', 'stop')],
            [_make_gen_out('ans2', 'stop')],
        ]

        with patch.object(api_server, 'VariableInterface') as mock_vi, \
             patch.object(api_server, 'check_request', return_value=None):
            _setup_variable_interface(mock_vi, n_sessions=3, gen_outputs=outputs)
            response = await api_server.chat_completions_v1(request, raw_request)

        assert isinstance(response, dict)
        choices = response['choices']
        assert len(choices) == 3
        assert [c['index'] for c in choices] == [0, 1, 2]
        assert choices[0]['message']['content'] == 'ans0'
        assert choices[1]['message']['content'] == 'ans1'
        assert choices[2]['message']['content'] == 'ans2'

    @pytest.mark.asyncio
    async def test_n3_usage_aggregates_completion_tokens(self):
        from lmdeploy.serve.openai import api_server

        request = ChatCompletionRequest(model='test-model', messages=[{'role': 'user', 'content': 'hi'}], n=3)
        raw_request = _make_raw_request()

        # Each generator produces 10 completion tokens
        outputs = [[_make_gen_out('a', 'stop', input_token_len=5, generate_token_len=10)]] * 3

        with patch.object(api_server, 'VariableInterface') as mock_vi, \
             patch.object(api_server, 'check_request', return_value=None):
            _setup_variable_interface(mock_vi, n_sessions=3, gen_outputs=outputs)
            response = await api_server.chat_completions_v1(request, raw_request)

        usage = response['usage']
        assert usage['prompt_tokens'] == 5  # counted once (shared input)
        assert usage['completion_tokens'] == 30  # 3 * 10
        assert usage['total_tokens'] == 35

    @pytest.mark.asyncio
    async def test_n1_uses_request_session_id(self):
        from lmdeploy.serve.openai import api_server

        request = ChatCompletionRequest(model='test-model',
                                        messages=[{
                                            'role': 'user',
                                            'content': 'hi'
                                        }],
                                        n=1,
                                        session_id=77)
        raw_request = _make_raw_request()

        with patch.object(api_server, 'VariableInterface') as mock_vi, \
             patch.object(api_server, 'check_request', return_value=None):
            _setup_variable_interface(mock_vi, n_sessions=1)
            await api_server.chat_completions_v1(request, raw_request)

        mock_vi.get_session.assert_called_once_with(77)

    @pytest.mark.asyncio
    async def test_n3_uses_auto_sessions(self):
        from lmdeploy.serve.openai import api_server

        request = ChatCompletionRequest(model='test-model', messages=[{'role': 'user', 'content': 'hi'}], n=3)
        raw_request = _make_raw_request()

        with patch.object(api_server, 'VariableInterface') as mock_vi, \
             patch.object(api_server, 'check_request', return_value=None):
            _setup_variable_interface(mock_vi, n_sessions=3, gen_outputs=[[_make_gen_out('x', 'stop')]] * 3)
            await api_server.chat_completions_v1(request, raw_request)

        # All 3 sessions should be auto-assigned (-1)
        calls = mock_vi.get_session.call_args_list
        assert len(calls) == 3
        assert all(c.args[0] == -1 for c in calls)

    @pytest.mark.asyncio
    async def test_n3_seeds_are_offset(self):
        """When seed is set and n>1, generators should use seed, seed+1,
        seed+2."""
        from lmdeploy.serve.openai import api_server

        request = ChatCompletionRequest(model='test-model', messages=[{'role': 'user', 'content': 'hi'}], n=3, seed=100)
        raw_request = _make_raw_request()

        captured_configs = []

        def _generate(*args, **kwargs):
            captured_configs.append(kwargs.get('gen_config'))
            return _async_gen(_make_gen_out('x', 'stop'))

        with patch.object(api_server, 'VariableInterface') as mock_vi, \
             patch.object(api_server, 'check_request', return_value=None):
            _setup_variable_interface(mock_vi, n_sessions=3)
            mock_vi.async_engine.generate.side_effect = _generate
            await api_server.chat_completions_v1(request, raw_request)

        seeds = [cfg.random_seed for cfg in captured_configs]
        assert seeds == [100, 101, 102]


class TestCompletionsN:

    @pytest.mark.asyncio
    async def test_n1_single_prompt_one_choice(self):
        from lmdeploy.serve.openai import api_server

        request = CompletionRequest(model='test-model', prompt='hi', n=1)
        raw_request = _make_raw_request()

        with patch.object(api_server, 'VariableInterface') as mock_vi, \
             patch.object(api_server, 'check_request', return_value=None):
            _setup_variable_interface(mock_vi, n_sessions=1, gen_outputs=[[_make_gen_out('out0', 'stop')]])
            response = await api_server.completions_v1(request, raw_request)

        assert len(response['choices']) == 1
        assert response['choices'][0]['index'] == 0

    @pytest.mark.asyncio
    async def test_n3_single_prompt_three_choices(self):
        from lmdeploy.serve.openai import api_server

        request = CompletionRequest(model='test-model', prompt='hi', n=3)
        raw_request = _make_raw_request()

        outputs = [
            [_make_gen_out('out0', 'stop')],
            [_make_gen_out('out1', 'stop')],
            [_make_gen_out('out2', 'stop')],
        ]

        with patch.object(api_server, 'VariableInterface') as mock_vi, \
             patch.object(api_server, 'check_request', return_value=None):
            _setup_variable_interface(mock_vi, n_sessions=3, gen_outputs=outputs)
            response = await api_server.completions_v1(request, raw_request)

        choices = response['choices']
        assert len(choices) == 3
        assert [c['index'] for c in choices] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_n2_two_prompts_four_choices(self):
        """2 prompts × n=2 = 4 choices, indexed 0..3."""
        from lmdeploy.serve.openai import api_server

        request = CompletionRequest(model='test-model', prompt=['p0', 'p1'], n=2)
        raw_request = _make_raw_request()

        outputs = [[_make_gen_out(f'out{i}', 'stop')] for i in range(4)]

        with patch.object(api_server, 'VariableInterface') as mock_vi, \
             patch.object(api_server, 'check_request', return_value=None):
            _setup_variable_interface(mock_vi, n_sessions=4, gen_outputs=outputs)
            response = await api_server.completions_v1(request, raw_request)

        choices = response['choices']
        assert len(choices) == 4
        assert [c['index'] for c in choices] == [0, 1, 2, 3]
