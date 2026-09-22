# Copyright (c) OpenMMLab. All rights reserved.

from types import SimpleNamespace

import pytest

from lmdeploy.serve.openai.chat_completions.validation import check_request
from lmdeploy.serve.openai.protocol import ChatCompletionRequest


class _SessionManager:

    @staticmethod
    def has(session_id):
        return False


def _server_context():
    return SimpleNamespace(
        engine_config=SimpleNamespace(
            logprobs_mode='raw_logprobs',
            enable_return_routed_experts=False,
        ),
        session_manager=_SessionManager(),
        response_parser_cls=SimpleNamespace(tool_parser_cls=object()),
    )


def _request(**kwargs):
    request_kwargs = dict(
        model='fake-model',
        messages=[{'role': 'user', 'content': 'hi'}],
    )
    request_kwargs.update(kwargs)
    return ChatCompletionRequest(**request_kwargs)


@pytest.mark.parametrize(
    ('request_kwargs', 'error_fragment'),
    [
        pytest.param({'messages': []}, 'messages', id='empty-messages'),
        pytest.param({'messages': [], 'input_ids': []}, 'input_ids', id='empty-input-ids'),
        pytest.param({'min_p': -0.1}, 'min_p', id='min-p-below-range'),
        pytest.param({'min_p': 1.1}, 'min_p', id='min-p-above-range'),
        pytest.param({'max_completion_tokens': 0}, 'max_completion_tokens', id='zero-max-completion-tokens'),
        pytest.param({'max_tokens': 0}, 'max_tokens', id='zero-max-tokens'),
        pytest.param({'min_new_tokens': -1}, 'min_new_tokens', id='negative-min-new-tokens'),
        pytest.param({'logprobs': True, 'top_logprobs': 21}, 'top_logprobs', id='top-logprobs-above-range'),
    ],
)
def test_chat_completions_rejects_invalid_request(request_kwargs, error_fragment):
    error = check_request(_request(**request_kwargs), _server_context())

    assert error_fragment in error


def test_chat_completions_accepts_raw_input_ids_without_messages():
    error = check_request(
        _request(messages=[], input_ids=[1, 2, 3]),
        _server_context(),
    )

    assert error == ''


def test_chat_completions_accepts_null_n_as_unspecified():
    assert check_request(_request(n=None), _server_context()) == ''
