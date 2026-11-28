# Copyright (c) OpenMMLab. All rights reserved.
"""Request validation for the ``/v1/completions`` endpoint."""
from types import SimpleNamespace

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.serve.openai.endpoints.completions import check_request
from lmdeploy.serve.openai.protocol import CompletionRequest


class _SessionManager:

    @staticmethod
    def has(session_id):
        return False


def _server_context():
    return SimpleNamespace(
        engine_config=PytorchEngineConfig(),
        session_manager=_SessionManager(),
    )


def _request(**kwargs):
    request_kwargs = dict(
        model='fake-model',
        prompt='hi',
    )
    request_kwargs.update(kwargs)
    return CompletionRequest(**request_kwargs)


def test_valid_sampling_bounds_pass():
    assert check_request(_request(min_p=None), _server_context()) == ''
    assert check_request(_request(min_p=0.05), _server_context()) == ''
    assert check_request(_request(min_p=1.0), _server_context()) == ''


def test_out_of_range_min_p_fails_like_chat():
    for value in (-0.1, 1.1):
        message = check_request(_request(min_p=value), _server_context())
        assert 'min_p' in message
        assert str(value) in message
