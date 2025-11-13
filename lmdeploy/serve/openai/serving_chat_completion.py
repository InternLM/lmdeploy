# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING

from .protocol import ChatCompletionRequest

if TYPE_CHECKING:
    from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig


def check_request(request: ChatCompletionRequest, engine_config: 'TurbomindEngineConfig | PytorchEngineConfig') -> str:
    if not isinstance(request, ChatCompletionRequest):
        raise TypeError(f'Invalid request type, expected ChatCompletionRequest, got {type(request)}')

    # Check logprobs settings
    try:
        logprobs_mode = engine_config.logprobs_mode
        logprobs = request.logprobs
        top_logprobs = request.top_logprobs or 0
        if logprobs_mode is None and (logprobs or top_logprobs > 0):
            return (f'Logprobs({logprobs})/top_logprobs({top_logprobs}) requested '
                    'but not enabled logprobs_mode in engine configuration')
        if logprobs_mode is not None and (logprobs or top_logprobs < 0):
            return (f'top_logprobs({top_logprobs}) must be non-negative '
                    'when logprobs_mode is enabled in engine configuration.')
    except AttributeError:
        pass

    # check sampling settings
    if request.n <= 0:
        return f'The n {request.n!r} must be a positive int.'
    if not (0 < request.top_p <= 1):
        return f'The top_p {request.top_p!r} must be in (0, 1].'
    if request.top_k < 0:
        return f'The top_k {request.top_k!r} cannot be a negative integer.'
    if not (0 <= request.temperature <= 2):
        return f'The temperature {request.temperature!r} must be in [0, 2]'

    return ''
