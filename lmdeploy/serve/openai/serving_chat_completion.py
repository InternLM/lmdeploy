# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING

from .protocol import ChatCompletionRequest

if TYPE_CHECKING:
    from .api_server import VariableInterface


def check_request(request: ChatCompletionRequest, server_context: 'VariableInterface') -> str:
    engine_config = server_context.get_engine_config()
    session_manager = server_context.get_session_manager()
    try:
        # Check logprobs settings
        logprobs_mode = engine_config.logprobs_mode
        logprobs = request.logprobs
        top_logprobs = request.top_logprobs or 0
        if logprobs_mode is None and (logprobs or top_logprobs > 0):
            return (f'Logprobs({logprobs})/top_logprobs({top_logprobs}) requested '
                    'but not enabled logprobs_mode in engine configuration')
        if logprobs_mode is not None and (top_logprobs < 0 or (not logprobs and top_logprobs > 0)):
            return (f'Invalid logprobs({logprobs})/top_logprobs({top_logprobs}) requested '
                    'when logprobs_mode is enabled in engine configuration.')
    except AttributeError:
        pass

    if session_manager.has(request.session_id):
        return f'The session_id {request.session_id!r} is occupied.'

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
