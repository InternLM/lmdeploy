# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING

from .protocol import GenerateReqInput

if TYPE_CHECKING:
    from .api_server import VariableInterface


def check_request(request: GenerateReqInput, server_context: 'VariableInterface') -> str:
    engine_config = server_context.get_engine_config()
    session_manager = server_context.get_session_manager()
    try:
        # Check logprobs settings
        logprobs_mode = engine_config.logprobs_mode
        return_logprob = request.return_logprob
        if logprobs_mode is None and return_logprob:
            return f'return_logprob({return_logprob}) requested but not enabled logprobs_mode in engine configuration.'
    except AttributeError:
        pass

    if (request.prompt is not None) ^ (request.input_ids is None):
        return 'You must specify exactly one of prompt or input_ids'

    if request.prompt is not None and request.prompt == '':
        return 'The prompt must not be an empty string'

    if request.input_ids is not None and len(request.input_ids) == 0:
        return 'The input_ids must not be an empty list'

    if request.max_tokens is not None and request.max_tokens <= 0:
        return f'The max_tokens {request.max_tokens!r} must be a positive integer.'

    if session_manager.has(request.session_id):
        return f'The session_id {request.session_id!r} is occupied.'

    # check sampling settings
    if not (0 < request.top_p <= 1):
        return f'The top_p {request.top_p!r} must be in (0, 1].'
    if request.top_k < 0:
        return f'The top_k {request.top_k!r} cannot be a negative integer.'
    if not (0 <= request.temperature <= 2):
        return f'The temperature {request.temperature!r} must be in [0, 2]'

    return ''
