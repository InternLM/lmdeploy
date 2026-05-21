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

    # Validate input_ids and image_data constraints.
    # messages has higher priority. input_ids and image_data are only used when
    # messages is empty (None, '', or []). image_data requires input_ids.
    messages_empty = (request.messages is None
                      or request.messages == ''
                      or (isinstance(request.messages, list) and len(request.messages) == 0))
    if not messages_empty:
        # messages is active — input_ids and image_data must not be set
        if request.input_ids is not None:
            return 'input_ids cannot be used when messages is non-empty. messages takes priority.'
        if request.image_data is not None:
            return 'image_data cannot be used when messages is non-empty. messages takes priority.'
    else:
        # messages is empty — input_ids and image_data are the active inputs
        if request.input_ids is not None and len(request.input_ids) == 0:
            return 'The input_ids must not be an empty list.'
        if request.image_data is not None and request.input_ids is None:
            return 'image_data requires input_ids to be set when messages is empty.'

    if request.return_routed_experts and not engine_config.enable_return_routed_experts:
        return ('routed experts requested but not configured in engine configuration. '
    'May start api_server with --enable-return-routed-experts flag.')

    return ''
