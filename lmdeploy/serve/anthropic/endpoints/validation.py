# Copyright (c) OpenMMLab. All rights reserved.
"""Request validation for the Anthropic ``/v1/messages`` endpoint."""

from __future__ import annotations

from http import HTTPStatus

from fastapi import Request

from ..adapter import get_model_list
from ..errors import create_error_response
from ..protocol import MessagesRequest


def check_request(request: MessagesRequest, raw_request: Request, server_context):
    """Validate Anthropic Messages request parameters."""

    header_error = _validate_headers(raw_request)
    if header_error is not None:
        return header_error

    model_error = _validate_model(request, server_context)
    if model_error is not None:
        return model_error

    sampling_error = _validate_sampling_request(request)
    if sampling_error is not None:
        return sampling_error

    extended_outputs_error = _validate_extended_outputs(request, server_context)
    if extended_outputs_error is not None:
        return extended_outputs_error

    input_error = _validate_input_request(request)
    if input_error is not None:
        return input_error

    return _validate_tool_choice_request(
        request,
        server_context.response_parser_cls,
    )


def messages_empty(request: MessagesRequest) -> bool:
    """Whether request uses the raw input_ids fallback path."""
    return request.messages is None or len(request.messages) == 0


def _validate_headers(raw_request: Request):
    anthropic_version = raw_request.headers.get('anthropic-version')
    if not anthropic_version:
        return create_error_response(HTTPStatus.BAD_REQUEST, 'Missing required header: anthropic-version')
    return None


def _validate_model(request: MessagesRequest, server_context):
    if request.model not in get_model_list(server_context):
        return create_error_response(
            HTTPStatus.NOT_FOUND,
            f'The model {request.model!r} does not exist.',
            error_type='not_found_error',
        )
    return None


def _validate_extended_outputs(request: MessagesRequest, server_context):
    # TurbomindEngineConfig has neither field; treat missing attrs as disabled.
    engine_config = server_context.engine_config
    logprobs_mode = getattr(engine_config, 'logprobs_mode', None)
    if request.return_logprob and logprobs_mode is None:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f'return_logprob={request.return_logprob} was requested, but '
            'logprobs_mode is not enabled in the engine configuration.')

    if request.return_routed_experts and not getattr(engine_config, 'enable_return_routed_experts', False):
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            ('routed experts requested but not configured in engine configuration. '
             'May start the api_server with --enable-return-routed-experts flag.'))

    return None


def _validate_sampling_request(request: MessagesRequest):
    if request.temperature is not None and not (0 <= request.temperature <= 1):
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f'temperature {request.temperature!r} must be in [0, 1].')
    if request.top_p is not None and not (0 <= request.top_p <= 1):
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f'top_p {request.top_p!r} must be in [0, 1].')
    if request.top_k is not None and request.top_k < 0:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f'top_k {request.top_k!r} cannot be a negative integer.')
    return None


def _validate_input_request(request: MessagesRequest):
    # messages has higher priority. input_ids and image_data are only used when
    # messages is empty. image_data requires input_ids.
    if not messages_empty(request):
        if request.input_ids is not None:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                'input_ids cannot be used when messages is non-empty.')
        if request.image_data is not None:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                'image_data cannot be used when messages is non-empty.')
        return None

    if request.input_ids is None:
        if request.image_data is not None:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                'image_data requires input_ids to be set when messages is empty.')
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            'messages must not be empty unless input_ids is set.')
    if len(request.input_ids) == 0:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            'input_ids must not be an empty list.')
    if request.system is not None:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            'system cannot be used when input_ids is set because raw input_ids bypass message rendering.')
    if request.tools:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            'tools cannot be used when input_ids is set because raw input_ids bypass message rendering.')
    if request.tool_choice is not None and not _is_tool_choice_auto(request.tool_choice):
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            'tool_choice cannot be used when input_ids is set because raw input_ids bypass message rendering.')
    return None


def _validate_tool_choice_request(request: MessagesRequest, parser_cls):
    if _is_tool_choice_any(request.tool_choice) and not request.tools:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            '`tool_choice={"type":"any"}` requires at least one tool.')

    if _is_tool_choice_tool(request.tool_choice):
        tool_name = request.tool_choice.name
        if not request.tools:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                '`tool_choice={"type":"tool"}` requires at least one tool.')
        tool_names = {tool.name for tool in request.tools}
        if tool_name not in tool_names:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                f"Tool choice 'tool' not found in `tools`: {tool_name!r}.")

    if request.tools and (parser_cls is None or parser_cls.tool_parser_cls is None):
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            'Please launch the api_server with --tool-call-parser if you want to use tool calling.')

    return None


def _is_tool_choice_auto(tool_choice) -> bool:
    if tool_choice is None:
        return True
    if isinstance(tool_choice, str):
        return tool_choice == 'auto'
    return tool_choice.type == 'auto'


def _is_tool_choice_any(tool_choice) -> bool:
    if tool_choice is None:
        return False
    if isinstance(tool_choice, str):
        return tool_choice == 'any'
    return tool_choice.type == 'any'


def _is_tool_choice_tool(tool_choice) -> bool:
    return tool_choice is not None and not isinstance(tool_choice, str) and tool_choice.type == 'tool'
