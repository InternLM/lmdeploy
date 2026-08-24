# Copyright (c) OpenMMLab. All rights reserved.
"""Exceptions for the serve module."""

import enum
from http import HTTPStatus


class ErrorCode(str, enum.Enum):
    """Stable, transport-independent serving error codes."""

    INVALID_REQUEST = 'invalid_request'
    UNAUTHORIZED = 'unauthorized'
    MODEL_NOT_FOUND = 'model_not_found'
    REQUEST_CONFLICT = 'request_conflict'
    CONTEXT_LENGTH_EXCEEDED = 'context_length_exceeded'
    UNSUPPORTED_FEATURE = 'unsupported_feature'
    PREPROCESS_FAILED = 'preprocess_failed'
    REQUEST_CANCELLED = 'request_cancelled'
    ENGINE_UNAVAILABLE = 'engine_unavailable'
    INTERNAL_ERROR = 'internal_error'


ERROR_STATUS = {
    ErrorCode.INVALID_REQUEST: HTTPStatus.BAD_REQUEST,
    ErrorCode.UNAUTHORIZED: HTTPStatus.UNAUTHORIZED,
    ErrorCode.MODEL_NOT_FOUND: HTTPStatus.NOT_FOUND,
    ErrorCode.REQUEST_CONFLICT: HTTPStatus.CONFLICT,
    ErrorCode.CONTEXT_LENGTH_EXCEEDED: HTTPStatus.BAD_REQUEST,
    ErrorCode.UNSUPPORTED_FEATURE: HTTPStatus.BAD_REQUEST,
    ErrorCode.PREPROCESS_FAILED: HTTPStatus.BAD_REQUEST,
    ErrorCode.REQUEST_CANCELLED: 499,
    ErrorCode.ENGINE_UNAVAILABLE: HTTPStatus.SERVICE_UNAVAILABLE,
    ErrorCode.INTERNAL_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,
}

ERROR_MESSAGES = {
    ErrorCode.INVALID_REQUEST: 'The request is invalid.',
    ErrorCode.UNAUTHORIZED: 'Unauthorized.',
    ErrorCode.MODEL_NOT_FOUND: 'The requested model does not exist.',
    ErrorCode.REQUEST_CONFLICT: 'The request conflicts with existing server state.',
    ErrorCode.CONTEXT_LENGTH_EXCEEDED: 'The request exceeds the model context length.',
    ErrorCode.UNSUPPORTED_FEATURE: 'The requested feature is not supported.',
    ErrorCode.PREPROCESS_FAILED: 'Request preprocessing failed.',
    ErrorCode.REQUEST_CANCELLED: 'The request was cancelled.',
    ErrorCode.ENGINE_UNAVAILABLE: 'The inference engine is unavailable.',
    ErrorCode.INTERNAL_ERROR: 'An internal server error occurred.',
}


class RequestError(RuntimeError):
    """A request failure that can be rendered by a serving protocol."""

    def __init__(self, code: ErrorCode, message: str | None = None):
        self.code = code
        self.message = message or ERROR_MESSAGES[code]
        self.status_code = ERROR_STATUS[code]
        super().__init__(self.message)


class SafeRunException(Exception):
    """Exception raised by safe_run to avoid upper layer handling the original
    exception again.

    This exception wraps the original exception that occurred during safe_run execution.
    """
