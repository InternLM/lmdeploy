# Copyright (c) OpenMMLab. All rights reserved.
"""Shared Anthropic request header validation."""

from datetime import date
from http import HTTPStatus

from fastapi import Request

from .errors import create_error_response


def validate_anthropic_headers(raw_request: Request):
    version = raw_request.headers.get('anthropic-version')
    if not version:
        return create_error_response(HTTPStatus.BAD_REQUEST, 'Missing required header: anthropic-version')
    try:
        # Check syntax and calendar validity without a stale version allowlist.
        # This does not negotiate version-specific API semantics.
        if len(version) != 10 or date.fromisoformat(version).isoformat() != version:
            raise ValueError
    except ValueError:
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     'Invalid anthropic-version header: expected a valid YYYY-MM-DD date.')
    return None
