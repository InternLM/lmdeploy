# Copyright (c) OpenMMLab. All rights reserved.
"""DistServe-only OpenAI body fields and the public-path trust check.

`migration_request`, `with_cache`, and `preserve_cache` are injected by the
DistServe proxy onto Prefill/Decode replicas. They are not part of the public
OpenAI API and must not be honored from untrusted clients: a syntactically
valid but semantically bad `migration_request` can raise in the Decode
migration loop and tear down `EngineLoop` via `asyncio.FIRST_EXCEPTION`.
"""
from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus

from pydantic import ValidationError

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.conn.protocol import MigrationRequest
from lmdeploy.serve.openai.errors import create_error_response

DISTSERVE_PROXY_HEADER = 'x-lmdeploy-distserve-proxy'
DISTSERVE_PROXY_HEADER_VALUE = '1'
DISTSERVE_BODY_FIELDS = ('migration_request', 'with_cache', 'preserve_cache')

_DISTSERVE_FIELD_ERROR = (
    'DistServe-only fields (migration_request, with_cache, preserve_cache) '
    'are not accepted on the public OpenAI API.')


@dataclass(frozen=True)
class DistServeRequestFields:
    """DistServe fields forwarded into GenerationConfig."""

    migration_request: MigrationRequest | None = None
    with_cache: bool = False
    preserve_cache: bool = False


def distserve_proxy_headers() -> dict[str, str]:
    """Headers the DistServe proxy sets on Prefill/Decode replica calls."""
    return {DISTSERVE_PROXY_HEADER: DISTSERVE_PROXY_HEADER_VALUE}


def _header_value(raw_request, name: str) -> str:
    headers = getattr(raw_request, 'headers', None)
    if headers is None:
        return ''
    getter = getattr(headers, 'get', None)
    if getter is None:
        return ''
    value = getter(name)
    if value is None:
        value = getter(name.lower())
    if value is None:
        return ''
    return str(value).strip()


def is_trusted_distserve_replica_request(raw_request, engine_config) -> bool:
    """True when DistServe body fields may be forwarded into the engine."""
    role = getattr(engine_config, 'role', None)
    if role not in (EngineRole.Prefill, EngineRole.Decode):
        return False
    return _header_value(raw_request, DISTSERVE_PROXY_HEADER) == DISTSERVE_PROXY_HEADER_VALUE


def pop_distserve_request_fields(json_request: dict, raw_request, engine_config):
    """Pop DistServe-only fields from an OpenAI JSON body.

    Returns ``(fields, error_response)``. ``error_response`` is an HTTP 400
    when an untrusted client injected these fields; callers must not forward
    them into the engine in that case.
    """
    migration_raw = json_request.pop('migration_request', None)
    with_cache = bool(json_request.pop('with_cache', False))
    preserve_cache = json_request.pop('preserve_cache', False)
    has_distserve_fields = bool(migration_raw or with_cache or preserve_cache)
    if not has_distserve_fields:
        return DistServeRequestFields(), None

    if not is_trusted_distserve_replica_request(raw_request, engine_config):
        return DistServeRequestFields(), create_error_response(
            HTTPStatus.BAD_REQUEST, _DISTSERVE_FIELD_ERROR)

    migration_request = None
    if migration_raw:
        try:
            migration_request = MigrationRequest.model_validate(migration_raw)
        except (ValidationError, ValueError, TypeError) as exc:
            return DistServeRequestFields(), create_error_response(
                HTTPStatus.BAD_REQUEST,
                f'Invalid migration_request: {exc}')
    return DistServeRequestFields(
        migration_request=migration_request,
        with_cache=with_cache,
        preserve_cache=bool(preserve_cache),
    ), None
