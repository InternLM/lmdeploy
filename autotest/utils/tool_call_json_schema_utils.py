"""Autotest adapter for Kimi-Vendor-Verifier hard-schema tool-call tests."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

from utils.config_utils import (
    build_interface_launch_extra,
    get_config,
    get_interface_profiles,
    iter_model_yaml_entries,
)

HARD_SCHEMA_SKIP_NO_SUITE = (
    'model yaml interface profile does not include hard_schema suite'
)
HARD_SCHEMA_SKIP_NO_PARSER = (
    'hard_schema requires tool-call-parser in interface yaml extra'
)
HARD_SCHEMA_MAX_TOKENS = 8192


def _kvv_root() -> Path:
    override = os.environ.get('KIMI_VENDOR_VERIFIER_ROOT', '').strip()
    if override:
        return Path(override)
    path = get_config().get('kimi_vendor_verifier_path', '')
    if not path:
        raise FileNotFoundError(
            'kimi_vendor_verifier_path missing in env_paths.yml '
            '(or set KIMI_VENDOR_VERIFIER_ROOT)',
        )
    return Path(str(path))


@lru_cache(maxsize=1)
def kvv_validator():
    root = str(_kvv_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from tests.tool_call_json_schema import validator as mod

    return mod


def _case_dir() -> Path:
    raw = os.environ.get('WALLE_CASE_DIR', '').strip()
    if raw:
        return Path(raw)
    return _kvv_root() / 'testdata/walle_validator_cases/validator_cases'


def kvv_load_selected_cases() -> list[tuple[Any, Any, str]]:
    mod = kvv_validator()
    cases = mod.load_cases(_case_dir())
    return mod.select_cases(
        cases,
        selection='all',
        requested_cases=set(),
        max_cases=None,
    )


def _interface_profiles(model_case: str, backend: str) -> Iterator[tuple[dict, dict]]:
    for entry in iter_model_yaml_entries(model_case):
        for prof in get_interface_profiles(entry, backend):
            extra = build_interface_launch_extra(
                entry,
                backend,
                suites=prof.get('suites') or [],
                interface_extra=prof.get('extra'),
            )
            yield prof, extra


def model_has_hard_schema_suite(model_case: str, backend: str) -> bool:
    return _model_has_hard_schema_suite(model_case, backend)


def model_has_tool_call_parser(model_case: str, backend: str) -> bool:
    return _model_has_tool_call_parser(model_case, backend)


@cache
def _model_has_hard_schema_suite(model_case: str, backend: str) -> bool:
    return any('hard_schema' in (prof.get('suites') or []) for prof, _ in _interface_profiles(model_case, backend))


@cache
def _model_has_tool_call_parser(model_case: str, backend: str) -> bool:
    return any(extra.get('tool-call-parser') for _, extra in _interface_profiles(model_case, backend))


def resolve_hard_schema_thinking(model_case: str, backend: str) -> tuple[bool, str]:
    for _, extra in _interface_profiles(model_case, backend):
        enabled = extra.get('enable_thinking', extra.get('enable-thinking'))
        if enabled is True:
            return True, 'opensource'
        chat_kwargs = extra.get('chat-template-kwargs') or {}
        if chat_kwargs.get('enable_thinking') is True:
            return True, 'opensource'
    return False, 'opensource'
