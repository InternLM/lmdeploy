"""Hard-schema tool-call validation (walle MFJS cases via Kimi-Vendor-
Verifier)."""

from __future__ import annotations

from typing import Any

from utils.tool_call_json_schema_utils import (
    HARD_SCHEMA_MAX_TOKENS,
    kvv_validator,
    resolve_hard_schema_thinking,
)

from .conftest import _apply_hard_schema_marks, _ToolCallTestBase


@_apply_hard_schema_marks
class TestToolCallJsonSchema(_ToolCallTestBase):
    def test_tool_call_schema_matches_case_schema(
            self, backend, model_case, case_info: tuple[Any, Any, str, str]):
        case, schema, selection_reason, mode = case_info
        thinking, think_mode = resolve_hard_schema_thinking(model_case, backend)
        client, model_name = self._get_client()
        mod = kvv_validator()
        response = mod.send_tool_schema(
            client,
            model_name,
            schema,
            HARD_SCHEMA_MAX_TOKENS,
            thinking,
            think_mode,
            stream=mode == 'stream',
        )
        assert response.accepted, (
            f'{case.suite}:{case.line} [{mode}] ({selection_reason}) '
            f'tool schema rejected: {response.message}'
        )
        valid, message = mod.validate_arguments(schema, response.arguments)
        assert valid, (
            f'{case.suite}:{case.line} [{mode}] ({selection_reason}) '
            f'arguments validation failed: {message}; response: {response.message}'
        )
