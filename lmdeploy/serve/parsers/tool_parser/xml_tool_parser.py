# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
from dataclasses import dataclass, field
from json.encoder import encode_basestring
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from lmdeploy.serve.openai.protocol import (
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)

from .tool_parser import ToolParser

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest


XmlParsePhase = Literal['function', 'arg_start', 'arg_name', 'arg_value', 'done']


@dataclass
class XmlParseState:
    """Shared semantic state for one streamed XML-like tool call.

    ``phase`` starts at ``function`` because supported formats place the
    function identity before every argument. ``func_name`` persists for the
    whole call, while ``arg_name`` identifies only the active argument.
    """

    phase: XmlParsePhase = 'function'
    func_name: str | None = None
    arg_name: str | None = None


@dataclass
class _XmlArgState:
    """Internal state used to stream or buffer the active argument value."""

    mode: Literal['undecided', 'buffered', 'streaming'] = 'undecided'
    pending_suffix: str = ''
    buffered_parts: list[str] = field(default_factory=list)


class XmlToolParser(ToolParser):
    """Base class for sequential, function-first XML-like tool calls.

    One outer tool block must contain one function identity followed by zero or
    more non-interleaved arguments. Each argument consists of a name, raw
    value, and closing marker. Argument-first formats do not satisfy this
    contract because schema lookup requires the function identity. Formats
    without a close-delimited value are also outside the contract.

    Streaming follows this state machine::

        function
          |\
          | +-- no-argument terminator -----------------------> done
          | _consume_function
          v
        arg_start -- payload terminator --> done
          | _consume_arg_start
          v
        arg_name
          | _consume_arg_name
          v
        arg_value
          | XmlToolParser._consume_arg_value
          +-----------------------------> arg_start

    A syntax hook returns ``None`` when it needs more input, or a tuple
    containing the next cursor and the syntax it recognized. Outer closing
    markers must remain unconsumed for :meth:`ToolParser.feed_tool_block`.

    Subclasses own syntax markers and recognition for the first three phases;
    they return function names, argument names, and the applicable next phase
    without accessing ``self._state``. This class alone applies those results
    to semantic state. It also owns value streaming, JSON emission, schema
    coercion, and the transition from ``done`` to ``_payload_closed``.
    Unquoted string values are emitted immediately; only undecided syntax and
    non-streamable typed values are retained. Subclasses supply
    ``arg_value_close_tag`` and its token-aligned proper prefixes in
    ``arg_value_close_prefixes``. A dialect may override
    ``_stable_arg_value_end`` when it can dispatch those prefixes more
    efficiently.

    Complete parsing remains dialect-specific. Subclasses pass ordered raw
    argument pairs to ``_build_tool_call`` so conversion and duplicate-key
    preservation stay common.
    """

    # Raw argument values end immediately before this dialect-specific marker.
    arg_value_close_tag: ClassVar[str]
    # Proper value-close prefixes that end at decoded-token boundaries.
    # An empty tuple means the complete closing marker is one token.
    arg_value_close_prefixes: ClassVar[tuple[str, ...]] = ()
    # Whether to remove one protocol-added newline around each complete value.
    strip_value_newlines: ClassVar[bool] = False

    def __init__(self):
        super().__init__()
        self._function_param_schemas: dict[str, dict[str, dict[str, Any]]] = {}
        self._has_emitted_json_start = False
        self._json_closed = False
        self._emitted_arg_count = 0
        self._state = XmlParseState()
        self._arg_state = _XmlArgState()

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Cache parameter schemas and apply common tool request
        adjustments."""
        self._function_param_schemas = self._build_function_param_schemas(request)
        return super().adjust_request(request)

    def begin_tool_block(self) -> None:
        """Begin a logical call and reset XML-to-JSON streaming state."""
        super().begin_tool_block()
        self._begin_call()
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        """Reset syntax, argument, and emitted-JSON state for one block."""
        self._has_emitted_json_start = False
        self._json_closed = False
        self._emitted_arg_count = 0
        self._state = XmlParseState()
        self._arg_state = _XmlArgState()

    def _consume_payload(self, payload: str, json_fragments: list[str], *, final: bool) -> int:
        """Drive the format-specific state machine over buffered payload text.

        Syntax hooks return recognized names and transitions without mutating
        shared state. ``None`` retains the unconsumed suffix. A syntax hook may
        return ``pos`` because every successful syntax result also changes the
        phase. JSON produced by the base value consumer is appended to
        ``json_fragments``.

        Args:
            payload: Buffered text inside the outer tool block.
            json_fragments: Destination for OpenAI-compatible argument JSON.
            final: Whether no more generated text will follow.

        Returns:
            Number of leading payload characters that were consumed.
        """
        pos = 0
        state = self._state
        payload_len = len(payload)

        while pos < payload_len:
            phase = state.phase
            if phase == 'function':
                result = self._consume_function(payload, pos, final)
                if result is None:
                    break
                pos, state.func_name, state.phase = result
            elif phase == 'arg_start':
                result = self._consume_arg_start(payload, pos)
                if result is None:
                    break
                pos, state.phase = result
            elif phase == 'arg_name':
                result = self._consume_arg_name(payload, pos)
                if result is None:
                    break
                pos, state.arg_name = result
                state.phase = 'arg_value'
            elif phase == 'arg_value':
                next_pos = self._consume_arg_value(payload, pos, json_fragments)
                if next_pos is None:
                    break
                pos = next_pos
            elif phase == 'done':
                while pos < payload_len and payload[pos].isspace():
                    pos += 1
                break
            else:
                break

        if state.phase == 'done':
            self._payload_closed = True
        return pos

    def _consume_function(
        self,
        payload: str,
        pos: int,
        final: bool,
    ) -> tuple[int, str, Literal['arg_start', 'done']] | None:
        """Resolve the leading function identity.

        Return the next stable cursor, function name, and next phase. Return
        ``None`` when the identity or following syntax is incomplete.
        """
        raise NotImplementedError('XmlToolParser._consume_function has not been implemented!')

    def _consume_arg_start(self, payload: str, pos: int) -> tuple[int, Literal['arg_name', 'done']] | None:
        """Consume an argument opener or recognize the payload terminator.

        Return the next stable cursor and either ``arg_name`` for an opener or
        ``done`` for a terminator. An outer terminator must not be included in
        the returned prefix. Return ``None`` while neither decision is stable.
        """
        raise NotImplementedError('XmlToolParser._consume_arg_start has not been implemented!')

    def _consume_arg_name(self, payload: str, pos: int) -> tuple[int, str] | None:
        """Resolve the argument name and advance to its raw value.

        Return the first raw-value position and argument name. The driver then
        enters ``arg_value``. Return ``None`` until both are known.
        """
        raise NotImplementedError('XmlToolParser._consume_arg_name has not been implemented!')

    def _consume_arg_value(self, payload: str, pos: int, json_fragments: list[str]) -> int | None:
        """Consume stable raw value text and finish on the configured
        marker."""
        close_tag = self.arg_value_close_tag
        value_end = payload.find(close_tag, pos)
        if value_end >= 0:
            self._consume_arg_delta(payload[pos:value_end], json_fragments)
            self._finish_arg(json_fragments)
            self._state.phase = 'arg_start'
            return value_end + len(close_tag)

        raw_end = self._stable_arg_value_end(payload, pos)
        if raw_end == pos:
            # The entire remaining payload, i.e. payload[pos:], is a possible closing-tag prefix.
            # Keep it buffered until more input resolves the boundary.
            return None
        self._consume_arg_delta(payload[pos:raw_end], json_fragments)
        return raw_end

    def _consume_stream_payload(self, text: str, deltas: list[DeltaToolCall], *, final: bool) -> int:
        """Consume XML-like payload text and emit function-call deltas.

        Format-specific syntax parsing is delegated to ``_consume_payload``.
        This method owns the shared emission order: function name first when
        available, followed by coalesced JSON argument fragments.

        Args:
            text: Buffered payload text after the outer opening marker.
            deltas: Destination for parsed function-name and argument deltas.
            final: Whether no more generated text will follow.

        Returns:
            Number of leading characters that may be discarded from ``text``.
        """
        json_fragments: list[str] = []
        consumed = self._consume_payload(text, json_fragments, final=final)

        if self._state.func_name is not None and not self._name_emitted:
            self._emit_delta(
                deltas,
                name=self._state.func_name,
            )
            self._name_emitted = True

        should_close = self._payload_closed or final
        if should_close and not self._has_emitted_json_start:
            json_fragments.append('{')
            self._has_emitted_json_start = True
        if should_close and self._has_emitted_json_start and not self._json_closed:
            json_fragments.append('}')
            self._json_closed = True

        if json_fragments:
            self._emit_delta(
                deltas,
                arguments=''.join(json_fragments),
            )
        if final and should_close:
            self._payload_closed = True
        return consumed

    def _build_function_param_schemas(self, request: ChatCompletionRequest) -> dict[str, dict[str, dict[str, Any]]]:
        """Build function->parameter schema map from request tools."""
        if not request.tools:
            return {}

        out: dict[str, dict[str, dict[str, Any]]] = {}
        for tool in request.tools:
            parameters = tool.function.parameters
            if not isinstance(parameters, dict):
                continue
            properties = parameters.get('properties')
            if not isinstance(properties, dict):
                continue

            param_schemas = {name: schema for name, schema in properties.items() if isinstance(schema, dict)}
            if param_schemas:
                out[tool.function.name] = param_schemas
        return out

    def _consume_arg_delta(self, raw: str, json_fragments: list[str]) -> None:
        """Buffer typed or quoted values and stream plain string fragments.

        ``buffered_parts`` retains fragments while the mode is ``undecided``.
        The first non-whitespace fragment selects ``buffered`` for a non-string
        schema or quoted value, and ``streaming`` otherwise. The selected mode
        remains unchanged until :meth:`_finish_arg` resets the argument state.

        Args:
            raw: Stable raw value text that cannot belong to the closing marker.
            json_fragments: Destination for OpenAI-compatible JSON fragments.
        """
        arg_name = self._state.arg_name
        if arg_name is None:
            return

        arg_state = self._arg_state
        if arg_state.mode == 'buffered':
            arg_state.buffered_parts.append(raw)
            return

        if arg_state.mode == 'streaming':
            self._stream_string_delta(raw, json_fragments)
            return

        arg_state.buffered_parts.append(raw)
        # Earlier undecided fragments contain only whitespace, so the first
        # non-whitespace character, if any, must occur in the current fragment.
        if not raw.lstrip():
            return

        text = ''.join(arg_state.buffered_parts)
        if self.strip_value_newlines:
            if text.startswith('\r\n'):
                text = text[2:]
            elif text.startswith('\n'):
                text = text[1:]

        param_schema = self._get_param_schema(self._state.func_name, arg_name)
        schema_type = self._get_schema_type(param_schema)
        if param_schema is not None and schema_type != 'string':
            arg_state.mode = 'buffered'
            return
        if text.lstrip().startswith('"'):
            arg_state.mode = 'buffered'
            return

        arg_state.mode = 'streaming'
        arg_state.buffered_parts.clear()
        self._append_streaming_arg_header(json_fragments, arg_name)
        self._stream_string_delta(text, json_fragments)

    def _stream_string_delta(self, text: str, json_fragments: list[str]) -> None:
        """Append escaped text while retaining a possible final newline."""
        arg_state = self._arg_state
        text = arg_state.pending_suffix + text
        arg_state.pending_suffix = ''
        if self.strip_value_newlines and text:
            last = text[-1]
            if last == '\n':
                if len(text) > 1 and text[-2] == '\r':
                    stable = text[:-2]
                    arg_state.pending_suffix = '\r\n'
                else:
                    stable = text[:-1]
                    arg_state.pending_suffix = '\n'
            elif last == '\r':
                stable = text[:-1]
                arg_state.pending_suffix = '\r'
            else:
                stable = text
        else:
            stable = text
        if not stable:
            return

        json_fragments.append(encode_basestring(stable)[1:-1])

    def _finish_arg(self, json_fragments: list[str]) -> None:
        """Finish the active argument and reset its transient state.

        Streaming strings only need their retained suffix and closing quote. Buffered values are normalized, schema-
        coerced, and serialized in one fragment.
        """
        arg_name = self._state.arg_name
        if arg_name is None:
            self._arg_state = _XmlArgState()
            return

        if self._arg_state.mode == 'streaming':
            if self._arg_state.pending_suffix == '\r':
                json_fragments.append('\\r')
            json_fragments.append('"')
        else:
            raw_value = ''.join(self._arg_state.buffered_parts)
            raw_value = self._normalize_complete_value(raw_value)
            func_name = self._state.func_name
            schema = self._get_param_schema(func_name, arg_name)
            value = self._coerce_arg_value(raw_value, schema)
            self._append_completed_arg(json_fragments, arg_name, value)
        self._state.arg_name = None
        self._arg_state = _XmlArgState()

    def _append_json_start(self, json_fragments: list[str]) -> None:
        """Append the opening argument-object brace exactly once."""
        if not self._has_emitted_json_start:
            json_fragments.append('{')
            self._has_emitted_json_start = True

    def _append_completed_arg(self, json_fragments: list[str], arg_name: str, value: Any) -> None:
        """Serialize one complete key-value pair into the argument object."""
        self._append_json_start(json_fragments)
        prefix = ', ' if self._emitted_arg_count else ''
        key = json.dumps(arg_name, ensure_ascii=False)
        json_fragments.append(f'{prefix}{key}: {json.dumps(value, ensure_ascii=False)}')
        self._emitted_arg_count += 1

    def _append_streaming_arg_header(self, json_fragments: list[str], arg_name: str) -> None:
        """Append a string argument's key and opening quote."""
        self._append_json_start(json_fragments)
        prefix = ', ' if self._emitted_arg_count else ''
        json_fragments.append(f'{prefix}{json.dumps(arg_name, ensure_ascii=False)}: "')
        self._emitted_arg_count += 1

    def _normalize_complete_value(self, raw_value: str) -> str:
        """Remove protocol formatting newlines from a complete value."""
        if not self.strip_value_newlines:
            return raw_value
        if raw_value.startswith('\r\n'):
            raw_value = raw_value[2:]
        elif raw_value.startswith('\n'):
            raw_value = raw_value[1:]
        if raw_value.endswith('\r\n'):
            raw_value = raw_value[:-2]
        elif raw_value.endswith('\n'):
            raw_value = raw_value[:-1]
        return raw_value

    def _get_param_schema(self, func_name: str | None, param_name: str) -> dict[str, Any] | None:
        """Return the cached schema for one function parameter, if
        available."""
        if func_name is None:
            return None
        param_schema = self._function_param_schemas.get(func_name, {}).get(param_name)
        return param_schema if isinstance(param_schema, dict) else None

    @staticmethod
    def _get_schema_type(schema: dict[str, Any] | None) -> str | None:
        """Return the effective JSON type from an optional parameter schema."""
        return XmlToolParser._resolve_schema_type(schema) if schema is not None else None

    def _stable_arg_value_end(self, payload: str, start: int) -> int:
        """Return the stable value end using token-aligned marker prefixes."""
        return self._stable_prefix_end(payload, self.arg_value_close_prefixes, start)

    @staticmethod
    def _resolve_schema_type(param_schema: dict[str, Any]) -> str | None:
        """Resolve a scalar or nullable-list JSON Schema ``type`` value."""
        schema_type = param_schema.get('type')
        if isinstance(schema_type, str):
            return schema_type
        if isinstance(schema_type, list):
            for item in schema_type:
                if isinstance(item, str) and item != 'null':
                    return item
            for item in schema_type:
                if isinstance(item, str):
                    return item
        return None

    @staticmethod
    def _coerce_value(raw_value: str, schema_type: str | None) -> Any:
        """Coerce a complete raw value when it matches the declared JSON type.

        Values that cannot be converted without guessing are returned in their original string form.
        """
        original = raw_value
        raw_value = raw_value.strip()
        if schema_type is None or schema_type == 'string':
            if not raw_value.startswith('"'):
                return original
            try:
                parsed_val = json.loads(raw_value)
                return parsed_val if isinstance(parsed_val, str) else original
            except json.JSONDecodeError:
                return original

        if schema_type == 'integer':
            try:
                parsed_val = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed_val = raw_value
            if isinstance(parsed_val, bool):
                return original
            if isinstance(parsed_val, int):
                return parsed_val
            return original

        if schema_type == 'number':
            try:
                parsed_val = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed_val = raw_value
            if isinstance(parsed_val, bool):
                return original
            if isinstance(parsed_val, (int, float)):
                return parsed_val
            return original

        if schema_type == 'boolean':
            lowered = raw_value.lower()
            if lowered == 'true':
                return True
            if lowered == 'false':
                return False
            return original

        if schema_type == 'null':
            return None if raw_value.lower() == 'null' else original

        if schema_type == 'array':
            try:
                parsed_val = json.loads(raw_value)
            except json.JSONDecodeError:
                return original
            return parsed_val if isinstance(parsed_val, list) else original

        if schema_type == 'object':
            try:
                parsed_val = json.loads(raw_value)
            except json.JSONDecodeError:
                return original
            return parsed_val if isinstance(parsed_val, dict) else original

        return original

    def _coerce_arg_value(self, raw_value: Any, schema: dict[str, Any] | None) -> Any:
        """Apply schema-aware coercion to one argument value."""
        if not isinstance(raw_value, str):
            return raw_value
        return self._coerce_value(raw_value, self._get_schema_type(schema)) if schema is not None else raw_value

    def _get_coerced_args(self, func_name: str | None, raw_arg_pairs: list[tuple[str, str]]) -> list[tuple[str, Any]]:
        """Coerce argument pairs without changing their order or
        multiplicity."""
        if not func_name or not raw_arg_pairs:
            return raw_arg_pairs
        param_schemas = self._function_param_schemas.get(func_name, {})
        if not param_schemas:
            return raw_arg_pairs

        coerced: list[tuple[str, Any]] = []
        for key, value in raw_arg_pairs:
            schema = param_schemas.get(key)
            coerced.append((key, self._coerce_arg_value(value, schema)))
        return coerced

    def _build_tool_call(self, func_name: str, raw_arg_pairs: list[tuple[str, str]]) -> ToolCall:
        """Build one tool call from ordered, potentially duplicate raw
        pairs."""
        arg_pairs = self._get_coerced_args(func_name, raw_arg_pairs)
        return ToolCall(function=FunctionCall(name=func_name, arguments=self._dump_argument_pairs(arg_pairs)))

    @staticmethod
    def _dump_argument_pairs(arg_pairs: list[tuple[str, Any]]) -> str:
        """Serialize ordered pairs as a JSON object while preserving duplicate
        keys."""
        fields = (
            f'{json.dumps(name, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}'
            for name, value in arg_pairs
        )
        return '{' + ', '.join(fields) + '}'
