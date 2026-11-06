# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from json.encoder import encode_basestring
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from lmdeploy.serve.openai.protocol import DeltaToolCall
from lmdeploy.utils import get_logger

from .tool_parser import ToolParser

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest


logger = get_logger('lmdeploy')

# Unified depth budget for schema resolution across ``$ref`` chains and union
# nesting. Hostile or accidental cycles in request-provided tool schemas must
# degrade to the string-preserving fallback instead of raising RecursionError
# after a full generation turn.
_MAX_SCHEMA_RESOLVE_DEPTH = 32


def _loads_strict_json(text: str) -> Any:
    """Parse ``text`` as strict JSON, rejecting the ``NaN``/``Infinity``
    extensions ``json.loads`` accepts by default.

    A coerced non-finite float would reach ``arguments`` and make the whole
    payload unparseable for strict JSON clients. The hook raises
    ``ValueError`` so callers fall back to the raw string form.
    """

    def _reject_constant(token: str) -> Any:
        raise ValueError(f'non-standard JSON constant: {token}')

    return json.loads(text, parse_constant=_reject_constant)


def _is_strict_json_value(value: Any) -> bool:
    """Return whether ``value`` serializes to strict JSON (no non-finite
    floats, at any nesting depth)."""

    # Iterative on purpose: model output can nest deeply enough that a
    # recursive walk would itself raise RecursionError.
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, bool):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                return False
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return True


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

    Complete parsing is inherited from :class:`ToolParser`, which feeds the
    full payload through this same streaming state machine and joins its
    emitted argument fragments.
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
        self._function_root_schemas: dict[str, dict[str, Any]] = {}
        self._has_emitted_json_start = False
        self._json_closed = False
        self._emitted_arg_count = 0
        self._state = XmlParseState()
        self._arg_state = _XmlArgState()

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Cache parameter schemas and apply common tool request
        adjustments."""
        built = self._build_function_param_schemas(request)
        if (not isinstance(built, tuple) or len(built) != 2
                or not all(isinstance(entry, dict) for entry in built)):
            # Overrides predating the two-map return value must not silently
            # poison schema lookups with unpacked mapping keys.
            logger.warning('ignoring invalid _build_function_param_schemas result: '
                           'expected (param schemas, root schemas) dicts')
            built = {}, {}
        self._function_param_schemas, self._function_root_schemas = built
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
        value_end = self._find_structure(payload, close_tag, pos)
        if value_end >= 0:
            self._consume_arg_delta(payload[pos:value_end], json_fragments)
            self._finish_arg(json_fragments)
            self._state.phase = 'arg_start'
            return value_end + len(close_tag)

        if self._active_boundaries is None:
            raw_end = self._stable_arg_value_end(payload, pos)
        elif self._uses_token_ids(close_tag):
            raw_end = len(payload)
        else:
            prefixes = tuple(close_tag[:i] for i in range(1, len(close_tag)))
            raw_end = self._stable_prefix_end(payload, prefixes, pos)
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

    def _build_function_param_schemas(
            self, request: ChatCompletionRequest
    ) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
        """Build function->parameter and function->root schema maps."""
        if not request.tools:
            return {}, {}

        out: dict[str, dict[str, dict[str, Any]]] = {}
        roots: dict[str, dict[str, Any]] = {}
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
                roots[tool.function.name] = parameters
        return out, roots

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
        schema_type, allows_null = self._resolve_schema_coercion(
            param_schema, self._get_function_root_schema(self._state.func_name))
        # A nullable schema must buffer: the bare word ``null`` can only be
        # recognized once the value is complete, so it cannot stream raw.
        if param_schema is not None and (schema_type != 'string' or allows_null):
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
            root_schema = self._get_function_root_schema(func_name)
            value = self._coerce_arg_value(raw_value, schema, root_schema)
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

    def _get_function_root_schema(self, func_name: str | None) -> dict[str, Any] | None:
        """Return the full parameters schema holding ``$defs`` for a
        function."""
        root_schema = self._function_root_schemas.get(func_name)
        return root_schema if isinstance(root_schema, dict) else None

    def _stable_arg_value_end(self, payload: str, start: int) -> int:
        """Return the stable value end using token-aligned marker prefixes."""
        return self._stable_prefix_end(payload, self.arg_value_close_prefixes, start)

    @staticmethod
    def _json_type_of(value: Any) -> str | None:
        """Map a Python value to its JSON Schema type name."""
        if isinstance(value, bool):
            return 'boolean'
        if isinstance(value, int):
            return 'integer'
        if isinstance(value, float):
            return 'number'
        if isinstance(value, str):
            return 'string'
        if value is None:
            return 'null'
        if isinstance(value, list):
            return 'array'
        if isinstance(value, dict):
            return 'object'
        return None

    @staticmethod
    def _deref_schema(schema: dict[str, Any] | None, root_schema: dict[str, Any] | None,
                      depth: int = 0) -> dict[str, Any] | None:
        """Resolve a local ``$ref`` pointer (``#/$defs/...``) against the root.

        Pointer tokens are unescaped per RFC 6901 first (``~1`` decodes to ``/``,
        ``~0`` to ``~``). Returns the referenced schema, or the original schema when the pointer
        is absent, external, unresolvable, or the shared ``depth`` budget is exhausted.
        """
        if not isinstance(schema, dict) or depth >= _MAX_SCHEMA_RESOLVE_DEPTH:
            return schema
        ref = schema.get('$ref')
        if not isinstance(ref, str) or not ref.startswith('#/'):
            return schema
        target: Any = root_schema if isinstance(root_schema, dict) else None
        for token in ref[2:].split('/'):
            # RFC 6901: decode pointer escapes, ``~1`` before ``~0``.
            part = token.replace('~1', '/').replace('~0', '~')
            if not isinstance(target, dict) or part not in target:
                return schema
            target = target[part]
        if not isinstance(target, dict):
            return schema
        return XmlToolParser._deref_schema(target, root_schema, depth + 1)

    @classmethod
    def _resolve_schema_coercion(cls, param_schema: dict[str, Any] | None,
                                 root_schema: dict[str, Any] | None = None,
                                 depth: int = 0,
                                 ) -> tuple[str | None, bool]:
        """Resolve the effective JSON type and nullability of a schema.

        Returns ``(schema_type, allows_null)``. ``$ref`` pointers are
        resolved against ``root_schema``; constraints adjacent to a ``$ref``
        apply alongside the referenced target (Draft 2020-12), so the
        referencing schema's own ``type``/``enum``/``const`` governs
        coercion when present. ``enum`` and ``const`` values imply their
        type; ``schema_type`` is ``None`` only when no single JSON type
        governs the value (mixed unions, typeless schemas).

        ``depth`` is one shared budget across ``$ref`` chains and union
        nesting: cyclic or pathologically deep schemas degrade to the
        string-preserving fallback (logged) instead of recursing without
        bound. A schema whose ``$ref`` cannot be resolved (external pointer,
        missing target, exhausted budget) also resolves to ``string`` so raw
        values are kept instead of guessing a JSON type.
        """
        if not isinstance(param_schema, dict):
            return None, False
        if depth >= _MAX_SCHEMA_RESOLVE_DEPTH:
            logger.debug('tool schema resolution exceeded %d levels; coercing arguments as '
                         'raw strings', _MAX_SCHEMA_RESOLVE_DEPTH)
            # A budget-exhausted ``$ref`` chain degrades like an unresolvable
            # one; exhausted union nesting keeps the JSON-value fallback.
            if isinstance(param_schema.get('$ref'), str):
                return 'string', False
            return None, False
        target = cls._deref_schema(param_schema, root_schema, depth)
        if target is param_schema:
            # No local ``$ref`` was followed: interpret the schema itself.
            own_type, own_nullable = cls._resolve_local_schema(param_schema, root_schema, depth)
            if own_type is None and isinstance(param_schema.get('$ref'), str):
                logger.debug('unresolvable $ref %s in tool schema; coercing arguments as raw '
                             'strings', param_schema['$ref'])
                return 'string', False
            return own_type, own_nullable
        own_type, own_nullable = cls._resolve_local_schema(param_schema, root_schema, depth)
        target_type, target_nullable = cls._resolve_schema_coercion(target, root_schema, depth + 1)
        if own_type is None:
            return target_type, target_nullable
        # The sibling constraints and the referenced target both apply: the
        # sibling type governs coercion, and null is allowed only when both
        # sides accept it.
        return own_type, own_nullable and target_nullable

    @classmethod
    def _resolve_local_schema(cls, schema: dict[str, Any] | None,
                              root_schema: dict[str, Any] | None = None,
                              depth: int = 0,
                              ) -> tuple[str | None, bool]:
        """Resolve a schema's own constraints, ignoring a ``$ref`` keyword.

        Union branches and ``allOf`` members resolve through
        ``_resolve_schema_coercion`` so ``$ref`` pointers nested inside them
        are still followed.
        """
        if not isinstance(schema, dict):
            return None, False
        schema_type = schema.get('type')
        if isinstance(schema_type, str):
            return schema_type, schema_type == 'null'
        if isinstance(schema_type, list):
            types = [item for item in schema_type if isinstance(item, str)]
            nullable = 'null' in types
            for item in types:
                if item != 'null':
                    return item, nullable
            return None, nullable
        if 'anyOf' in schema or 'oneOf' in schema:
            # Resolve a union only when every non-null branch agrees on one
            # JSON type; mixed unions coerce the raw value as JSON instead.
            branches = schema.get('anyOf') or schema.get('oneOf') or []
            resolved = [cls._resolve_schema_coercion(branch, root_schema, depth + 1)
                        for branch in branches if isinstance(branch, dict)]
            nullable = any(t == 'null' or allows_null for t, allows_null in resolved)
            distinct = {t for t, _ in resolved if t not in (None, 'null')}
            if len(distinct) == 1:
                return distinct.pop(), nullable
            return None, nullable
        if isinstance(schema.get('allOf'), list):
            # Every ``allOf`` member constrains the same value: coerce to the
            # common type when all resolved members agree, and allow null
            # only when every member accepts it.
            resolved = [cls._resolve_schema_coercion(branch, root_schema, depth + 1)
                        for branch in schema['allOf'] if isinstance(branch, dict)]
            if resolved:
                nullable = all(allows_null for _, allows_null in resolved)
                distinct = {t for t, _ in resolved if t not in (None, 'null')}
                if len(distinct) == 1:
                    return distinct.pop(), nullable
                return None, nullable
            # An empty or boolean-only ``allOf`` constrains nothing: keep
            # resolving the schema's own keywords below.
        if isinstance(schema.get('enum'), list):
            value_types = {cls._json_type_of(value) for value in schema['enum']}
            nullable = 'null' in value_types
            distinct = {t for t in value_types if t not in (None, 'null')}
            if len(distinct) == 1:
                return distinct.pop(), nullable
            return None, nullable
        if 'const' in schema:
            const_type = cls._json_type_of(schema['const'])
            return const_type, const_type == 'null'
        if isinstance(schema.get('properties'), dict):
            return 'object', False
        if schema.get('items') is not None:
            return 'array', False
        return None, False

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
                parsed_val = _loads_strict_json(raw_value)
                return parsed_val if isinstance(parsed_val, str) else original
            except (ValueError, RecursionError):
                return original

        if schema_type == 'integer':
            try:
                parsed_val = _loads_strict_json(raw_value)
            except (ValueError, RecursionError):
                parsed_val = raw_value
            if isinstance(parsed_val, bool):
                return original
            if isinstance(parsed_val, int):
                return parsed_val
            return original

        if schema_type == 'number':
            try:
                parsed_val = _loads_strict_json(raw_value)
            except (ValueError, RecursionError):
                parsed_val = raw_value
            if isinstance(parsed_val, bool):
                return original
            if isinstance(parsed_val, (int, float)) and _is_strict_json_value(parsed_val):
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
                parsed_val = _loads_strict_json(raw_value)
            except (ValueError, RecursionError):
                return original
            if isinstance(parsed_val, list) and _is_strict_json_value(parsed_val):
                return parsed_val
            return original

        if schema_type == 'object':
            try:
                parsed_val = _loads_strict_json(raw_value)
            except (ValueError, RecursionError):
                return original
            if isinstance(parsed_val, dict) and _is_strict_json_value(parsed_val):
                return parsed_val
            return original

        return original

    @staticmethod
    def _coerce_json_value(raw_value: str) -> Any:
        """Coerce a complete raw value whose schema declares no single JSON
        type.

        Mixed unions and typeless schemas impose no scalar type, so the model-emitted bare JSON form is kept: quoted
        values decode as strings, other valid JSON keeps its native type, and anything unparseable or non-strict
        (``NaN``, ``Infinity``) keeps its original string form.
        """
        original = raw_value
        raw_value = raw_value.strip()
        if not raw_value:
            return original
        if raw_value.startswith('"'):
            try:
                return _loads_strict_json(raw_value)
            except (ValueError, RecursionError):
                return original
        try:
            parsed_val = _loads_strict_json(raw_value)
        except (ValueError, RecursionError):
            return original
        return parsed_val if _is_strict_json_value(parsed_val) else original

    def _coerce_arg_value(self, raw_value: Any, schema: dict[str, Any] | None,
                          root_schema: dict[str, Any] | None = None) -> Any:
        """Apply schema-aware coercion to one argument value."""
        if not isinstance(raw_value, str):
            return raw_value
        if schema is None:
            return raw_value
        schema_type, allows_null = self._resolve_schema_coercion(schema, root_schema)
        if raw_value.strip().lower() == 'null':
            # Models emit a bare ``null`` routinely for optional parameters;
            # keep it as the string unless the schema really accepts null.
            return None if allows_null else raw_value
        if schema_type is None:
            return self._coerce_json_value(raw_value)
        return self._coerce_value(raw_value, schema_type)
