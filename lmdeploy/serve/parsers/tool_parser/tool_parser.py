# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import shortuuid
from mmengine import Registry

from lmdeploy.serve.openai.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest, Tool

ToolParserManager = Registry('tool_parser', locations=['lmdeploy.serve.parsers.tool_parser'])

# Client-visible indices are non-negative; these values encode filtering state.
_PENDING_INDEX = -2
_REJECTED_INDEX = -1


@dataclass
class _ToolCallParts:
    """Complete-call fields accumulated from streaming deltas."""

    call_id: str | None = None
    name: str | None = None
    arguments: list[str] = field(default_factory=list)


def dump_tools(request: ChatCompletionRequest) -> ChatCompletionRequest:
    """Dump tools to a list of dicts to fit jinja chat template."""
    from lmdeploy.serve.openai.protocol import AllowedToolChoice

    if isinstance(request.tool_choice, AllowedToolChoice):
        allowed_names: set[str] = set()
        allowed_functions: list[dict] = []
        for t in request.tool_choice.allowed_tools.tools:
            func = t.get('function', {})
            if isinstance(func, dict) and 'name' in func:
                allowed_names.add(func['name'])
                allowed_functions.append(func)

        if not request.tools:
            return request.model_copy(update={'tools': allowed_functions or None})

        request_tool_names = {item.function.name for item in request.tools}
        missing = sorted(allowed_names - request_tool_names)
        if missing:
            raise ValueError(f'Allowed tool(s) not found in request.tools: {missing}')

        tools = [item.function.model_dump() for item in request.tools if item.function.name in allowed_names]
        return request.model_copy(update={'tools': tools})

    if not request.tools:
        return request.model_copy(update={'tools': None})

    if not isinstance(request.tool_choice, str):
        tools = [
            item.function.model_dump() for item in request.tools
            if item.function.name == request.tool_choice.function.name
        ]
    else:
        tools = [item.function.model_dump() for item in request.tools]
    return request.model_copy(update={'tools': tools})


class ToolParser:
    """Base class for model-specific tool-call boundary extractors.

    Streaming uses two distinct completion states. ``_payload_closed`` means
    the model-specific parser has reached the end of the inner payload; the
    outer closing marker may still be absent at that point. ``block_closed``
    means the complete outer block has been consumed, or the final stream chunk
    has forced it closed, so the response parser may leave tool mode.
    """

    structural_tag_model: ClassVar[str | None] = None
    reasoning_structural_tag_model: ClassVar[str | None] = None
    # Proper close-marker prefixes that end at decoded-token boundaries.
    # An empty tuple means the complete closing marker is one token.
    tool_close_prefixes: ClassVar[tuple[str, ...]] = ()

    def __init__(self):
        self._close_tag: str | None = None
        self._active_tool_call_id: str = ''
        self._active_tool_index: int = -1
        self._name_emitted: bool = False
        self._allowed_tool_names: set[str] | None = None
        self._output_index: int = _PENDING_INDEX
        self._next_output_index: int = 0
        self._first_delta: bool = True
        self._pending_deltas: list[DeltaToolCall] = []
        self._payload_closed: bool = False
        self.block_closed: bool = False

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust request payload before rendering, if needed."""
        request = dump_tools(request)
        self._allowed_tool_names = self._get_allowed_tool_names(request)
        return request

    @classmethod
    def build_required_response_format(cls, tools: list[Tool], *, reasoning: bool) -> dict[str, Any]:
        """Build the XGrammar structural-tag format for required tool calls."""
        structural_tag_model = cls.structural_tag_model
        if reasoning and cls.reasoning_structural_tag_model is not None:
            structural_tag_model = cls.reasoning_structural_tag_model
        if structural_tag_model is None:
            raise ValueError(f'Tool parser {cls.__name__!r} does not support `tool_choice="required"`.')
        import xgrammar as xgr

        return xgr.get_model_structural_tag(
            structural_tag_model,
            [tool.model_dump(mode='json') for tool in tools],
            tool_choice='required',
            reasoning=reasoning,
        ).model_dump(mode='json')

    @staticmethod
    def _get_allowed_tool_names(request: ChatCompletionRequest) -> set[str]:
        """Return function names exposed by the effective request tools."""
        names: set[str] = set()
        for tool in request.tools or []:
            if isinstance(tool, dict):
                function = tool.get('function', tool)
                name = function.get('name') if isinstance(function, dict) else None
            else:
                name = tool.function.name
            if isinstance(name, str):
                names.add(name)
        return names

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        """Return tool opening tag string, or None if unsupported."""
        raise NotImplementedError('ToolParser.get_tool_open_tag has not been implemented!')

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        """Return tool closing tag string, or None if unsupported."""
        raise NotImplementedError('ToolParser.get_tool_close_tag has not been implemented!')

    def begin_tool_block(self) -> None:
        """Initialize one streamed outer block after its opening marker.

        Concrete parsers begin each logical call with ``_begin_call`` once its
        call boundary is known.
        """
        self._close_tag = self.get_tool_close_tag()
        self._payload_closed = False
        self.block_closed = False

    def _begin_call(self, call_id: str | None = None) -> None:
        """Initialize routing state for the next sequential tool call.

        Args:
            call_id: Model-provided call ID. An OpenAI-compatible ID is
                generated when the protocol does not provide one.
        """
        self._active_tool_index += 1
        self._active_tool_call_id = call_id if call_id is not None else f'chatcmpl-tool-{shortuuid.random()}'
        self._name_emitted = False
        self._output_index = self._active_tool_index if self._allowed_tool_names is None else _PENDING_INDEX
        self._first_delta = True
        self._pending_deltas.clear()

    def feed_tool_block(self, text: str, deltas: list[DeltaToolCall], *, final: bool) -> int:
        """Consume a buffered prefix of an open streamed tool block.

        ``text`` begins immediately after the outer opening marker and may
        include a previously retained marker prefix, newly decoded text, a
        complete closing marker, and trailing assistant content. Parsed tool
        call fragments are appended to ``deltas``. The caller must retain
        ``text[consumed:]`` and pass it back with the next chunk.

        The model-specific parser consumes only the inner payload. Once that
        payload is closed, later calls bypass it and this method consumes the
        configured outer closing marker before closing the complete block.

        Args:
            text: Buffered text following the tool opening marker.
            deltas: Output list to which parsed streaming deltas are appended.
            final: Whether this is the final generated stream chunk.

        Returns:
            Number of leading characters that the caller may safely discard.
        """
        if self.block_closed:
            return 0

        close_tag = self._close_tag
        consumed = 0
        if not self._payload_closed:
            consumed = self._consume_stream_payload(text, deltas, final=final)
        if self._payload_closed:
            if close_tag is None:
                self._close_tool_block()
                return consumed

            close_at = text.find(close_tag, consumed)
            if close_at >= 0:
                self._close_tool_block()
                return close_at + len(close_tag)
            if final:
                self._close_tool_block()
                return len(text)
            if not self.tool_close_prefixes:
                return len(text)
            return self._stable_prefix_end(text, self.tool_close_prefixes, consumed)

        if final:
            self._close_tool_block()
            return len(text)
        return consumed

    def _consume_stream_payload(self, text: str, deltas: list[DeltaToolCall], *, final: bool) -> int:
        raise NotImplementedError('ToolParser._consume_stream_payload has not been implemented!')

    def _close_tool_block(self) -> None:
        self.block_closed = True
        self._active_tool_call_id = ''
        self._pending_deltas.clear()

    def _emit_delta(
        self,
        deltas: list[DeltaToolCall],
        *,
        name: str | None = None,
        arguments: str | None = None,
    ) -> None:
        """Emit one fragment of the active call after tool-name filtering.

        Arguments seen before the function name are buffered so accepted calls
        retain source order. Rejected calls and their buffered arguments are
        discarded. ``_begin_call`` must be called before the first fragment of
        every logical tool call.
        """
        if name is None and arguments is None:
            return

        output_index = self._output_index
        hold_until_name = False
        if output_index < 0:
            if output_index == _REJECTED_INDEX:
                return
            if name is None:
                # The placeholder is rewritten if the later name is accepted.
                output_index = self._active_tool_index
                hold_until_name = True
            elif self._allowed_tool_names is None or name not in self._allowed_tool_names:
                self._pending_deltas.clear()
                self._output_index = _REJECTED_INDEX
                return
            else:
                output_index = self._next_output_index
                self._next_output_index += 1
                self._output_index = output_index

        if self._first_delta:
            self._first_delta = False
            call_id = self._active_tool_call_id
            call_type = 'function'
        else:
            call_id = call_type = None
        delta = DeltaToolCall(
            id=call_id,
            index=output_index,
            type=call_type,
            function=DeltaFunctionCall(name=name, arguments=arguments),
        )
        if hold_until_name:
            self._pending_deltas.append(delta)
            return

        if name is not None and self._pending_deltas:
            # Flush arguments before a late name to preserve generation order.
            for pending_delta in self._pending_deltas:
                pending_delta.index = output_index
            deltas.extend(self._pending_deltas)
            self._pending_deltas.clear()
        deltas.append(delta)

    @staticmethod
    def build_tool_calls(deltas: list[DeltaToolCall]) -> list[ToolCall]:
        """Build complete calls by grouping ordered deltas by output index.

        The first ID and name are retained, while every argument fragment is joined in emission order. An index without
        an emitted name does not produce a complete call.
        """
        tool_calls: list[ToolCall] = []

        parts_by_index: dict[int, _ToolCallParts] = {}
        for delta in deltas:
            parts = parts_by_index.get(delta.index)
            if parts is None:
                parts = _ToolCallParts()
                parts_by_index[delta.index] = parts
            if delta.id is not None and parts.call_id is None:
                parts.call_id = delta.id
            function = delta.function
            if function is None:
                continue
            if function.name is not None and parts.name is None:
                parts.name = function.name
            if function.arguments is not None:
                parts.arguments.append(function.arguments)

        for parts in parts_by_index.values():
            if parts.name is None:
                continue
            function = FunctionCall(name=parts.name, arguments=''.join(parts.arguments))
            if parts.call_id is None:
                tool_calls.append(ToolCall(function=function))
            else:
                tool_calls.append(ToolCall(id=parts.call_id, function=function))
        return tool_calls

    @staticmethod
    def _stable_prefix_end(text: str, marker_prefixes: tuple[str, ...], start: int = 0) -> int:
        """Find the consumable prefix before a token-aligned marker suffix.

        ``marker_prefixes`` contains the proper marker prefixes that can occur
        at real decoded-token boundaries, ordered longest first. The caller is
        expected to have established that no complete marker is present. This
        avoids testing every character boundary or tokenizing again in the hot
        path.

        Args:
            text: Buffered text ending in a possible marker prefix.
            marker_prefixes: Legal token-aligned proper prefixes of the marker.
            start: Earliest offset at which a retained suffix may begin.

        Returns:
            End offset of the stable prefix. ``text`` before this offset may be
            consumed; text at and after it must remain buffered.
        """
        for prefix in marker_prefixes:
            if text.endswith(prefix, start):
                return len(text) - len(prefix)
        return len(text)
