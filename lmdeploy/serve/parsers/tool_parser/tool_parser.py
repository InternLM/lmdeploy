# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import shortuuid
from mmengine import Registry

from lmdeploy.serve.openai.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
    ToolCall,
)

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest, Tool

ToolParserManager = Registry('tool_parser', locations=['lmdeploy.serve.parsers.tool_parser'])


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
        self._stream_close_tag: str | None = None
        self._active_tool_call_id: str = ''
        self._active_tool_index: int = -1
        self._name_emitted: bool = False
        self._emitted_tool_indices: set[int] = set()
        self._allowed_tool_names: set[str] | None = None
        self._stream_internal_tool_index: int = -1
        self._stream_accepted_tool_index: int = -1
        self._stream_visible_tool_index: int = -1
        self._stream_tool_name_seen: bool = False
        self._pending_tool_deltas: list[DeltaToolCall] = []
        self._next_stream_tool_index: int = 0
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

    def filter_tool_calls(self, calls: list[ToolCall]) -> list[ToolCall]:
        """Keep complete calls whose names occur in the request tools."""
        if self._allowed_tool_names is None:
            return calls
        return [call for call in calls if call.function.name in self._allowed_tool_names]

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        """Return tool opening tag string, or None if unsupported."""
        raise NotImplementedError('ToolParser.get_tool_open_tag has not been implemented!')

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        """Return tool closing tag string, or None if unsupported."""
        raise NotImplementedError('ToolParser.get_tool_close_tag has not been implemented!')

    def begin_tool_block(self) -> None:
        """Initialize one streamed tool block after its opening marker."""
        self._stream_close_tag = self.get_tool_close_tag()
        self._active_tool_index += 1
        self._active_tool_call_id = f'chatcmpl-tool-{shortuuid.random()}'
        self._name_emitted = False
        self._payload_closed = False
        self.block_closed = False

    def feed_tool_block(self, text: str, deltas: list[DeltaToolCall], *, final: bool) -> int:
        """Consume a buffered prefix of an open streamed tool block.

        ``text`` begins immediately after the outer opening marker and may
        include a previously retained marker prefix, newly decoded text, a
        complete closing marker, and trailing assistant content. Parsed tool
        call fragments are appended to ``deltas``. The caller must retain
        ``text[consumed:]`` and pass it back with the next chunk.

        The model-specific parser first consumes its inner payload. Once that
        payload is closed, this method consumes the configured outer closing
        marker before closing the complete block.

        Args:
            text: Buffered text following the tool opening marker.
            deltas: Output list to which parsed streaming deltas are appended.
            final: Whether this is the final generated stream chunk.

        Returns:
            Number of leading characters that the caller may safely discard.
        """
        if self.block_closed:
            return 0

        close_tag = self._stream_close_tag
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
        self._pending_tool_deltas.clear()

    def _emit_delta(
        self,
        deltas: list[DeltaToolCall],
        *,
        index: int,
        tool_call_id: str,
        name: str | None = None,
        arguments: str | None = None,
    ) -> None:
        """Append one source-order delta after request tool-name filtering."""
        if name is None and arguments is None:
            return

        output_index = index
        hold_until_name = False
        if name is None and index == self._stream_accepted_tool_index:
            output_index = self._stream_visible_tool_index
        elif self._allowed_tool_names is not None:
            if index != self._stream_internal_tool_index:
                self._pending_tool_deltas.clear()
                self._stream_internal_tool_index = index
                self._stream_visible_tool_index = -1
                self._stream_tool_name_seen = False

            if name is not None:
                self._stream_tool_name_seen = True
                if name not in self._allowed_tool_names:
                    self._pending_tool_deltas.clear()
                    return
                output_index = self._next_stream_tool_index
                self._next_stream_tool_index += 1
                self._stream_accepted_tool_index = index
                self._stream_visible_tool_index = output_index
            elif self._stream_tool_name_seen:
                if self._stream_visible_tool_index < 0:
                    return
                output_index = self._stream_visible_tool_index
            else:
                hold_until_name = True

        first = index not in self._emitted_tool_indices
        if first:
            self._emitted_tool_indices.add(index)
        delta = DeltaToolCall(
            id=tool_call_id if first else None,
            index=output_index,
            type='function' if first else None,
            function=DeltaFunctionCall(name=name, arguments=arguments),
        )
        if hold_until_name:
            self._pending_tool_deltas.append(delta)
            return

        if name is not None and self._pending_tool_deltas:
            for pending in self._pending_tool_deltas:
                pending.index = output_index
            deltas.extend(self._pending_tool_deltas)
            self._pending_tool_deltas.clear()
        deltas.append(delta)

    def parse_tool_block(self, text: str, start: int, tool_calls: list[ToolCall]) -> int:
        """Parse one complete, non-streamed tool block.

        Args:
            text: Complete generated response containing the tool block.
            start: Index of the first payload character after its opening
                marker.
            tool_calls: Output list to which parsed calls are appended.

        Returns:
            Absolute index of the first character after the consumed block. If
            the configured closing marker is absent, returns ``len(text)``
            without appending a call.
        """
        close_tag = self.get_tool_close_tag()
        if close_tag is None:
            end = len(text)
        else:
            end = text.find(close_tag, start)
            if end < 0:
                return len(text)

        parsed = self.parse_tool_call_complete(text[start:end])
        if isinstance(parsed, list):
            tool_calls.extend(parsed)
        elif parsed is not None:
            tool_calls.append(parsed)
        return end + (len(close_tag) if close_tag is not None else 0)

    def parse_tool_call_complete(self, payload: str) -> ToolCall | list[ToolCall] | None:
        """Parse one complete tool payload into OpenAI tool call objects."""
        raise NotImplementedError('ToolParser.parse_tool_call_complete has not been implemented!')

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
