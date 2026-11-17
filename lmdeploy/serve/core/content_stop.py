# Copyright (c) OpenMMLab. All rights reserved.
"""Request-local stop strings for parsed assistant content."""

from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.serve.openai.protocol import DeltaMessage


class BufferedEngineStream:
    """Peek at backend tokens without committing them to the response or
    metrics."""

    def __init__(self, source: AsyncIterator[EngineOutput]):
        self.source = aiter(source)
        self.buffer: deque[EngineOutput] = deque()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.buffer:
            return self.buffer.popleft()
        return await anext(self.source)

    async def following_ids(self, batch: EngineOutput, offset: int, stop_ids: list[int]) -> AsyncIterator[int | None]:
        """Yield visible IDs after offset, preserving batch boundaries for
        normal consumption."""
        buffered = iter(tuple(self.buffer))
        while True:
            if batch.status not in (ResponseType.SUCCESS, ResponseType.FINISH):
                yield None  # Leave an aborted or failed stream to normal processing.
                return
            native_stop = bool(batch.token_ids and batch.token_ids[-1] in stop_ids)
            end = len(batch.token_ids) - native_stop
            for index in range(offset, end):
                yield batch.token_ids[index]
            if native_stop or batch.status == ResponseType.FINISH:
                return
            batch = next(buffered, None)
            if batch is None:
                batch = await anext(self.source, None)
                if batch is None:
                    return
                self.buffer.append(batch)
            offset = 0


class ChatContentStop:
    """Match request stop strings across content deltas, after channel
    parsing."""

    def __init__(self, words: list[str], include_stop: bool,
                 preview_text: Callable[[list[int]], str | None] | None = None,
                 resolve_separator: Callable[[AsyncIterator[int | None]], Awaitable[None]] | None = None):
        self.resolve_separator = resolve_separator if any(word.endswith('\n') for word in words) else None
        self.preview_text = preview_text
        self.words = words
        self.include_stop = include_stop
        self.pending = ''
        self.matched: str | None = None

    def can_skip_token_steps(self, ids: list[int]) -> bool:
        """Keep this engine batch intact only if no stop can complete in it."""
        if self.preview_text is None:
            return False
        text = self.preview_text(ids)
        if text is None:
            return False
        candidate = self.pending + text
        return not any(word in candidate for word in self.words)

    def _consume(self, content: str, can_stop: bool) -> str:
        pending_len = len(self.pending)
        text = self.pending + content
        self.pending = ''
        best_end = len(text) + 1
        if can_stop:
            for word in self.words:
                # A match must end in this delta. A shorter word in pending may
                # have completed before min_new_tokens while a longer one was open.
                pos = text.find(word, max(0, pending_len - len(word) + 1))
                if pos >= 0 and pos + len(word) < best_end:
                    self.matched = word
                    best_end = pos + len(word)
        if self.matched is not None:
            end = best_end if self.include_stop else best_end - len(self.matched)
            return text[:end]

        # Only hold back a suffix that can still complete a stop string.
        keep = 0
        for word in self.words:
            pos = text.find(word[0], max(0, len(text) - len(word) + 1))
            while pos >= 0 and len(text) - pos > keep:
                if word.startswith(text[pos:]):
                    keep = len(text) - pos
                    break
                pos = text.find(word[0], pos + 1)
        if keep:
            self.pending = text[-keep:]
            return text[:-keep]
        return text

    def filter_deltas(self,
                      deltas: list[tuple[DeltaMessage, bool]],
                      final: bool,
                      *,
                      can_stop: bool = True) -> list[tuple[DeltaMessage, bool]]:
        if self.matched is not None:
            return []
        result = []
        for message, tool_emitted in deltas:
            if self.pending and (message.reasoning_content or message.tool_calls):
                result.append((DeltaMessage(content=self.pending), False))
                self.pending = ''
            if message.content:
                content = self._consume(message.content, can_stop)
                message = message.model_copy(update={'content': content or None})
                if content or message.reasoning_content or message.tool_calls:
                    result.append((message, tool_emitted))
                if self.matched is not None:
                    break
            else:
                result.append((message, tool_emitted))
        if final and self.pending:
            result.append((DeltaMessage(content=self.pending), False))
            self.pending = ''
        return result
