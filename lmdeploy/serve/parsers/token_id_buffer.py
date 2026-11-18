# Copyright (c) OpenMMLab. All rights reserved.
"""Explicit text and token-derived boundaries used by the response parser."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class ControlSpan:
    start: int
    end: int
    tag: str


@dataclass(frozen=True)
class LocatedText:
    text: str
    # None is legacy text; () means no declared control IDs were found.
    spans: tuple[ControlSpan, ...] | None = None


class BoundaryIndex:
    """Index once per append; all consumers share it through relative views."""

    def __init__(self, text: str, spans: tuple[ControlSpan, ...]):
        previous_end = 0
        by_tag: dict[str, list[int]] = {}
        for span in spans:
            if (span.start < previous_end or span.end > len(text)
                    or span.end != span.start + len(span.tag) or not span.tag
                    or not text.startswith(span.tag, span.start)):
                raise ValueError('Invalid control-token boundary')
            previous_end = span.end
            by_tag.setdefault(span.tag, []).append(span.start)
        self.spans = spans
        self.starts = tuple(span.start for span in spans)
        self.by_tag = {tag: tuple(starts) for tag, starts in by_tag.items()}

    def check_cut(self, offset: int) -> None:
        i = bisect_right(self.starts, offset) - 1
        if i >= 0 and self.spans[i].start < offset < self.spans[i].end:
            raise ValueError('Cannot consume part of a control token')


@dataclass(frozen=True)
class BoundaryView:
    index: BoundaryIndex
    origin: int
    end: int
    token_id_terminals: frozenset[str]

    def find(self, tag: str, start: int = 0) -> int:
        starts = self.index.by_tag.get(tag, ())
        i = bisect_left(starts, self.origin + max(start, 0))
        if i < len(starts) and starts[i] + len(tag) <= self.end:
            return starts[i] - self.origin
        return -1


class ParseBuffer:
    """A consumable string plus optional boundaries, never a str subclass."""

    def __init__(self, *, token_id_terminals: tuple[str, ...] = ()):
        self.token_id_terminals = frozenset(token_id_terminals)
        self._text = ''
        self._offset = 0
        self._index = BoundaryIndex('', ()) if self.token_id_terminals else None

    def __len__(self) -> int:
        return len(self._text) - self._offset

    @property
    def strict(self) -> bool:
        return self._index is not None

    @property
    def text(self) -> str:
        return self._text[self._offset:]

    @property
    def boundaries(self) -> BoundaryView | None:
        if self._index is None:
            return None
        return BoundaryView(self._index, self._offset, len(self._text), self.token_id_terminals)

    def append(self, located: LocatedText) -> None:
        if self.strict != (located.spans is not None):
            raise ValueError('Cannot mix legacy text and token-derived boundaries')
        if not located.text and not located.spans:
            return
        text = self.text
        # An empty index remains valid when appending ordinary text.
        if self._index is not None and (self._index.spans or located.spans):
            assert located.spans is not None
            first = bisect_left(self._index.starts, self._offset)
            spans = tuple(ControlSpan(s.start - self._offset, s.end - self._offset, s.tag)
                          for s in self._index.spans[first:])
            spans += tuple(ControlSpan(s.start + len(text), s.end + len(text), s.tag) for s in located.spans)
            self._index = BoundaryIndex(text + located.text, spans)
        self._text = text + located.text
        self._offset = 0

    def take(self, size: int) -> str:
        if not 0 <= size <= len(self):
            raise ValueError('Invalid parser consumption length')
        end = self._offset + size
        if self._index is not None and self._index.spans:
            self._index.check_cut(end)
        result = self._text[self._offset:end]
        self._offset = end
        if end == len(self._text):
            self._text = ''
            self._offset = 0
            if self._index is not None and self._index.spans:
                self._index = BoundaryIndex('', ())
        return result

    def find_structure(self, tag: str, start: int = 0) -> int:
        if tag not in self.token_id_terminals:
            return self.text.find(tag, start)
        assert self._index is not None
        if not self._index.spans:
            return -1
        return BoundaryView(self._index, self._offset, len(self._text), self.token_id_terminals).find(tag, start)

    def startswith_structure(self, tag: str, start: int = 0) -> bool:
        return self.find_structure(tag, start) == start
