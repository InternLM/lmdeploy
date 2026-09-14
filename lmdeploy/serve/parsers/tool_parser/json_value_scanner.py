# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import re

_JSON_CONTAINER_SPECIAL = re.compile(r'["{}\[\]]')


class JsonValueScanner:
    """Find the lexical boundary of one streamed JSON value.

    This deliberately does not validate JSON grammar. It only tracks strings, escapes, and container depth so callers
    can forward source bytes without accumulating or reserializing them.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset lexical state for a new value."""
        self.complete = False
        self.started = False
        self._mode: str | None = None
        self._depth = 0
        self._in_string = False
        self._escape_next = False

    def feed(self, text: str, start: int = 0, end: int | None = None) -> int:
        """Return the end of the prefix belonging to this value."""
        if self.complete:
            return start

        pos = start
        size = len(text) if end is None else end
        if not self.started:
            while pos < size and text[pos].isspace():
                pos += 1
            if pos == size:
                return pos

            self.started = True
            char = text[pos]
            if char == '"':
                self._mode = 'string'
                self._in_string = True
                pos += 1
            elif char in '{[':
                self._mode = 'container'
                self._depth = 1
                pos += 1
            else:
                self._mode = 'scalar'

        if self._mode == 'scalar':
            while pos < size:
                char = text[pos]
                if char.isspace() or char in ',}]':
                    self.complete = True
                    break
                pos += 1
            return pos

        # Tiny streaming chunks are common. Scanning them directly is cheaper
        # than entering the find/regex fast path below.
        if size - pos <= 4:
            while pos < size:
                char = text[pos]
                pos += 1
                if self._escape_next:
                    self._escape_next = False
                elif self._in_string:
                    if char == '\\':
                        self._escape_next = True
                    elif char == '"':
                        self._in_string = False
                        if self._mode == 'string':
                            self.complete = True
                            break
                elif char == '"':
                    self._in_string = True
                elif char in '{[':
                    self._depth += 1
                elif char in '}]':
                    self._depth -= 1
                    if self._depth == 0:
                        self.complete = True
                        break
            return pos

        while pos < size:
            if self._escape_next:
                self._escape_next = False
                pos += 1
                continue

            if self._in_string:
                quote_at = text.find('"', pos, size)
                if quote_at < 0:
                    # Only an odd trailing backslash run affects the next
                    # chunk; all earlier escapes are already self-contained.
                    slash_at = size
                    while slash_at > pos and text[slash_at - 1] == '\\':
                        slash_at -= 1
                    self._escape_next = (size - slash_at) % 2 == 1
                    return size

                # A quote is escaped exactly when its immediately preceding
                # backslash run has odd length. Each run is inspected once.
                slash_at = quote_at
                while slash_at > pos and text[slash_at - 1] == '\\':
                    slash_at -= 1
                pos = quote_at + 1
                if (quote_at - slash_at) % 2 == 1:
                    continue

                self._in_string = False
                if self._mode == 'string':
                    self.complete = True
                    break
                continue

            # Search all container syntax in one pass. Separate ``str.find``
            # calls would repeatedly scan the same suffix when values contain
            # many short strings.
            match = _JSON_CONTAINER_SPECIAL.search(text, pos, size)
            if match is None:
                return size

            char = match.group()
            pos = match.end()
            if char == '"':
                self._in_string = True
            elif char in '{[':
                self._depth += 1
            else:
                self._depth -= 1
                if self._depth == 0:
                    self.complete = True
                    break
        return pos

    def finish(self) -> None:
        """Finish a root scalar at an externally known boundary."""
        if self.started and self._mode == 'scalar':
            self.complete = True

    def finish_scalar(self) -> bool:
        """Finish a root scalar at a caller-owned protocol marker."""
        self.finish()
        return self.complete

    @property
    def in_string(self) -> bool:
        """Whether a protocol-looking marker is currently string data."""
        return self._in_string
