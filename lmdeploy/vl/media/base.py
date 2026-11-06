# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/media/base.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

_T = TypeVar('_T')


class MediaInputError(Exception):
    """Failed to read media that the client supplied by path.

    Raised only at the boundary that opens a user-supplied path directly, with a message built from that path and a
    generic errno description, so the message is safe to return to the client. Server-internal I/O errors (e.g.
    temporary files staged for downloaded media) must never be wrapped into this exception.
    """


class MediaIO(ABC, Generic[_T]):

    @abstractmethod
    def load_bytes(self, data: bytes) -> _T:
        raise NotImplementedError

    @abstractmethod
    def load_base64(self, media_type: str, data: str) -> _T:
        raise NotImplementedError

    @abstractmethod
    def load_file(self, filepath: Path) -> _T:
        raise NotImplementedError
