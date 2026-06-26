# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/media/base.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

_T = TypeVar('_T')


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
