# Copyright (c) OpenMMLab. All rights reserved.
"""Per-encoder image preprocessing cache, accessed by its single worker."""
import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import torch
from PIL import Image

from lmdeploy.pytorch.multimodal.data_type import make_multimodal_content_hash
from lmdeploy.vl.media.image import EncodedImage


@dataclass(frozen=True)
class ProcessedImage:
    """Immutable image features; prompt offsets belong to each request."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    content_hash: str

    @property
    def nbytes(self) -> int:
        return (self.pixel_values.untyped_storage().nbytes()
                + self.image_grid_thw.untyped_storage().nbytes())


class ImageProcessorCache:
    """Own an LRU's tensor references and actual storage byte budget.

    Eviction drops the cache's reference. In-flight requests retain their own references until terminal cleanup, so
    eviction cannot invalidate them.
    """

    def __init__(self, max_bytes: int = 0):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self._entries: OrderedDict[tuple, ProcessedImage] = OrderedDict()

    @staticmethod
    def key(image, dtype: torch.dtype | None, size: dict | None) -> tuple:
        if isinstance(image, EncodedImage):
            content = hashlib.sha256(image.data).hexdigest()
            identity = ('encoded', image.mode, content)
        elif isinstance(image, Image.Image):
            content = hashlib.sha256(image.tobytes()).hexdigest()
            identity = ('pixels', image.mode, image.size, content)
        else:
            # Offline pipelines can also supply NumPy arrays or CPU tensors.
            identity = ('array', make_multimodal_content_hash(image, None))
        return identity, dtype, tuple(sorted(size.items())) if size else ()

    def get(self, key: tuple) -> ProcessedImage | None:
        item = self._entries.get(key)
        if item is not None:
            self._entries.move_to_end(key)
        return item

    def put(self, key: tuple, item: ProcessedImage) -> None:
        if item.nbytes > self.max_bytes:
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.current_bytes -= previous.nbytes
        while self._entries and self.current_bytes + item.nbytes > self.max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self.current_bytes -= evicted.nbytes
        self._entries[key] = item
        self.current_bytes += item.nbytes
