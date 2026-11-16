# Copyright (c) OpenMMLab. All rights reserved.
"""Per-encoder image preprocessing cache, accessed by its single worker."""
import hashlib
import resource
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from lmdeploy.pytorch.multimodal.data_type import make_multimodal_content_hash
from lmdeploy.utils import get_logger
from lmdeploy.vl.media.image import EncodedImage

logger = get_logger('lmdeploy')


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
    """Own an LRU's tensor references, storage byte budget and entry limit.

    With shared storage, each entry retains one pixel storage FD. Keep at most 256 entries and use at most a quarter of
    the process FD limit, leaving room for RPC and other open files. Eviction drops the cache's reference. In-flight
    requests retain their own references until terminal cleanup, so eviction cannot invalidate them.
    """

    def __init__(self, max_bytes: int = 0, *, share_memory: bool = False):
        self.max_bytes = max_bytes
        self.share_memory = share_memory
        self.current_bytes = 0
        self._max_entries = 256
        fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        if fd_limit != resource.RLIM_INFINITY:
            self._max_entries = min(self._max_entries, fd_limit // 4)
        self._entries: OrderedDict[tuple, ProcessedImage] = OrderedDict()

    def process(self, images: list, preprocess: Callable[[Any, dict | None], ProcessedImage],
                dtype: torch.dtype | None, size: dict | None) -> list[ProcessedImage]:
        """Reuse image features, preparing cold storage for the selected
        transport.

        ``preprocess`` handles one decoded image and its size override, including
        any model-specific content hash. Only FD-capable transports opt into
        shared storage; cache reuse does not depend on that capability.
        """
        cache_enabled = self.max_bytes > 0 and self._max_entries > 0
        share_pixels = self.share_memory and torch.multiprocessing.get_sharing_strategy() == 'file_descriptor'
        processed = []
        for image in images:
            key = self.key(image, dtype, size) if cache_enabled else None
            item = self.get(key) if key is not None else None
            if item is None:
                decoded = image.decode() if isinstance(image, EncodedImage) else image
                item = preprocess(decoded, size)
                # Release cold temporaries before processing the next image.
                del decoded
                if share_pixels:
                    try:
                        item.pixel_values.share_memory_()
                    except RuntimeError as error:
                        # PyTorch can report "Success (0)" for a failed shm allocation.
                        if 'unable to allocate shared memory(shm)' not in str(error):
                            raise
                        share_pixels = False
                        logger.warning('Image shared-memory allocation failed (%s); using ordinary CPU tensors for '
                                       'the remaining images in this request. Increase /dev/shm to retain '
                                       'shared-memory IPC.', error)
                if key is not None:
                    self.put(key, item)
            processed.append(item)
        return processed

    @staticmethod
    def key(image, dtype: torch.dtype | None, size: dict | None) -> tuple:
        if isinstance(image, EncodedImage):
            content = hashlib.sha256(image.data).hexdigest()
            identity = ('encoded', image.mode, content)
        elif isinstance(image, Image.Image):
            # Palette indices alone omit colors and transparency.
            pixels = image.convert('RGBA') if image.mode in ('P', 'PA') else image
            content = hashlib.sha256(pixels.tobytes()).hexdigest()
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
        if self._max_entries == 0 or item.nbytes > self.max_bytes:
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.current_bytes -= previous.nbytes
        while self._entries and (len(self._entries) >= self._max_entries
                                 or self.current_bytes + item.nbytes > self.max_bytes):
            _, evicted = self._entries.popitem(last=False)
            self.current_bytes -= evicted.nbytes
        self._entries[key] = item
        self.current_bytes += item.nbytes
