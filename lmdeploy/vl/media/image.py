# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/media/image.py

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pybase64
from PIL import Image, ImageFile

from .base import MediaIO

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class EncodedImage:
    """Compressed image retained until a preprocessing cache miss."""

    data: bytes
    mode: str

    def decode(self) -> Image.Image:
        with Image.open(BytesIO(self.data)) as image:
            return image.convert(self.mode)


class ImageMediaIO(MediaIO[Image.Image | EncodedImage]):

    def __init__(self, image_mode: str = 'RGB', defer_decode: bool = False, **kwargs) -> None:
        super().__init__()
        self.image_mode = image_mode
        self.defer_decode = defer_decode

        # for potential custom arguments from --media-io-kwargs
        self.kwargs = kwargs

    def load_bytes(self, data: bytes) -> Image.Image | EncodedImage:
        if self.defer_decode:
            return EncodedImage(data, self.image_mode)
        image = Image.open(BytesIO(data))
        return image.convert(self.image_mode)

    def load_base64(self, media_type: str, data: str) -> Image.Image | EncodedImage:
        return self.load_bytes(pybase64.b64decode(data))

    def load_file(self, file_path: Path) -> Image.Image | EncodedImage:
        with open(file_path, 'rb') as f:
            data = f.read()
        return self.load_bytes(data)

    def encode_base64(self, image: Image.Image, image_format: str = 'PNG') -> str:
        with BytesIO() as buffer:
            image = image.convert(self.image_mode)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return pybase64.b64encode(data).decode('utf-8')
