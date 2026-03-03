# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/media/image.py

from io import BytesIO
from pathlib import Path

import pybase64
from PIL import Image

from .base import MediaIO


class ImageMediaIO(MediaIO[Image.Image]):

    def __init__(self, image_mode: str = 'RGB', **kwargs) -> None:
        super().__init__()
        self.image_mode = image_mode

        # for potential custom arguments from --media-io-kwargs
        self.kwargs = kwargs

    def load_bytes(self, data: bytes) -> Image.Image:
        image = Image.open(BytesIO(data))
        return image.convert(self.image_mode)

    def load_base64(self, media_type: str, data: str) -> Image.Image:
        return self.load_bytes(pybase64.b64decode(data))

    def load_file(self, file_path: Path) -> Image.Image:
        with open(file_path, 'rb') as f:
            data = f.read()
        image = Image.open(BytesIO(data))
        return image.convert(self.image_mode)

    def encode_base64(self, image: Image.Image, image_format: str = 'PNG') -> str:
        with BytesIO() as buffer:
            image = image.convert(self.image_mode)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return pybase64.b64encode(data).decode('utf-8')
