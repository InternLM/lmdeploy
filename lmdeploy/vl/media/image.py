# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/media/image.py

from contextlib import nullcontext
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from struct import error as StructError
from threading import Lock

import pybase64
from PIL import Image, ImageFile, ImageOps

from lmdeploy.utils import get_logger

from .base import MediaIO

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = get_logger('lmdeploy')
_shared_pil_lock = Lock()


def apply_exif_orientation(image: Image.Image, *, shared: bool = True) -> Image.Image:
    """Normalize orientation without mutating the caller's pixels or
    metadata."""
    # Caller-owned PIL images can share a lazy decoder across messages/requests.
    with _shared_pil_lock if shared else nullcontext():
        image.load()  # Pixel decoding failures must not be mistaken for invalid metadata.
        try:
            # JPEG opening may cache an empty EXIF after suppressing a parse error.
            # Validate the raw payload too, before convert/copy makes it parse again.
            if raw_exif := image.info.get('exif'):
                Image.Exif().load(raw_exif)
            if image.getexif().get(274, 1) not in range(2, 9):
                return image
            return ImageOps.exif_transpose(image)
        except (SyntaxError, ValueError, TypeError, OSError, StructError):
            logger.warning('Ignoring invalid image orientation metadata.')
            image = image.copy()
            for key in ('exif', 'Raw profile type exif', 'XML:com.adobe.xmp', 'xmp'):
                image.info.pop(key, None)
            return image


@dataclass(frozen=True)
class EncodedImage:
    """Compressed image retained until a preprocessing cache miss."""

    data: bytes
    mode: str

    def decode(self) -> Image.Image:
        with Image.open(BytesIO(self.data)) as image:
            return apply_exif_orientation(image, shared=False).convert(self.mode)


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
        return apply_exif_orientation(image, shared=False).convert(self.image_mode)

    def load_base64(self, media_type: str, data: str) -> Image.Image | EncodedImage:
        return self.load_bytes(pybase64.b64decode(data))

    def load_file(self, file_path: Path) -> Image.Image | EncodedImage:
        with open(file_path, 'rb') as f:
            data = f.read()
        return self.load_bytes(data)

    def encode_base64(self, image: Image.Image, image_format: str = 'PNG') -> str:
        with BytesIO() as buffer:
            image = apply_exif_orientation(image).convert(self.image_mode)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return pybase64.b64encode(data).decode('utf-8')
