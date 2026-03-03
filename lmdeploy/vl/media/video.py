# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/media/video.py

import base64
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from lmdeploy.utils import get_logger

from .base import MediaIO
from .image import ImageMediaIO
from .video_loader import (DecordVideoLoader, OpenCVVideoLoader, TorchCodecVideoLoader, TorchVisionVideoLoader,
                           VideoLoader)

logger = get_logger('lmdeploy')


class VideoMediaIO(MediaIO[tuple[npt.NDArray, dict[str, Any]]]):

    def __init__(
        self,
        image_io: ImageMediaIO,
        num_frames: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()

        self.image_io = image_io
        self.num_frames = num_frames

        # for potential custom arguments from --media-io-kwargs
        self.kwargs = kwargs
        self.video_loader = self._get_video_loader_backend()

    def _get_video_loader_backend(self) -> VideoLoader:
        """Determines the best available video loader backend."""
        # vLLM:          OpenCV
        # SGLang:        Decord
        # qwen-vl-utils: TorchCodec -> Decord -> TorchVision (deprecated soon)
        backends = [
            ('cv2', OpenCVVideoLoader),
            ('decord', DecordVideoLoader),
            ('torchcodec', TorchCodecVideoLoader),
            ('torchvision', TorchVisionVideoLoader),
        ]

        for module_name, loader_cls in backends:
            try:
                __import__(module_name)
                return loader_cls()
            except (ImportError, RuntimeError):
                logger.warning(f"Video backend '{module_name}' not found. Trying next backend...")
                continue

        raise ImportError(
            'No video backend found. Install either opencv-python-headless, decord, torchcodec, or torchvision.')

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, dict[str, Any]]:
        return self.video_loader.load_bytes(data, num_frames=self.num_frames, **self.kwargs)

    def load_base64(self, media_type: str, data: str) -> tuple[npt.NDArray, dict[str, Any]]:
        if media_type.lower() == 'video/jpeg':
            load_frame = partial(
                self.image_io.load_base64,
                'image/jpeg',
            )

            return np.stack([np.asarray(load_frame(frame_data)) for frame_data in data.split(',')]), {}

        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, dict[str, Any]]:
        return self.video_loader.load_file(filepath)

    def encode_base64(
        self,
        media: npt.NDArray,
        *,
        video_format: str = 'JPEG',
    ) -> str:
        video = media

        if video_format == 'JPEG':
            encode_frame = partial(
                self.image_io.encode_base64,
                image_format=video_format,
            )

            return ','.join(encode_frame(Image.fromarray(frame)) for frame in video)

        msg = 'Only JPEG format is supported for now.'
        raise NotImplementedError(msg)
