# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/media/video.py

import base64
import os
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from lmdeploy.utils import get_logger

from .base import MediaIO
from .image import ImageMediaIO
from .video_loader import (
    DecordVideoLoader,
    OpenCVVideoLoader,
    TorchCodecVideoLoader,
    TorchVisionVideoLoader,
    VideoLoader,
)

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
        """Determines the best available video loader backend. Override with
        LMDEPLOY_VIDEO_BACKEND={cv2,decord,torchcodec,torchvision}.

        vLLM:          OpenCV
        SGLang:        Decord
        qwen-vl-utils: TorchCodec -> Decord -> TorchVision (deprecated soon)
        """

        backends = [
            ('cv2', OpenCVVideoLoader),
            ('decord', DecordVideoLoader),
            ('torchcodec', TorchCodecVideoLoader),
            ('torchvision', TorchVisionVideoLoader),
        ]
        backend_map = dict(backends)

        preferred = os.getenv('LMDEPLOY_VIDEO_BACKEND', None)
        if preferred is not None:
            if preferred not in backend_map:
                valid = list(backend_map.keys())
                raise ValueError(
                    f"LMDEPLOY_VIDEO_BACKEND='{preferred}' is not valid. "
                    f'Choose from: {valid}')
            try:
                __import__(preferred)
                return backend_map[preferred]()
            except (ImportError, RuntimeError) as e:
                raise ImportError(
                    f"Video backend '{preferred}' (set via LMDEPLOY_VIDEO_BACKEND) "
                    f'is not available: {e}') from e

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

            # NOTE: known issue in https://github.com/QwenLM/Qwen3-VL/issues/1643
            # when passing a video as a sequence of JPEG frames, we cannot obtain the video metadata
            # therefore we construct a default metadata dictionary with common values.
            frames = np.stack([np.asarray(load_frame(frame_data)) for frame_data in data.split(',')])

            total_frames_num = int(frames.shape[0])
            fps = float(self.kwargs.get('fps', 2))  # default to 2 fps if not specified
            duration = (total_frames_num / fps) if fps > 0 else 0
            frame_idx = list(range(total_frames_num))

            metadata = {
                'total_num_frames': total_frames_num,
                'fps': fps,
                'duration': duration,
                'video_backend': 'jpeg_sequence',
                'frames_indices': frame_idx,
            }

            logger.info('Loading video from base64-encoded JPEG frames misses video metadata.'
                        f'Fall back to default metadata values:\n{metadata}')
            return frames, metadata

        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, dict[str, Any]]:
        return self.video_loader.load_file(filepath, num_frames=self.num_frames, **self.kwargs)

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
