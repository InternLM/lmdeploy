# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Tuple

import numpy.typing as npt
from PIL import Image

from .media.connection import load_from_url
from .media.image import ImageMediaIO
from .media.time_series import TimeSeriesMediaIO
from .media.video import VideoMediaIO


def load_image(image_url: str, **kwargs) -> Image.Image:
    """Fetch and decode an image from a URL, path, or base64 string."""
    image_io = ImageMediaIO(**kwargs)
    return load_from_url(image_url, image_io)


def load_video(video_url: str, **kwargs) -> Tuple[npt.NDArray, Dict[str, Any]]:
    """Fetch and decode video frames from a URL, path, or base64 string."""
    image_io = ImageMediaIO()
    video_io = VideoMediaIO(image_io=image_io, **kwargs)
    return load_from_url(video_url, video_io)


def load_time_series(ts_url: str, **kwargs) -> npt.NDArray:
    """Fetch and decode time-series data from a URL or path or base64
    string.."""
    ts_io = TimeSeriesMediaIO(**kwargs)
    return load_from_url(ts_url, ts_io)


def encode_image_base64(image: str | Image.Image, format: str = 'PNG', **kwargs) -> str:
    """Encode an image path or PIL image to a base64 string."""
    if isinstance(image, str):
        image = load_image(image, **kwargs)
    image_io = ImageMediaIO(**kwargs)
    return image_io.encode_base64(image, image_format=format)


def encode_video_base64(video: str | npt.NDArray, format: str = 'JPEG', **kwargs) -> str:
    """Encode video (path or frames) to a base64 string."""
    if isinstance(video, str):
        video, _ = load_video(video, **kwargs)
    image_io = ImageMediaIO()
    video_io = VideoMediaIO(image_io=image_io, **kwargs)
    return video_io.encode_base64(video, video_format=format)


def encode_time_series_base64(data: str | npt.NDArray, **kwargs) -> str:
    """Encode time-series (path or numpy array) to a base64 string."""
    if isinstance(data, str):
        data = load_time_series(data, **kwargs)
    ts_io = TimeSeriesMediaIO(**kwargs)
    return ts_io.encode_base64(data)
