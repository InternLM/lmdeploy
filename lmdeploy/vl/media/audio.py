# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/media/audio.py

import base64
from io import BytesIO
from pathlib import Path

import numpy.typing as npt

from .base import MediaIO


class AudioMediaIO(MediaIO[tuple[npt.NDArray, float]]):

    def __init__(self, **kwargs) -> None:
        super().__init__()

        # lazy import to avoid dependency issues for users who don't use audio features
        try:
            import librosa
            self._librosa = librosa
        except ImportError:
            raise ImportError('Please install librosa via `pip install librosa`.')

        try:
            import soundfile
            self._soundfile = soundfile
        except ImportError:
            raise ImportError('Please install soundfile via `pip install soundfile`.')

        # for potential custom arguments from --media-io-kwargs
        self.kwargs = kwargs

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, float]:
        # sr = None, preserves the original sampling rate of the audio file
        return self._librosa.load(BytesIO(data), sr=None)

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, float]:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        return self._librosa.load(filepath, sr=None)

    def encode_base64(
        self,
        media: tuple[npt.NDArray, int],
        *,
        audio_format: str = 'WAV',
    ) -> str:
        audio, sr = media

        with BytesIO() as buffer:
            self._soundfile.write(buffer, audio, sr, format=audio_format)
            data = buffer.getvalue()

        return base64.b64encode(data).decode('utf-8')
