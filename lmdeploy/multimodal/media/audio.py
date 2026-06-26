# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/media/audio.py

import base64
from io import BytesIO
from pathlib import Path

import numpy.typing as npt

from .base import MediaIO


class AudioMediaIO(MediaIO[tuple[npt.NDArray, int]]):

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

        # Qwen3-Omni's feature extractor expects 16 kHz audio; allow explicit
        # media-io overrides but resample to that rate by default.
        self.sampling_rate = kwargs.get('sampling_rate', kwargs.get('sample_rate', 16000))

        # for potential custom arguments from --media-io-kwargs
        self.kwargs = kwargs

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, int]:
        return self._librosa.load(BytesIO(data), sr=self.sampling_rate)

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, int]:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, int]:
        return self._librosa.load(filepath, sr=self.sampling_rate)

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
