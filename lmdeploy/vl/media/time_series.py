# Copyright (c) OpenMMLab. All rights reserved.
from io import BytesIO
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pybase64

from lmdeploy.utils import get_logger

from .base import MediaIO

logger = get_logger('lmdeploy')


class TimeSeriesMediaIO(MediaIO[npt.NDArray]):

    def __init__(self, **kwargs):
        super().__init__()

        # for potential custom arguments from --media-io-kwargs
        self.kwargs = kwargs

    def load_bytes(self, data: bytes) -> npt.NDArray:
        ts_array = np.load(BytesIO(data), allow_pickle=False)
        return ts_array

    def load_base64(self, media_type: str, data: str) -> npt.NDArray:
        return self.load_bytes(pybase64.b64decode(data))

    def load_file(self, filepath: Path) -> npt.NDArray:
        suffix = filepath.suffix.lower()

        if suffix == '.npy':
            return np.load(filepath)
        elif suffix == '.csv':
            try:
                ts_array = np.genfromtxt(filepath, delimiter=',', dtype=np.float32)
                if ts_array.size == 0:
                    raise ValueError(f'CSV file {filepath} yielded no data.')
                return ts_array
            except Exception as e:
                logger.error(f'Failed to load CSV {filepath}: {e}')
                raise
        elif suffix in ['.wav', '.mp3', '.flac']:
            try:
                import soundfile as sf
            except ImportError:
                raise ImportError('Please install soundfile via `pip install soundfile`.')

            ts_array, _ = sf.read(filepath)
            return ts_array

        raise ValueError(f'Unsupported file format: {suffix}')

    def encode_base64(self, data: npt.NDArray) -> str:
        """Encode numpy array to base64 string using NPY format."""
        buffer = BytesIO()
        np.save(buffer, data, allow_pickle=False)
        return pybase64.b64encode(buffer.getvalue()).decode('utf-8')
