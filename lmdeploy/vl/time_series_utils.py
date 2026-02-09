# Copyright (c) OpenMMLab. All rights reserved.
import os
from io import BytesIO

import numpy as np
import pybase64
import requests

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

FETCH_TIMEOUT = int(os.environ.get('LMDEPLOY_FETCH_TIMEOUT', 10))
HEADERS = {
    'User-Agent':
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}


def encode_time_series_base64(data: str | np.ndarray) -> str:
    """Encode time series data to base64.

    Supports: HTTP URL, local path, or numpy array.
    """
    buffered = BytesIO()

    try:
        if isinstance(data, str):
            if data.startswith('http'):
                response = requests.get(data, headers=HEADERS, timeout=FETCH_TIMEOUT)
                response.raise_for_status()
                ts_array = _load_bytes(response.content, data)
            elif data.startswith('file://'):
                path = data.removeprefix('file://')
                ts_array = _load_path(path)
            elif os.path.exists(data):
                ts_array = _load_path(data)
            else:
                raise ValueError(f'Path does not exist: {data}')
        elif isinstance(data, np.ndarray):
            ts_array = data
        else:
            raise TypeError(f'Expected str or np.ndarray, got {type(data)}')

        np.save(buffered, ts_array)

    except Exception as error:
        data_info = str(data)[:100] + ' ...' if isinstance(data, str) and len(data) > 100 else str(data)
        logger.error(f'{error}, data={data_info}')
        np.save(buffered, np.zeros((6000, 3), dtype=np.float32))  # dummy

    return pybase64.b64encode(buffered.getvalue()).decode('utf-8')


def load_time_series_from_base64(ts_base64: bytes | str) -> np.ndarray:
    """Load time series from base64 format."""
    if isinstance(ts_base64, str):
        ts_base64 = ts_base64.encode('utf-8')
    return np.load(BytesIO(pybase64.b64decode(ts_base64)), alow_pickle=False)


def load_time_series(data_source: str | np.ndarray) -> np.ndarray:
    """Load time series from URL, local path, base64 data URL, or numpy
    array."""
    try:
        if isinstance(data_source, np.ndarray):
            return data_source

        if data_source.startswith('http'):
            response = requests.get(data_source, headers=HEADERS, timeout=FETCH_TIMEOUT)
            response.raise_for_status()
            return _load_bytes(response.content, data_source)

        if data_source.startswith('data:time_series'):
            return load_time_series_from_base64(data_source.split(',')[1])

        if data_source.startswith('file://'):
            path = data_source.removeprefix('file://')
            return _load_path(path)

        if os.path.exists(data_source):
            return _load_path(data_source)

        raise ValueError(f'Invalid data source: {data_source}')
    except Exception as error:
        data_info = str(data_source)[:100] + ' ...' if isinstance(data_source,
                                                                  str) and len(data_source) > 100 else str(data_source)
        logger.error(f'{error}, data_source={data_info}')
        return np.zeros((6000, 3), dtype=np.float32)  # dummy


def _load_bytes(content: bytes, hint: str = '') -> np.ndarray:
    """Auto-detect format from bytes.

    Try: npy -> csv -> audio.
    """
    hint = hint.lower()

    # Format hints from URL/path
    if '.npy' in hint:
        return np.load(BytesIO(content))
    if '.csv' in hint:
        return _load_csv(content)
    if any(ext in hint for ext in ['.wav', '.mp3', '.flac']):
        return _load_audio(content)

    # Fallback: try all formats
    loaders = [lambda: np.load(BytesIO(content)), lambda: _load_csv(content), lambda: _load_audio(content)]
    for loader in loaders:
        try:
            return loader()
        except Exception:
            continue
    raise ValueError(f'Cannot detect format from bytes: {hint[:50]}')


def _load_path(path: str) -> np.ndarray:
    """Load from local file path based on extension."""
    ext = os.path.splitext(path)[-1].lower()

    if ext == '.npy':
        return np.load(path)
    if ext == '.csv':
        return _load_csv(path)
    if ext in ['.wav', '.mp3', '.flac']:
        return _load_audio(path)

    raise ValueError(f'Unsupported format: {ext}')


def _load_csv(source: bytes | str) -> np.ndarray:
    """Load CSV from bytes or file path."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError('Please install pandas to process csv files.')

    if isinstance(source, bytes):
        return pd.read_csv(BytesIO(source), header=None).values
    return pd.read_csv(source, header=None).values


def _load_audio(source: bytes | str) -> np.ndarray:
    """Load audio from bytes or file path."""
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError('Please install soundfile to process audio files.')

    if isinstance(source, bytes):
        source = BytesIO(source)
    ts, sr = sf.read(source)
    return ts
