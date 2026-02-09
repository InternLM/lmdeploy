# Copyright (c) OpenMMLab. All rights reserved.
import os
import pickle
from io import BytesIO
from typing import Union

import pybase64
import requests
from numpy.typing import NDArray
from PIL import Image, ImageFile

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def encode_image_base64(image: Union[str, Image.Image]) -> str:
    """Encode raw data to base64 format."""
    buffered = BytesIO()
    FETCH_TIMEOUT = int(os.environ.get('LMDEPLOY_FETCH_TIMEOUT', 10))
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        if isinstance(image, str):
            url_or_path = image
            if url_or_path.startswith('http'):
                response = requests.get(url_or_path, headers=headers, timeout=FETCH_TIMEOUT)
                response.raise_for_status()
                buffered.write(response.content)
            elif os.path.exists(url_or_path):
                with open(url_or_path, 'rb') as image_file:
                    buffered.write(image_file.read())
        elif isinstance(image, Image.Image):
            image.save(buffered, format='PNG')
    except Exception as error:
        if isinstance(image, str) and len(image) > 100:
            image = image[:100] + ' ...'
        logger.error(f'{error}, image={image}')
        # use dummy image
        image = Image.new('RGB', (32, 32))
        image.save(buffered, format='PNG')
    res = pybase64.b64encode(buffered.getvalue()).decode('utf-8')
    return res


def load_image_from_base64(image: Union[bytes, str]) -> Image.Image:
    """Load image from base64 format."""
    return Image.open(BytesIO(pybase64.b64decode(image)))


def load_image(image_url: Union[str, Image.Image]) -> Image.Image:
    """Load image from url, local path or openai GPT4V."""
    FETCH_TIMEOUT = int(os.environ.get('LMDEPLOY_FETCH_TIMEOUT', 10))
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if isinstance(image_url, Image.Image):
            img = image_url
        elif image_url.startswith('http'):
            response = requests.get(image_url, headers=headers, timeout=FETCH_TIMEOUT)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        elif image_url.startswith('data:image'):
            img = load_image_from_base64(image_url.split(',')[1])
        else:
            # Load image from local path
            img = Image.open(image_url)

        # check image valid
        img = img.convert('RGB')
    except Exception as error:
        if isinstance(image_url, str) and len(image_url) > 100:
            image_url = image_url[:100] + ' ...'
        logger.error(f'{error}, image_url={image_url}')
        # use dummy image
        img = Image.new('RGB', (32, 32))

    return img


def encode_time_series_base64(path: str) -> str:
    """Encode raw time series data to base64 format."""

    # TODO: zhouxinyu, support url or numpy array input
    assert type(path) == str, f'Expected path to be a string, but got {type(path)}'

    ext = os.path.splitext(path)[-1].lower()
    if ext in ['.wav', '.mp3', '.flac']:
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError('Please install soundfile to process audio files.')
        ts_input, sr = sf.read(path)  # ts_input: np.ndarray, shape [T] or [T, C]
    elif ext == '.csv':
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('Please install pandas to process csv files.')
        df = pd.read_csv(path, header=None)
        ts_input = df.values  # [T, C]
    elif ext == '.npy':
        try:
            import numpy as np
        except ImportError:
            raise ImportError('Please install numpy to process npy files.')
        ts_input = np.load(path)  # [T, C]
    else:
        raise ValueError(f'Unsupported file format: {ext}')

    # numpy ndarray -> bytes
    ts_bytes = pickle.dumps(ts_input)

    # bytes -> base64
    ts_base64 = pybase64.b64encode(ts_bytes).decode('utf-8')
    return ts_base64


def load_time_series_from_base64(ts_base64: str) -> NDArray:
    """Load time series data from base64 format."""
    # base64 -> bytes
    ts_bytes = pybase64.b64decode(ts_base64)

    # bytes -> numpy ndarray
    ts_input = pickle.loads(ts_bytes)
    return ts_input


def load_time_series(time_series_url: str) -> NDArray:
    """Load time series data from url or local path."""
    # TODO: zhouxinyu, support loading time series data from url or numpy array input

    assert type(time_series_url) == str, f'Expected time_series_url to be a string, but got {type(time_series_url)}'

    ts_input = None

    if time_series_url.startswith('http'):
        pass
    elif time_series_url.startswith('data:time_series'):
        ts_input = load_time_series_from_base64(time_series_url.split(',')[1])
    else:
        pass

    return ts_input
