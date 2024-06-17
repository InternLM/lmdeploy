# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os
from io import BytesIO
from typing import Union

import requests
from PIL import Image


def encode_image_base64(image: Union[str, Image.Image]) -> str:
    """encode raw date to base64 format."""
    res = ''
    if isinstance(image, str):
        url_or_path = image
        if url_or_path.startswith('http'):
            FETCH_TIMEOUT = int(os.environ.get('LMDEPLOY_FETCH_TIMEOUT', 10))
            headers = {
                'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
            try:
                response = requests.get(url_or_path,
                                        headers=headers,
                                        timeout=FETCH_TIMEOUT)
                response.raise_for_status()
                res = base64.b64encode(response.content).decode('utf-8')
            except Exception:
                pass
        elif os.path.exists(url_or_path):
            with open(url_or_path, 'rb') as image_file:
                res = base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        res = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return res


def load_image_from_base64(image: Union[bytes, str]) -> Image.Image:
    """load image from base64 format."""
    return Image.open(BytesIO(base64.b64decode(image)))


def load_image(image_url: Union[str, Image.Image]) -> Image.Image:
    """load image from url, local path or openai GPT4V."""
    if isinstance(image_url, Image.Image):
        return image_url

    FETCH_TIMEOUT = int(os.environ.get('LMDEPLOY_FETCH_TIMEOUT', 10))
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    if image_url.startswith('http'):
        response = requests.get(image_url,
                                headers=headers,
                                timeout=FETCH_TIMEOUT)
        response.raise_for_status()

        # Open the image using PIL
        img = Image.open(BytesIO(response.content))
    elif image_url.startswith('data:image'):
        img = load_image_from_base64(image_url.split(',')[1])
    else:
        # Load image from local path
        img = Image.open(image_url)

    return img
