# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os
from io import BytesIO
from typing import Union

import requests
from PIL import Image, ImageFile

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def encode_image_base64(image: Union[str, Image.Image]) -> str:
    """encode raw date to base64 format."""
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
                response = requests.get(url_or_path,
                                        headers=headers,
                                        timeout=FETCH_TIMEOUT)
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
    res = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return res


def load_image_from_base64(image: Union[bytes, str]) -> Image.Image:
    """load image from base64 format."""
    return Image.open(BytesIO(base64.b64decode(image)))


def load_image(image_url: Union[str, Image.Image]) -> Image.Image:
    """load image from url, local path or openai GPT4V."""
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
            response = requests.get(image_url,
                                    headers=headers,
                                    timeout=FETCH_TIMEOUT)
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
