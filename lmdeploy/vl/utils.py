# Copyright (c) OpenMMLab. All rights reserved.
import base64
from io import BytesIO
from typing import Union

import requests
from PIL import Image


def encode_image_base64(image: Image.Image) -> str:
    """encode image to base64 format."""
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def load_image_from_base64(image: Union[bytes, str]) -> Image.Image:
    """load image from base64 format."""
    return Image.open(BytesIO(base64.b64decode(image)))


def load_image(image_url: str) -> Image.Image:
    """load image from url, local path or openai GPT4V."""

    headers = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    if image_url.startswith('http'):
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()

        # Open the image using PIL
        img = Image.open(BytesIO(response.content))
    elif image_url.startswith('data:image'):
        img = load_image_from_base64(image_url.split(',')[1])
    else:
        img = Image.open(image_url)

    return img
