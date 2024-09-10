from lmdeploy.vl.utils import (encode_image_base64, load_image,
                               load_image_from_base64)


def test_encode_image_base64():
    url = 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'  # noqa E501
    im1 = load_image(url)
    base64 = encode_image_base64(url)
    im2 = load_image_from_base64(base64)
    assert im1 == im2.convert('RGB')


def test_load_truncated_image():
    url = 'https://github.com/irexyc/lmdeploy/releases/download/v0.0.1/tr.jpeg'
    im = load_image(url)
    assert im.width == 1638
    assert im.height == 2048


def test_load_invalid_url():
    url = ('https://raw.githubusercontent.com/open-mmlab/'
           'mmdeploy/main/tests/data/tiger.jpeg')
    # invalid
    im1 = load_image(url[:-1])
    assert im1.width == 32
    assert im1.height == 32
    # valid
    im2 = load_image(url)
    assert im2.height == 182
    assert im2.width == 278


def test_load_invalid_base64():
    base64 = 'data:image/jpeg;base64,xxx'
    im = load_image(base64)
    assert im.width == 32
    assert im.height == 32
