from lmdeploy.vl.utils import (encode_image_base64, load_image,
                               load_image_from_base64)


def test_encode_image_base64():
    url = 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'  # noqa E501
    im1 = load_image(url)
    base64 = encode_image_base64(url)
    im2 = load_image_from_base64(base64)
    assert im1 == im2
