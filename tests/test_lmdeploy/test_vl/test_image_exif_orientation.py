# Copyright (c) OpenMMLab. All rights reserved.
from io import BytesIO

from PIL import Image

from lmdeploy.vl.media.image import ImageMediaIO


def test_image_loader_honors_exif_rotation_for_bytes_and_files(tmp_path):
    # 3x2 image; orientation 6 is a clockwise quarter turn on display.
    image = Image.new('RGB', (3, 2))
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    image.putdata(colors)
    exif = Image.Exif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, format='PNG', exif=exif)
    data = buffer.getvalue()
    path = tmp_path / 'oriented.png'
    path.write_bytes(data)
    loader = ImageMediaIO()
    for loaded in [loader.load_bytes(data), loader.load_file(path)]:
        assert loaded.size == (2, 3)
        assert list(loaded.getdata()) == [colors[i] for i in (3, 0, 4, 1, 5, 2)]
        assert loaded.getexif().get(274, 1) == 1
