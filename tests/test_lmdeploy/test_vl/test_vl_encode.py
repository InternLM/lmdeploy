import base64
import math
from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageFile, ImageOps

from lmdeploy.vl import (
    encode_audio_base64,
    encode_image_base64,
    encode_time_series_base64,
    encode_video_base64,
    load_audio,
    load_image,
    load_time_series,
    load_video,
)


def test_image_encode_decode():
    url = 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'

    img1 = load_image(url)
    # use PNG for lossless pixel-perfect comparison
    b64 = encode_image_base64(url, format='PNG')
    img2 = load_image(f'data:image/png;base64,{b64}')

    assert img1.size == img2.size
    assert img1.mode == img2.mode
    assert img1.tobytes() == img2.tobytes()


def test_video_encode_decode():
    # url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4'
    url = 'https://raw.githubusercontent.com/CUHKSZzxy/Online-Data/main/clip_3_removed.mp4'

    # num_frames=4 to keep test fast
    vid1, meta1 = load_video(url, num_frames=4)
    b64 = encode_video_base64(url, num_frames=4, format='JPEG')
    vid2, meta2 = load_video(f'data:video/jpeg;base64,{b64}')

    gt_meta = {
        'total_num_frames': 498,
        'fps': 29.97002997002997,
        'duration': 16.616600000000002,
        'video_backend': 'opencv',
        'frames_indices': [0, 165, 331, 497]
    }

    assert vid1.shape == vid2.shape
    assert np.mean(np.abs(vid1.astype(float) - vid2.astype(float))) < 2.0  # JPEG is lossy
    assert meta1['total_num_frames'] == gt_meta['total_num_frames']
    assert meta1['frames_indices'] == gt_meta['frames_indices']


def test_time_series_encode_decode():
    # url = "https://huggingface.co/internlm/Intern-S1-Pro/raw/main/0092638_seism.npy"
    url = 'https://raw.githubusercontent.com/CUHKSZzxy/Online-Data/main/0092638_seism.npy'

    ts1 = load_time_series(url)
    b64 = encode_time_series_base64(url)
    ts2 = load_time_series(f'data:time_series/npy;base64,{b64}')

    assert ts1.shape == ts2.shape
    assert np.allclose(ts1, ts2)


def test_audio_encode_decode():
    import pytest
    pytest.importorskip('librosa')
    pytest.importorskip('soundfile')

    sr = 16000
    seconds = 0.25
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)

    b64 = encode_audio_base64((audio, sr))
    audio_out, sr_out = load_audio(f'data:audio/wav;base64,{b64}', sampling_rate=sr)

    assert sr_out == sr
    assert audio_out.shape == audio.shape
    assert np.allclose(audio_out, audio, atol=1e-4)


@pytest.mark.parametrize('source_mode', ['L', 'RGBA', 'CMYK'])
@pytest.mark.parametrize('image_mode', ['RGB', 'L'])
def test_image_modes(source_mode, image_mode):
    image = Image.new('RGB', (32, 24), (120, 30, 200)).convert(source_mode)
    b64 = encode_image_base64(image, image_mode=image_mode)
    img_out = load_image(f'data:image/png;base64,{b64}', image_mode=image_mode)
    assert img_out.mode == image_mode
    assert img_out.tobytes() == image.convert(image_mode).tobytes()


@pytest.mark.parametrize('orientation', range(1, 10))
def test_image_exif_orientation(tmp_path, orientation):
    from lmdeploy.vl.media.image import ImageMediaIO

    y, x = np.indices((112, 56))
    image = Image.fromarray(np.stack([x * 4, y * 2, (x + y) % 256], axis=-1).astype(np.uint8))
    exif = Image.Exif()
    exif[274] = orientation
    buffer = BytesIO()
    image.save(buffer, format='JPEG', exif=exif)
    data = buffer.getvalue()
    path = tmp_path / 'oriented.jpg'
    path.write_bytes(data)
    original = Image.open(BytesIO(data))
    original_pixels = original.tobytes()
    expected = ImageOps.exif_transpose(original).convert('RGB')
    io = ImageMediaIO()
    outputs = [io.load_bytes(data), io.load_file(path),
               load_image('data:image/jpeg;base64,' + base64.b64encode(data).decode()),
               load_image('data:image/png;base64,' + encode_image_base64(original))]
    for actual in outputs:
        assert actual.size == expected.size
        assert actual.tobytes() == expected.tobytes()
        assert ImageOps.exif_transpose(actual).tobytes() == actual.tobytes()
        assert actual.getexif().get(274, 1) not in range(2, 9)
    jpeg = load_image('data:image/jpeg;base64,' + encode_image_base64(original, format='JPEG'))
    assert jpeg.size == expected.size
    assert np.abs(np.asarray(jpeg).astype(float) - np.asarray(expected).astype(float)).mean() < 3
    assert original.size == (56, 112)
    assert original.tobytes() == original_pixels
    assert original.getexif()[274] == orientation


def test_image_malformed_exif_remains_decodable():
    import struct

    from lmdeploy.vl.media.image import ImageMediaIO

    buffer = BytesIO()
    Image.new('RGB', (56, 112), 'red').save(buffer, format='JPEG')
    jpeg = buffer.getvalue()
    # A real APP1 segment with an invalid TIFF header; Pillow can still decode pixels.
    payload = b'Exif\x00\x00XXXX\x00\x00\x00\x08' + bytes(32)
    data = jpeg[:2] + b'\xff\xe1' + struct.pack('>H', len(payload) + 2) + payload + jpeg[2:]
    reference = Image.open(BytesIO(data)).convert('RGB')
    actual = ImageMediaIO().load_bytes(data)
    assert actual.size == reference.size
    assert actual.tobytes() == reference.tobytes()
    assert not actual.getexif()


@pytest.mark.parametrize('orientation', [1, 6])
def test_jpeg_video_frames_preserve_metadata(orientation):
    frames = []
    expected = []
    for color in ['red', 'blue']:
        buffer = BytesIO()
        exif = Image.Exif()
        exif[274] = orientation
        Image.new('RGB', (56, 112), color).save(buffer, format='JPEG', exif=exif)
        data = buffer.getvalue()
        frames.append(base64.b64encode(data).decode())
        expected.append(np.asarray(ImageOps.exif_transpose(Image.open(BytesIO(data))).convert('RGB')))
    video, metadata = load_video('data:video/jpeg;base64,' + ','.join(frames), fps=2)
    assert np.array_equal(video, np.stack(expected))
    assert metadata == dict(total_num_frames=2, fps=2.0, duration=1.0,
                            video_backend='jpeg_sequence', frames_indices=[0, 1])


@pytest.mark.parametrize('orientation', [1, 6])
def test_truncated_image(monkeypatch, orientation):
    buffer = BytesIO()
    exif = Image.Exif()
    exif[274] = orientation
    Image.new('RGB', (1638, 2048), (120, 30, 200)).save(buffer, format='JPEG', exif=exif)
    data = buffer.getvalue()[:-128]  # Truncate scan data; a missing EOI alone may be tolerated by the decoder.
    with monkeypatch.context() as context:
        context.setattr(ImageFile, 'LOAD_TRUNCATED_IMAGES', False)
        with pytest.raises(OSError, match='truncated'):
            Image.open(BytesIO(data)).load()

    # A data URL exercises the HTTP byte decoder without a network dependency.
    im = load_image('data:image/jpeg;base64,' + base64.b64encode(data).decode())
    assert im.size == ((1638, 2048) if orientation == 1 else (2048, 1638))
    assert im.mode == 'RGB'


def test_single_frame_video():
    url = 'https://raw.githubusercontent.com/CUHKSZzxy/Online-Data/main/clip_3_removed.mp4'
    vid, meta = load_video(url, num_frames=1)
    assert vid.shape[0] == 1

    b64 = encode_video_base64(vid)
    assert isinstance(b64, str)
    assert ',' not in b64  # should only be one JPEG block, no commas


def test_video_sampling_params():
    url = 'https://raw.githubusercontent.com/CUHKSZzxy/Online-Data/main/clip_3_removed.mp4'

    # 1. test num_frames constraint
    num_frames = 5
    vid, meta = load_video(url, num_frames=num_frames)
    assert vid.shape[0] == num_frames
    assert len(meta['frames_indices']) == num_frames

    # 2. test fps constraint (original fps is ~29.97, duration ~16.6s)
    fps = 1
    vid, meta = load_video(url, fps=fps)
    expected_frames = max(1, int(math.floor(meta['duration'] * fps)))
    assert vid.shape[0] == expected_frames

    # 3. test both constraints (should take the minimum)
    # 10 fps x 16.6s ~= 166 frames > 10 frames, so will be limited by num_frames
    num_frames = 10
    fps = 10
    vid, meta = load_video(url, num_frames=num_frames, fps=fps)
    assert vid.shape[0] == num_frames

    # 1 fps x 16.6s ~= 16 frames < 100 frames, so will be limited by fps
    num_frames = 100
    fps = 1
    vid, meta = load_video(url, num_frames=num_frames, fps=fps)
    expected_frames = max(1, int(math.floor(meta['duration'] * fps)))
    assert vid.shape[0] == expected_frames


def test_invalid_inputs():
    # non-existent local path
    import pytest
    with pytest.raises(Exception):
        load_image('/non_existent/path/image.jpg')
    with pytest.raises(Exception):
        load_video('/non_existent/path/video.mp4')
    with pytest.raises(Exception):
        load_time_series('/non_existent/path/data.npy')
