import math

import numpy as np

from lmdeploy.vl import (
    encode_image_base64,
    encode_time_series_base64,
    encode_video_base64,
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


def test_image_modes():
    import numpy as np
    from PIL import Image

    grayscale_img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8)).convert('L')
    b64 = encode_image_base64(grayscale_img)  # should convert L -> RGB internally

    img_out = load_image(f'data:image/png;base64,{b64}')
    assert img_out.mode == 'RGB'


def test_truncated_image():
    url = 'https://github.com/irexyc/lmdeploy/releases/download/v0.0.1/tr.jpeg'
    im = load_image(url)
    assert im.width == 1638
    assert im.height == 2048


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
