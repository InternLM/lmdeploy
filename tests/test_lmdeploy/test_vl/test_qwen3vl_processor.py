import pytest

from lmdeploy.vl import load_image, load_video
from lmdeploy.vl.model.qwen3 import Qwen3VLModel

QWEN3VL_MODELS = [
    'Qwen/Qwen3-VL-4B-Instruct',
]

IMAGE_URL = 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
VIDEO_URL = 'https://raw.githubusercontent.com/CUHKSZzxy/Online-Data/main/clip_3_removed.mp4'


@pytest.fixture(scope='module', params=QWEN3VL_MODELS)
def qwen3vl_model(request):
    model = Qwen3VLModel(model_path=request.param)
    model.build_preprocessor()
    return model


@pytest.fixture(scope='module')
def pil_image():
    return load_image(IMAGE_URL)


@pytest.fixture(scope='module')
def video_data():
    return load_video(VIDEO_URL, num_frames=16)


@pytest.fixture
def sample_messages(pil_image):
    return [{'role': 'user', 'content': [{'type': 'image', 'data': pil_image}]}]


@pytest.fixture
def sample_video_messages(video_data):
    frames, metadata = video_data
    return [{'role': 'user', 'content': [{'type': 'video', 'data': frames, 'video_metadata': metadata}]}]


def _preprocess(model, messages, mm_processor_kwargs=None):
    """Call model.preprocess following the same flow as the engine:

    apply_chat_template → input_prompt → preprocess.
    """
    from lmdeploy.model import MODELS
    chat_template = MODELS.module_dict['hf'](model_path=model.model_path)
    input_prompt = model.apply_chat_template(messages, chat_template, sequence_start=True)
    result = model.preprocess(messages=list(messages), input_prompt=input_prompt,
                              mm_processor_kwargs=mm_processor_kwargs)
    return result['multimodal'][0]


def test_image_with_custom_pixels(qwen3vl_model, sample_messages):
    """Test that mm_processor_kwargs min/max pixels affect image preprocessing.

    compression ratio for qwen3vl is 32 = patch_size * spatial_merge_size = 16 * 2,
    image_processor.size['shortest_edge'] = 65536 = 64 * 32 * 32      (64 token budget),
    image_processor.size['longest_edge'] = 16777216 = 16384 * 32 * 32 (16384 token budget),
    """

    # [280, 1536]
    default_shape = _preprocess(qwen3vl_model, sample_messages)['pixel_values'].shape

    # [60, 1536]
    small_kwargs = {'image': {'min_pixels': 10 * 32 * 32, 'max_pixels': 20 * 32 * 32}}
    small_shape = _preprocess(qwen3vl_model, sample_messages,
                              mm_processor_kwargs=small_kwargs)['pixel_values'].shape

    # [468, 1536]
    large_kwargs = {'image': {'min_pixels': 100 * 32 * 32, 'max_pixels': 20000 * 32 * 32}}
    large_shape = _preprocess(qwen3vl_model, sample_messages,
                              mm_processor_kwargs=large_kwargs)['pixel_values'].shape

    assert small_shape[0] < default_shape[0] < large_shape[0]


def test_video_with_custom_pixels(qwen3vl_model, sample_video_messages):
    """Test that mm_processor_kwargs min/max pixels affect video preprocessing.

    Videos process at native resolution by default, so we compare two constrained ranges rather than comparing against
    the default. Per-frame shapes are compared (each multimodal item is one frame).
    """

    # [28160, 1536]
    default_shape = _preprocess(qwen3vl_model, sample_video_messages)['pixel_values_videos'].shape

    # [4, 1536]
    small_kwargs = {'video': {'min_pixels': 10 * 32 * 32, 'max_pixels': 20 * 32 * 32}}
    small_shape = _preprocess(qwen3vl_model, sample_video_messages,
                              mm_processor_kwargs=small_kwargs)['pixel_values_videos'].shape

    # [32, 1536]
    medium_kwargs = {'video': {'min_pixels': 50 * 32 * 32, 'max_pixels': 200 * 32 * 32}}
    medium_shape = _preprocess(qwen3vl_model, sample_video_messages,
                               mm_processor_kwargs=medium_kwargs)['pixel_values_videos'].shape

    assert small_shape[0] < medium_shape[0] <= default_shape[0]
