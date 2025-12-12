import copy

import pytest

from lmdeploy.vl.model.qwen3 import Qwen3VLModel
from lmdeploy.vl.utils import load_image

QWEN3VL_MODELS = [
    'Qwen/Qwen3-VL-4B-Instruct',
]

IMAGE_URL = ('https://raw.githubusercontent.com/open-mmlab/'
             'mmdeploy/main/tests/data/tiger.jpeg')


@pytest.fixture(scope='module', params=QWEN3VL_MODELS)
def qwen3vl_model(request):
    """Initialize Qwen3VLModel with a real model path."""
    model = Qwen3VLModel(model_path=request.param)
    model.build_preprocessor()
    return model


@pytest.fixture
def sample_messages():
    """Create sample messages for preprocessing using image_url."""
    pil_image = load_image(IMAGE_URL)
    return [{
        'role':
        'user',
        'content': [
            {
                'type': 'text',
                'text': 'Can you describe this image?'
            },
            {
                'type': 'image',
                'image': pil_image
            },
        ]
    }]


def test_qwen3vl_preprocess_with_custom_pixels(qwen3vl_model, sample_messages):
    """Test that mm_processor_kwargs with min/max pixels takes effect."""

    # compression ratio for qwen3vl is 32 = patch_size * spatial_merge_size = 16 * 2
    # qwen3vl_model.processor.image_processor.size['shortest_edge'] = 66536
    # 65536 = 64 * 32 * 32, indicates 64 image token budget
    # qwen3vl_model.processor.image_processor.size['longest_edge'] = 16777216
    # 16777216 = 16384 * 32 * 32, indicates 16384 image token budget

    # Default processing without custom arguments
    default_processed_messages = qwen3vl_model.preprocess(messages=copy.deepcopy(sample_messages))
    default_content = default_processed_messages[-1]['content']
    default_shape = default_content[0]['pixel_values'].shape  # [280, 1536]

    # Processing with smaller pixel range
    mm_processor_kwargs = {'min_pixels': 10 * 32 * 32, 'max_pixels': 20 * 32 * 32}
    custom_processed_messages = qwen3vl_model.preprocess(messages=copy.deepcopy(sample_messages),
                                                         mm_processor_kwargs=mm_processor_kwargs)
    custom_content = custom_processed_messages[-1]['content']
    custom_shape = custom_content[0]['pixel_values'].shape  # [60, 1536]

    assert default_shape != custom_shape, \
        'Default and custom processing should result in different shapes.'
    assert default_shape[0] > custom_shape[0], \
        'Custom processing with smaller pixel range should result in smaller image size.'

    # Processing with larger pixel range
    mm_processor_kwargs = {'min_pixels': 100 * 32 * 32, 'max_pixels': 20000 * 32 * 32}
    custom_processed_messages = qwen3vl_model.preprocess(messages=copy.deepcopy(sample_messages),
                                                         mm_processor_kwargs=mm_processor_kwargs)
    custom_content = custom_processed_messages[-1]['content']
    custom_shape = custom_content[0]['pixel_values'].shape  # [468, 1536]

    assert default_shape != custom_shape, \
        'Default and custom processing should result in different shapes.'
    assert default_shape[0] < custom_shape[0], \
        'Custom processing with larger pixel range should result in larger image size.'
