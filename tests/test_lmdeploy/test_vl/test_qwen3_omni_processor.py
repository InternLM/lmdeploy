from types import SimpleNamespace

import torch

from lmdeploy.pytorch.models.qwen3_omni_moe_thinker import Qwen3OmniInputProcessor
from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.base import MultimodalSpecialTokens, VisionModel
from lmdeploy.vl.model.preprocess_utils import get_expanded_mm_items
from lmdeploy.vl.model.qwen3_omni import Qwen3OmniModel


class FakeQwen3OmniProcessor:

    image_token = '<image>'
    audio_token = '<audio>'
    video_token = '<video>'

    def __init__(self):
        self.image_token_id = 10
        self.audio_token_id = 11
        self.video_token_id = 12
        self.image_processor = SimpleNamespace(merge_size=2, size={'shortest_edge': 4, 'longest_edge': 1024})
        self.video_processor = SimpleNamespace(merge_size=2,
                                               temporal_patch_size=2,
                                               size={'shortest_edge': 4, 'longest_edge': 1024})
        self.calls = []

    def __call__(self, text, images=None, videos=None, audio=None, return_tensors=None, **kwargs):
        self.calls.append(kwargs)
        data = {}
        token_counts = {}

        if images is not None:
            image_grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
            image_tokens = int(image_grid[0].prod().item() // self.image_processor.merge_size**2)
            token_counts[self.image_token] = (self.image_token_id, image_tokens)
            data['pixel_values'] = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
            data['image_grid_thw'] = image_grid

        if videos is not None:
            video_grid = torch.tensor([[2, 4, 4]], dtype=torch.long)
            video_tokens = int(video_grid[0].prod().item() // self.video_processor.merge_size**2)
            token_counts[self.video_token] = (self.video_token_id, video_tokens)
            data['pixel_values_videos'] = torch.arange(32 * 3, dtype=torch.float32).reshape(32, 3)
            data['video_grid_thw'] = video_grid
            data['video_second_per_grid'] = torch.tensor([2.0])

        if audio is not None:
            audio_frames = 300
            audio_tokens = 39
            token_counts[self.audio_token] = (self.audio_token_id, audio_tokens)
            data['input_features'] = torch.ones(1, 128, audio_frames)
            data['feature_attention_mask'] = torch.ones(1, audio_frames, dtype=torch.long)

        ids = [1]
        positions = []
        for mm_token, (token_id, token_count) in token_counts.items():
            positions.append((text[0].index(mm_token), token_id, token_count))
        for _, token_id, token_count in sorted(positions):
            ids.extend([token_id] * token_count)
            ids.append(2)

        data['input_ids'] = torch.tensor([ids], dtype=torch.long)
        return data


def _fake_model():
    model = Qwen3OmniModel.__new__(Qwen3OmniModel)
    model.processor = FakeQwen3OmniProcessor()
    model.image_token = model.processor.image_token
    model.audio_token = model.processor.audio_token
    model.video_token = model.processor.video_token
    model.image_token_id = model.processor.image_token_id
    model.audio_token_id = model.processor.audio_token_id
    model.video_token_id = model.processor.video_token_id
    model.mm_tokens = MultimodalSpecialTokens(image_token=model.image_token,
                                              audio_token=model.audio_token,
                                              video_token=model.video_token,
                                              image_token_id=model.image_token_id,
                                              audio_token_id=model.audio_token_id,
                                              video_token_id=model.video_token_id)
    return model


def test_image_only_new_preprocess_returns_token_span():
    model = _fake_model()
    messages = [{'role': 'user', 'content': [{'type': 'image', 'data': object()}]}]

    result = model.preprocess(messages, input_prompt=f'describe {model.image_token}')

    assert result['input_ids'].count(model.image_token_id) == 4
    assert len(result['multimodal']) == 1
    image = result['multimodal'][0]
    assert image['modality'] == Modality.IMAGE
    assert image['pixel_values'].shape == (16, 3)
    assert image['image_grid_thw'].tolist() == [1, 4, 4]
    assert image['offset'] == (1, 5)
    assert image['image_token_id'] == model.image_token_id
    assert 'mm_token_num' not in image


def test_qwen3_omni_uses_shared_new_preprocess_path():
    assert Qwen3OmniModel.preprocess is VisionModel.preprocess


def test_audio_only_new_preprocess_returns_audio_features_and_span():
    model = _fake_model()
    messages = [{'role': 'user', 'content': [{'type': 'audio', 'data': ('audio-array', 16000)}]}]

    result = model.preprocess(messages, input_prompt=f'transcribe {model.audio_token}')

    assert result['input_ids'].count(model.audio_token_id) == 39
    audio = result['multimodal'][0]
    assert audio['modality'] == Modality.AUDIO
    assert audio['input_features'].shape == (1, 128, 300)
    assert audio['feature_attention_mask'].shape == (1, 300)
    assert 'audio_feature_lengths' not in audio
    assert audio['offset'] == (1, 40)
    assert audio['audio_token_id'] == model.audio_token_id
    assert 'mm_token_num' not in audio


def test_video_only_new_preprocess_keeps_whole_video_item():
    model = _fake_model()
    messages = [{'role': 'user', 'content': [{'type': 'video', 'data': ['f0', 'f1'], 'video_metadata': {'fps': 1}}]}]

    result = model.preprocess(messages, input_prompt=f'describe {model.video_token}')

    assert result['input_ids'].count(model.video_token_id) == 8
    assert len(result['multimodal']) == 1
    video = result['multimodal'][0]
    assert video['modality'] == Modality.VIDEO
    assert video['pixel_values_videos'].shape == (32, 3)
    assert video['video_grid_thw'].tolist() == [2, 4, 4]
    assert video['offset'] == (1, 9)
    assert video['second_per_grid'] == 2.0
    assert video['video_token_id'] == model.video_token_id
    assert 'mm_token_num' not in video


def test_video_expansion_distinguishes_qwen3vl_frames_from_omni_whole_video():
    mm_tokens = MultimodalSpecialTokens(video_token='<video>', video_token_id=12)
    qwen3vl_style = {
        Modality.VIDEO: {
            'feature': torch.arange(32 * 3, dtype=torch.float32).reshape(32, 3),
            'video_grid_thw': torch.tensor([[2, 4, 4]], dtype=torch.long),
            'offset': [(1, 5), (7, 11)],
        }
    }
    omni_style = {
        Modality.VIDEO: {
            'feature': torch.arange(32 * 3, dtype=torch.float32).reshape(32, 3),
            'video_grid_thw': torch.tensor([[2, 4, 4]], dtype=torch.long),
            'video_second_per_grid': torch.tensor([2.0]),
            'offset': [(1, 9)],
        }
    }

    qwen3vl_items = get_expanded_mm_items(qwen3vl_style, mm_tokens)
    omni_items = get_expanded_mm_items(omni_style, mm_tokens)

    assert len(qwen3vl_items) == 2
    assert [item['video_grid_thw'].tolist() for item in qwen3vl_items] == [[1, 4, 4], [1, 4, 4]]
    assert [item['offset'] for item in qwen3vl_items] == [(1, 5), (7, 11)]
    assert len(omni_items) == 1
    assert omni_items[0]['video_grid_thw'].tolist() == [2, 4, 4]
    assert omni_items[0]['offset'] == (1, 9)
    assert omni_items[0]['second_per_grid'] == 2.0


def test_mixed_image_audio_video_preserves_prompt_order_and_independent_offsets():
    model = _fake_model()
    messages = [{
        'role':
        'user',
        'content': [
            {
                'type': 'image',
                'data': object(),
            },
            {
                'type': 'audio',
                'data': ('audio-array', 16000),
            },
            {
                'type': 'video',
                'data': ['f0', 'f1'],
                'video_metadata': {
                    'fps': 1,
                },
            },
        ],
    }]

    result = model.preprocess(messages,
                              input_prompt=f'{model.image_token} then {model.audio_token} then {model.video_token}',
                              mm_processor_kwargs={
                                  'image': {
                                      'min_pixels': 4,
                                      'max_pixels': 16,
                                  },
                                  'video': {
                                      'min_pixels': 4,
                                      'max_pixels': 16,
                                      'fps': 1,
                                  },
                              })

    assert [item['modality'] for item in result['multimodal']] == [Modality.IMAGE, Modality.AUDIO, Modality.VIDEO]
    assert [item['offset'] for item in result['multimodal']] == [(1, 5), (6, 45), (46, 54)]
    assert result['input_ids'].count(model.image_token_id) == 4
    assert result['input_ids'].count(model.audio_token_id) == 39
    assert result['input_ids'].count(model.video_token_id) == 8
    processor_kwargs = model.processor.calls[-1]
    assert processor_kwargs['padding'] is True
    assert processor_kwargs['images_kwargs']['size'] == {'shortest_edge': 4, 'longest_edge': 16}
    assert processor_kwargs['videos_kwargs']['size'] == {'shortest_edge': 4, 'longest_edge': 16}
    assert 'audio_kwargs' not in processor_kwargs


def test_qwen3_omni_input_processor_packs_mixed_modalities():
    processor = Qwen3OmniInputProcessor(config=SimpleNamespace())
    image = {
        'modality': Modality.IMAGE,
        'pixel_values': torch.ones(16, 3),
        'image_grid_thw': torch.tensor([1, 4, 4]),
        'offset': (1, 5),
        'image_token_id': 10,
    }
    audio = {
        'modality': Modality.AUDIO,
        'input_features': torch.ones(1, 128, 300),
        'feature_attention_mask': torch.cat(
            [torch.ones(1, 280, dtype=torch.long),
             torch.zeros(1, 20, dtype=torch.long)], dim=1),
        'offset': (6, 45),
        'audio_token_id': 11,
    }
    video = {
        'modality': Modality.VIDEO,
        'pixel_values_videos': torch.ones(32, 3),
        'video_grid_thw': torch.tensor([2, 4, 4]),
        'offset': (46, 54),
        'second_per_grid': 2.0,
        'video_token_id': 12,
    }

    result = processor.preprocess_input([1, 10, 11, 12], [image, audio, video])

    mm_data = result.input_multimodals['mm_data']
    assert [item.modality for item in mm_data] == [Modality.IMAGE, Modality.AUDIO, Modality.VIDEO]
    assert [(item.start, item.end) for item in mm_data] == [(1, 5), (6, 45), (46, 54)]
    assert mm_data[1].data.shape == (128, 280)
    assert mm_data[1].meta['audio_feature_lengths'].tolist() == [280]
    assert mm_data[1].mrope_pos_ids.shape == (39, 3)
    assert mm_data[1].mrope_pos_ids[0].tolist() == [0, 0, 0]
    assert mm_data[1].mrope_pos_ids[-1].tolist() == [38, 38, 38]
    assert mm_data[2].meta['second_per_grid'] == 2.0
