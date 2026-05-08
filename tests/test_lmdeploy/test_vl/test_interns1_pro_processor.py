from types import SimpleNamespace

import torch

from lmdeploy.pytorch.models.interns1_pro import InternS1ProForConditionalGeneration
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
from lmdeploy.vl.constants import Modality


def test_interns1_pro_prepare_uses_separate_visual_and_ts_masks():
    model = InternS1ProForConditionalGeneration.__new__(InternS1ProForConditionalGeneration)
    model.visual = SimpleNamespace(
        rot_pos_emb=lambda grid_thw: torch.ones(1, 2),
        fast_pos_embed_interpolate=lambda grid_thw: torch.ones(1, 2),
    )
    input_ids = torch.tensor([1, 10, 10, 10, 10, 13, 13])
    image = MultiModalData(
        modality=Modality.IMAGE,
        data=torch.ones(16, 3),
        start=1,
        end=5,
        meta={
            'grid_thw': torch.tensor([1, 4, 4]),
            'image_token_id': 10,
        },
    )
    time_series = MultiModalData(
        modality=Modality.TIME_SERIES,
        data=torch.ones(1, 4, 2),
        start=5,
        end=7,
        meta={
            'ts_lens': torch.tensor([4]),
            'ts_sr': torch.tensor([100.0]),
            'ts_token_id': 13,
        },
    )
    context = SimpleNamespace(
        input_ids=input_ids,
        position_ids=torch.arange(input_ids.numel()),
        attn_metadata=None,
        input_multimodals=[{
            'mm_data': [image, time_series],
        }],
        input_embeddings=None,
        input_embedding_indexing=None,
    )

    prepared = model.prepare_inputs_for_generation(past_key_values=[], context=context)

    assert prepared['image_mask'].tolist() == [False, True, True, True, True, False, False]
    assert prepared['ts_mask'].tolist() == [False, False, False, False, False, True, True]
    assert prepared['pixel_values'].shape == (16, 3)
    assert prepared['ts_values'].shape == (1, 4, 2)
    assert prepared['ts_lens'].tolist() == [4]
    assert prepared['ts_sr'].tolist() == [100.0]
