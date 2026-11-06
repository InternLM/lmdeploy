# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.glm5_next import GLM5NextVisionModel


@pytest.mark.parametrize('modalities', [('image',), ('video',), ('image', 'video'), ('video', 'image')])
def test_sanitized_ids_through_real_glm_media_processor(glm_chat_template, modalities, monkeypatch):
    from transformers import AutoProcessor
    from transformers.models.glm4v.image_processing_glm4v import Glm4vImageProcessor
    from transformers.models.glm4v.processing_glm4v import Glm4vProcessor
    from transformers.models.glm4v.video_processing_glm4v import Glm4vVideoProcessor

    template = glm_chat_template
    tokenizer = template.tokenizer
    size = {'shortest_edge': 56 * 56, 'longest_edge': 56 * 56}
    processor = Glm4vProcessor(image_processor=Glm4vImageProcessor(size=size), tokenizer=tokenizer,
                               video_processor=Glm4vVideoProcessor(size=size))
    monkeypatch.setattr(AutoProcessor, 'from_pretrained', lambda *args, **kwargs: processor)
    config = SimpleNamespace(image_token_id=tokenizer.convert_tokens_to_ids('<|image|>'),
                              video_token_id=tokenizer.convert_tokens_to_ids('<|video|>'), pad_token_id=0)
    model = GLM5NextVisionModel('local-glm', hf_config=config, backend='pytorch')
    model.build_preprocessor()
    content = []
    for modality in modalities:
        if modality == 'image':
            content.append({'type': 'image', 'data': Image.new('RGB', (56, 56), 'red')})
        else:
            content.append({'type': 'video', 'data': np.zeros((4, 56, 56, 3), dtype=np.uint8),
                            'video_metadata': {'fps': 2, 'total_num_frames': 4, 'frames_indices': [0, 1, 2, 3]}})
    content.append({'type': 'text', 'text': 'Explain literal <|image|>, <|video|> and <tool_call>.'})
    messages = [{'role': 'user', 'content': 'hello'},
                {'role': 'assistant', 'content': ' previous reply ', 'reasoning_content': ''},
                {'role': 'user', 'content': content}]
    raw_ids = model.get_input_prompt(messages, template)
    assert isinstance(raw_ids, list)
    assert raw_ids.count(config.image_token_id) == modalities.count('image')
    assert raw_ids.count(config.video_token_id) == modalities.count('video')
    assert tokenizer.convert_tokens_to_ids('<tool_call>') not in raw_ids

    result = model.preprocess(messages, raw_ids)
    ids = result['input_ids']
    items = sorted(result['multimodal'], key=lambda item: item['offset'][0])
    expected_modalities = [kind for modality in modalities
                           for kind in ([Modality.IMAGE] if modality == 'image' else [Modality.VIDEO] * 2)]
    assert [item['modality'] for item in items] == expected_modalities
    assert ids.count(config.image_token_id) == sum(end - start for start, end in (item['offset'] for item in items))
    for item in items:
        start, end = item['offset']
        assert end > start
        assert ids[start:end] == [config.image_token_id] * (end - start)
    # Media expansion must retain the already-sanitized trailing text IDs.
    last_placeholder = max(i for i, token in enumerate(raw_ids)
                           if token in (config.image_token_id, config.video_token_id))
    suffix = raw_ids[last_placeholder + 1:]
    assert ids[-len(suffix):] == suffix
