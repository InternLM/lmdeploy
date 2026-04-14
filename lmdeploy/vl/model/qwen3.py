# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

from transformers import AutoProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, MultomodalSpecialTokens, VisionModel

logger = get_logger('lmdeploy')


def check_transformers():
    try:
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError('please install latest transformers by '
                          'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class Qwen3VLModel(VisionModel):
    """Qwen3VL model."""

    _arch = ['Qwen3VLForConditionalGeneration', 'Qwen3VLMoeForConditionalGeneration']

    def build_preprocessor(self):
        check_transformers()
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

        # image tokens
        self.image_token = self.processor.image_token
        self.image_token_id = self.processor.image_token_id

        # video tokens
        self.video_token = self.processor.video_token
        self.video_token_id = self.processor.video_token_id

        # vision start and end tokens
        self.vision_start_token = self.processor.vision_start_token
        self.vision_end_token = self.processor.vision_end_token

        # special tokens
        self.mm_tokens = MultomodalSpecialTokens(
            image_token=self.image_token,
            video_token=self.video_token,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
        )

    def resolve_size_params(self, processor, mm_processor_kwargs: dict[str, Any] | None = None):
        default_min = processor.size['shortest_edge']
        default_max = processor.size['longest_edge']

        if not mm_processor_kwargs:
            return {'shortest_edge': default_min, 'longest_edge': default_max}

        min_pixels = mm_processor_kwargs.get('min_pixels', default_min)
        max_pixels = mm_processor_kwargs.get('max_pixels', default_max)

        if min_pixels > max_pixels:
            logger.warning(f'min_pixels {min_pixels} > max_pixels {max_pixels}, falling back to defaults.')
            return {'shortest_edge': default_min, 'longest_edge': default_max}

        return {'shortest_edge': min_pixels, 'longest_edge': max_pixels}

    def apply_chat_template(self, messages, chat_template, sequence_start, chat_template_kwargs=None):
        """Apply chat template to get the prompt."""
        chat_template_kwargs = chat_template_kwargs or {}
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        messages = [x for x in messages if x['role'] not in ['preprocess', 'forward']]
        if VisionModel.IMAGE_TOKEN_included(messages):
            # backward compatibility
            for message in messages:
                role, content = message['role'], message['content']
                if role != 'user' or isinstance(content, str):
                    prompt_messages.append(message)
                    continue
                content = [x['text'] for x in content if x['type'] == 'text']
                prompt = ''.join(content)
                prompt = prompt.replace(IMAGE_TOKEN, f'<|vision_start|>{self.image_token}<|vision_end|>')
                prompt_messages.append(dict(role='user', content=prompt))
        else:
            prompt_messages = messages

        prompt = chat_template.messages2prompt(prompt_messages, sequence_start, **chat_template_kwargs)
        return prompt
