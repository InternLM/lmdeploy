# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel


def check_transformers():
    try:
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError('please install latest transformers by '
                          'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class Qwen3VLModel(VisonModel):
    """Qwen3VL model."""

    _arch = ['Qwen3VLForConditionalGeneration', 'Qwen3VLMoeForConditionalGeneration']

    def build_preprocessor(self):
        check_transformers()
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        tokenizer = self.processor.tokenizer
        self.image_token = self.processor.image_token
        self.image_token_id = tokenizer.encode(self.image_token)[-1]

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refer to `super().preprocess()` for spec."""
        images = self.collect_images(messages)
        optional_keys = {'resized_height', 'resized_width', 'min_pixels', 'max_pixels'}
        outputs = []
        for image, params in images:
            image = image.convert('RGB')

            item = dict(type='image', image=image)
            item.update({key: params[key] for key in params.keys() if key in optional_keys})
            result = self.processor.image_processor(images=image, videos=None, return_tensors='pt')
            merge_length = self.processor.image_processor.merge_size**2
            image_tokens = result['image_grid_thw'].prod(dim=1) // merge_length
            result.update(dict(image_size=image.size, image_tokens=image_tokens, image_token_id=self.image_token_id))
            outputs.append(result)
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    def proc_messages(self, messages, chat_template, sequence_start, add_vision_id: Optional[bool] = False):
        """Apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        messages = [x for x in messages if x['role'] not in ['preprocess', 'forward']]
        if VisonModel.IMAGE_TOKEN_included(messages):
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
            if add_vision_id:
                prompt_messages = messages
            else:
                for message in messages:
                    role, content = message['role'], message['content']
                    if role != 'user' or isinstance(content, str):
                        prompt_messages.append(message)
                        continue
                    _content = []
                    for item in content:
                        if item['type'] == 'text':
                            _content.append(item['text'])
                        elif item['type'] in ['image', 'image_url']:
                            _content.append(f'<|vision_start|>{self.image_token}<|vision_end|>')
                        else:
                            raise ValueError(f'Unsupported message type: {item["type"]}')
                    message = dict(role=role, content=''.join(_content))
                    prompt_messages.append(message)
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start, add_vision_id=add_vision_id)
        return prompt, self.image_token

    def to_pytorch(self,
                   messages,
                   chat_template,
                   tokenizer,
                   sequence_start,
                   add_vision_id: Optional[bool] = False,
                   **kwargs):
        """Return to the information needed by pytorch engine."""
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start, add_vision_id)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

    def build_model(self):
        # TODO: implement for turbomind
        pass

    @torch.no_grad()
    def forward(self, messages: List[Dict], max_batch_size: int = 1) -> List[Dict]:
        # TODO: implement for turbomind
        pass

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        # TODO: implement for turbomind
        pass
