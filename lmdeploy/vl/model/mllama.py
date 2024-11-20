# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

import torch

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel


@VISION_MODELS.register_module()
class MllamaVLModel(VisonModel):
    """llama3.2 model."""

    _arch = 'MllamaForConditionalGeneration'

    def build_preprocessor(self):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.image_token_id = 128256

    def build_model(self):
        assert 0, 'cogvlm is not supported by turbomind'

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """refer to the spec of `super().preprocess`"""
        outputs = []
        for item in messages[-1]['content']:
            item_type = item['type']
            if item_type == 'image':
                image = item['image'].convert('RGB')
                results = self.processor.image_processor(images=image,
                                                         return_tensors='pt')
                results.update(image_size=image.size,
                               image_tokens=1,
                               image_token_id=self.image_token_id)
                outputs.append(results)
        return outputs

    @classmethod
    def proc_messages(cls, messages, chat_template, sequence_start):
        """apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<|image|>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            n_images = len(
                [1 for x in message['content'] if x['type'] == 'image'])
            content = [
                item['text'] for item in message['content']
                if item['type'] == 'text'
            ]
            prompt = (IMAGE_TOKEN) * n_images + content[0]
            prompt_messages.append(dict(role='user', content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    @torch.no_grad()
    def forward(self, inputs: List[Dict]) -> List[torch.Tensor]:
        assert 0, 'cogvlm is not supported by turbomind'

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template,
                                                 sequence_start)
        return super().to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer,
                                      sequence_start)

    def to_turbomind(self, messages, chat_template, sequence_start):
        assert 0, 'cogvlm is not supported by turbomind'
