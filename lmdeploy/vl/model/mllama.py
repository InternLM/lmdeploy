# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel


@VISION_MODELS.register_module()
class MllamaVLModel(VisonModel):
    """llama3.2 model."""

    _arch = 'MllamaForConditionalGeneration'

    def build_preprocessor(self):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.image_token_id = 128256

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """refer to the spec of `super().preprocess`"""
        images = self.collect_images(messages)
        outputs = []
        for image, params in images:
            image = image.convert('RGB')
            results = self.processor.image_processor(images=image,
                                                     return_tensors='pt')
            results.update(image_size=image.size,
                           image_tokens=1,
                           image_token_id=self.image_token_id)
            outputs.append(results)
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    @staticmethod
    def proc_messages(messages, chat_template, sequence_start):
        """apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<|image|>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['images', 'preprocess', 'forward']:
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

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template,
                                                 sequence_start)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer,
                                   sequence_start)
