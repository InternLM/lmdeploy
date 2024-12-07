# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from transformers import AutoConfig

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class GLM4VisionModel(VisonModel):
    """glm-4v-9b vision model."""

    _arch = 'ChatGLMModel'

    @classmethod
    def match(cls, config: AutoConfig):
        """check whether the config match the model."""
        arch = config.architectures[0]
        if arch == cls._arch and hasattr(config, 'vision_config'):
            return True
        return False

    def build_preprocessor(self):
        from torchvision import transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(
                (self.hf_config.vision_config['image_size'], ) * 2,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        image_size = self.hf_config.vision_config['image_size']
        patch_size = self.hf_config.vision_config['patch_size']
        self.n_token_per_image = 2 + (image_size // patch_size // 2)**2

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """refers to the spec of `super.preprocess()"""
        images = self.collect_images(messages)
        outputs = []
        for image, params in images:
            image = image.convert('RGB')
            pixel_values = self.image_transform(image)
            outputs.append(
                dict(pixel_values=pixel_values,
                     image_size=image.size,
                     image_tokens=self.n_token_per_image,
                     image_token_id=0))
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    @classmethod
    def proc_messages(cls, messages, chat_template, sequence_start):
        """apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            content = message['content']
            if isinstance(content, str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['preprocess', 'forward']:
                continue
            prompt = [x['text'] for x in content if x['type'] == 'text']
            n_images = len([1 for x in content if x['type'] == 'image'])
            prompt = ''.join([f'{IMAGE_TOKEN}\n'] * n_images) + prompt[0]
            prompt_messages.append(dict(role='user', content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template,
                                                 sequence_start)
        return super().to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer,
                                      sequence_start)
