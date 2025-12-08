# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from transformers import AutoConfig

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class GLM4VisionModel(VisionModel):
    """Glm-4v-9b vision model."""

    _arch = ['ChatGLMModel', 'ChatGLMForConditionalGeneration']

    @classmethod
    def match(cls, config: AutoConfig):
        """Check whether the config match the model."""
        arch = config.architectures[0] if config.architectures else None
        if arch in cls._arch and hasattr(config, 'vision_config'):
            return True
        return False

    def build_preprocessor(self):
        from torchvision import transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((self.hf_config.vision_config['image_size'], ) * 2,
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        image_size = self.hf_config.vision_config['image_size']
        patch_size = self.hf_config.vision_config['patch_size']
        self.n_token_per_image = 2 + (image_size // patch_size // 2)**2

    def build_model(self):
        if self.with_llm:
            from transformers import AutoModelForCausalLM
            self.vl_model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                                 device_map='cpu',
                                                                 trust_remote_code=True)
        else:
            raise NotImplementedError('turbomind has not supported glm4v yet')

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refers to the spec of `super.preprocess()"""
        outputs = []
        for message in messages:
            if not isinstance(message['content'], List):
                continue
            images = [x['image'] for x in message['content'] if x['type'] == 'image']
            if len(images) > 1:
                logger.warning(f'glm4v does not support the input of multiple images'
                               f' in a single chat round, but got {len(images)} images.')
            # we still pass all the images to the model and let the
            # model decide what to do
            images = [x.convert('RGB') for x in images]
            pixel_values = [self.image_transform(x) for x in images]
            outputs.extend([
                dict(pixel_values=_2,
                     image_size=_1.size,
                     image_tokens=self.n_token_per_image,
                     image_token_id=self.image_token_id) for _1, _2 in zip(images, pixel_values)
            ])
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    @staticmethod
    def proc_messages(messages, chat_template, sequence_start):
        """Apply chat template to get the prompt."""
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

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)
