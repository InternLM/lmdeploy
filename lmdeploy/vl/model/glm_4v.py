# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List

import torch
from transformers import AutoConfig

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging

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

    def build_model(self):
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from accelerate.utils import infer_auto_device_map

        with init_empty_weights(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_config(
                self.hf_config, trust_remote_code=True)
            if not self.with_llm:
                del self.model.transformer.embedding
                del self.model.transformer.rotary_pos_emb
                del self.model.transformer.encoder
                del self.model.transformer.output_layer
            else:
                self.vl_model = self.model

        no_split_module_classes = ['TransformerLayer']

        device_map = infer_auto_device_map(
            self.model,
            no_split_module_classes=no_split_module_classes,
            max_memory=self.max_memory,
            dtype=torch.half)

        same_device_keys = [
            ('transformer.vision.linear_proj', 'transformer.vision.boi',
             'transformer.vision.eoi')
        ]
        for keys in same_device_keys:
            keys = [k for k in keys if k in device_map]
            if len(keys) <= 1:
                continue
            for k in keys[1:]:
                device_map[k] = device_map[keys[0]]

        with disable_logging():
            load_checkpoint_and_dispatch(
                model=self.model,
                checkpoint=self.model_path,
                device_map=device_map if not self.with_llm else {'': 'cpu'},
                no_split_module_classes=no_split_module_classes,
                dtype=torch.half)
        self.model.eval()

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """get images and their corresponding preprocess parameters from
        messages, and perform preprocessing."""
        images = [x['content'] for x in messages if x['role'] == 'images']
        assert len(images) == 1
        images = images[0]
        outputs = []
        for image, params in images:
            image = image.convert('RGB')
            pixel_values = self.image_transform(image)
            outputs.append(
                dict(
                    pixel_values=pixel_values,
                    image_size=image.size,
                    image_tokens=1602,  # TODO
                    image_token_id=0))
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    @torch.no_grad()
    def forward(self, inputs: List[Dict]) -> List[torch.Tensor]:
        assert 0, 'glm4v is not supported by turbomind'

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
            elif message['role'] in ['images', 'preprocess', 'forward']:
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

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start):
        assert 0, 'glm4v is not supported by turbomind'
