# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class CogVLMVisionModel(VisonModel):
    """CogVLM vision model."""

    _arch = 'CogVLMForCausalLM'

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
        from accelerate.utils import get_balanced_memory, infer_auto_device_map
        with init_empty_weights(), warnings.catch_warnings():
            self.model = AutoModelForCausalLM.from_config(
                self.hf_config, trust_remote_code=True)
            if not self.with_llm:
                del self.model.lm_head
                for key in ['layers', 'norm', 'embed_tokens']:
                    setattr(self.model.model, key, None)
            else:
                self.vl_model = self.model

        no_split_module_classes = ['TransformerLayer']
        max_memory = get_balanced_memory(
            self.model,
            max_memory=self.max_memory,
            dtype=torch.half,
            no_split_module_classes=no_split_module_classes)
        device_map = infer_auto_device_map(
            self.model,
            no_split_module_classes=no_split_module_classes,
            max_memory=max_memory,
            dtype=torch.half)
        same_device_keys = [('model.vision.linear_proj', 'model.vision.boi',
                             'model.vision.eoi')]
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
        self.model = self.model.model.vision
        self.model.eval()

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """refer to the spec of `super().preprocess`"""
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
                    image_tokens=2306,  # TODO
                    image_token_id=0))
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    @torch.no_grad()
    def forward(self, inputs: List[Dict]) -> List[torch.Tensor]:
        assert 0, 'cogvlm is not supported by turbomind'

    @classmethod
    def proc_messages(cls, messages, chat_template, sequence_start):
        """apply chat template to get the prompt."""
        prompt_messages = []
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['images', 'preprocess', 'forward']:
                continue
            content = [
                x['text'] for x in message['content'] if x['type'] == 'text'
            ]
            n_images = len(
                [1 for x in message['content'] if x['type'] == 'image'])

            prompt_messages.append(
                dict(role='user', content=content[0], num_images=n_images))

        from lmdeploy.model import Vicuna
        llm_chat_template = Vicuna(eoa=chat_template.eoa,
                                   stop_words=chat_template.stop_words)
        prompt = ''
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for i, msg in enumerate(prompt_messages):
            num_images = msg.pop('num_images', 0)
            if num_images == 0:
                role = msg['role']
                msg = llm_chat_template.messages2prompt([msg], sequence_start
                                                        and i == 0)
                msg = dict(role=role, content=msg)
            prompt_i = chat_template.messages2prompt([msg], sequence_start
                                                     and i == 0)
            if num_images > 0:
                prompt_i = (IMAGE_TOKEN * num_images) + prompt_i
            prompt += prompt_i
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template,
                                                 sequence_start)
        return super().to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer,
                                      sequence_start)

    def to_turbomind(self, messages, chat_template, sequence_start):
        assert 0, 'cogvlm is not supported by turbomind'
