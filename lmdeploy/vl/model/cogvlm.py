# Copyright (c) OpenMMLab. All rights reserved.

import itertools
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

        from accelerate import init_empty_weights
        with init_empty_weights(), warnings.catch_warnings():
            self.model = AutoModelForCausalLM.from_config(
                self.hf_config, trust_remote_code=True)
            if not self.with_llm:
                del self.model.lm_head
                for key in ['layers', 'norm', 'embed_tokens']:
                    setattr(self.model.model, key, None)
            else:
                self.vl_model = self.model

    def build_model(self):
        from accelerate import load_checkpoint_and_dispatch
        from accelerate.utils import get_balanced_memory, infer_auto_device_map
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
        """get images and their corresponding preprocess parameters from
        messages, and perform preprocessing."""
        outputs = []
        for item in messages[-1]['content']:
            item_type = item['type']
            if item_type == 'image':
                image = item['image'].convert('RGB')
                pixel_values = self.image_transform(image)
                outputs.append(dict(pixel_values=pixel_values))
        return outputs

    @torch.no_grad()
    def forward(self, inputs: List[Dict]) -> List[torch.Tensor]:
        assert 0, 'cogvlm is not supported by turbomind'

    @classmethod
    def proc_messages(cls, messages, chat_template, sequence_start):
        # apply chat template to get the prompt
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            content = [
                x['text'] for x in message['content'] if x['type'] == 'text'
            ]
            content = content[0]
            # n_images = len(
            #     [1 for x in message['content'] if x['type'] == 'image'])
            # TODO
            # prompt_messages.append(dict(role='user', content=content))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)

        # collect all preprocessing result from messages
        preps = [
            message.pop('preprocess') for message in messages
            if 'preprocess' in message.keys()
        ]
        segs = prompt.split(IMAGE_TOKEN)
        # flatten the list
        preps = list(itertools.chain(*preps))
        assert len(segs) == len(preps) + 1, (
            f'the number of {IMAGE_TOKEN} is not equal '
            f'to input images, {len(segs) - 1} vs {len(preps)}')

        return prompt, segs, preps

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        prompt, segs, preps = self.proc_messages(messages, chat_template,
                                                 sequence_start)

        # calculate the image token offset for each image
        input_ids = []
        IMAGE_DUMMY_TOKEN_INDEX = 0
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(preps):
                preps[i - 1].update(offset=len(input_ids))
                image_tokens = 0  # TODO
                input_ids.extend([IMAGE_DUMMY_TOKEN_INDEX] * image_tokens)
            token_ids = tokenizer.encode(seg,
                                         add_bos=((i == 0) and sequence_start))
            input_ids.extend(token_ids)

        return dict(prompt=prompt, input_ids=input_ids, multimodal=preps)

    def to_turbomind(self, messages, chat_template, sequence_start):
        assert 0, 'cogvlm is not supported by turbomind'
