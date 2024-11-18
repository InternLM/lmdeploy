# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import warnings
from typing import Dict, List

import numpy as np
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
        from accelerate import init_empty_weights

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
        from accelerate import load_checkpoint_and_dispatch
        from accelerate.utils import infer_auto_device_map

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
        pixel_values = [x['pixel_values'] for x in inputs]
        outputs = torch.stack(pixel_values, dim=0).to(device='cuda:0',
                                                      dtype=torch.half)
        outputs = self.model.transformer.vision(outputs)
        outputs = torch.split(outputs, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs

    @classmethod
    def proc_messages(cls, messages, chat_template, sequence_start):
        # apply chat template to get the prompt
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            content = message['content']
            if isinstance(content, str):
                prompt_messages.append(message)
                continue

            prompt = [x['text'] for x in content if x['type'] == 'text']
            n_images = len([1 for x in content if x['type'] == 'image'])
            prompt = ''.join([f'{IMAGE_TOKEN}\n'] * n_images) + prompt[0]
            prompt_messages.append(dict(role='user', content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        segs = prompt.split(IMAGE_TOKEN)

        # collect all preprocessing result from messages
        preps = [
            message.pop('preprocess') for message in messages
            if 'preprocess' in message.keys()
        ]
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

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start):
        prompt, segs, features = self.proc_messages(messages, chat_template,
                                                    sequence_start)
        features = [x.cpu().numpy() for x in features]

        # tokenizer prompt, and get input_embeddings and input_embedding_ranges
        input_ids = []
        begins = []
        ends = []
        IMAGE_DUMMY_TOKEN_INDEX = 0
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(features):
                image_dim = features[i - 1].shape[0]
                begins.append(len(input_ids))
                ends.append(begins[-1] + image_dim)
                input_ids.extend([IMAGE_DUMMY_TOKEN_INDEX] * image_dim)
            seg_ids = tokenizer.encode(seg,
                                       add_bos=((i == 0) and sequence_start))
            input_ids.extend(seg_ids)
        ranges = np.stack([begins, ends], axis=1).tolist()
        return dict(prompt=prompt,
                    input_ids=input_ids,
                    input_embeddings=features,
                    input_embedding_ranges=ranges)
