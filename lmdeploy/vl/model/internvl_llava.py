# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import warnings
from contextlib import contextmanager
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import rewrite_ctx

from .utils import disable_logging, disable_transformers_logging

logger = get_logger('lmdeploy')


def check_llava_install():
    """check llava install."""
    try:
        from llava.model.multimodal_encoder.clip_encoder import \
            InternVisionModel  # noqa: F401
    except ImportError:
        raise ImportError(
            'To use LlavaVLModel, please install llava by '
            'pip install "git+https://github.com/OpenGVLab/InternVL#subdirectory=internvl_chat_llava" --no-deps'  # noqa: E501
        )


def _intern_vision_model__from_pretrained(vision_tower_name: str):
    logger.info(f'init empty InternVisionModel: {vision_tower_name}')
    from llava.model.multimodal_encoder.intern_vit_6b.modeling_intern_vit import (  # noqa: E501
        InternVisionConfig, InternVisionModel)
    config = InternVisionConfig.from_pretrained(vision_tower_name)
    model = InternVisionModel._from_config(config)
    model.requires_grad_(False)
    return model


def _intern_vl_model__from_pretrained(vision_tower_name: str):
    logger.info(f'init empty InternVLModel: {vision_tower_name}')
    from llava.model.multimodal_encoder.internvl_14b.modeling_internvl import (
        InternVLConfig, InternVLModel)
    config = InternVLConfig.from_pretrained(vision_tower_name)
    model = InternVLModel._from_config(config)
    model.requires_grad_(False)
    return model


@contextmanager
def init_empty_vit():
    """skip download vision model if possible."""
    origin_func_path = [
        'llava.model.multimodal_encoder.intern_vit_6b.modeling_intern_vit.InternVisionModel.from_pretrained',  # noqa: E501
        'llava.model.multimodal_encoder.internvl_14b.modeling_internvl.InternVLModel.from_pretrained',  # noqa: E501
    ]
    rewrite_func = [
        _intern_vision_model__from_pretrained,
        _intern_vl_model__from_pretrained
    ]
    with rewrite_ctx(origin_func_path, rewrite_func):
        yield


@VISION_MODELS.register_module()
class InternVLLlavaVisionModel(VisonModel):
    """Llava visual model."""

    @classmethod
    def match(cls, config: AutoConfig):
        """check whether the config match the model."""
        arch = config.architectures[0]
        if arch == 'LlavaLlamaForCausalLM':
            mm_vision_tower = getattr(config, 'mm_vision_tower', '')
            if 'OpenGVLab' in mm_vision_tower:
                return True
        return False

    def build_preprocessor(self):
        check_llava_install()
        # currently, only support llava llama
        from llava.model.language_model.llava_llama import (  # noqa
            LlavaConfig, LlavaLlamaForCausalLM)
        self.config = LlavaConfig.from_pretrained(self.model_path)
        assert self.config.model_type in ['llava', 'llava_llama'], \
            'currently, only support llava llama'
        # init empty model, skip layer initialization
        from accelerate import init_empty_weights
        with init_empty_weights(), warnings.catch_warnings(), \
                disable_transformers_logging():
            warnings.simplefilter('ignore')
            self.config.quantization_config = {
            }  # disable vision part quantization
            self.model = AutoModelForCausalLM.from_config(
                self.config, trust_remote_code=True)
            if not self.with_llm:
                del self.model.lm_head
                del self.model.model.embed_tokens
                del self.model.model.layers
                del self.model.model.norm
            else:
                self.vl_model = self.model

            with init_empty_vit():
                vision_tower = self.model.get_vision_tower()
                vision_tower.is_loaded = False
                vision_tower.load_model()
            crop_size = vision_tower.image_processor.crop_size['height']
            image_size = vision_tower.config.image_size
            patch_size = vision_tower.config.patch_size
            if crop_size != image_size:
                vision_tower.vision_tower.resize_pos_embeddings(
                    image_size, crop_size, patch_size)
                vision_tower.vision_tower.embeddings.image_size = crop_size
                vision_tower.config.image_size = crop_size
                vision_tower.image_processor.crop_size = dict(height=crop_size,
                                                              width=crop_size)
                vision_tower.image_processor.size = dict(
                    shortest_edge=crop_size)
            self.vision_tower = self.model.model.vision_tower.eval()
            self.mm_projector = self.model.model.mm_projector.eval()

    def build_model(self):
        """load weights for vision model."""
        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(
                model=self.model,
                max_memory=self.max_memory,
                checkpoint=self.model_path,
                device_map='auto' if not self.with_llm else {'': 'cpu'},
                no_split_module_classes=['InternVisionEncoderLayer'],
                dtype=torch.half)
        self.model = self.model.model.eval()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """encode images."""
        image_features = self.vision_tower(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """get images and their corresponding preprocess parameters from
        messages, and perform preprocessing."""
        from llava.mm_utils import process_images
        image_processor = self.vision_tower.image_processor
        outputs = []
        for item in messages[-1]['content']:
            item_type = item['type']
            if item_type == 'image':
                image = item['image'].convert('RGB')
                pixel_values = process_images(image, image_processor,
                                              self.config)
                outputs.append(dict(pixel_values=pixel_values))
        return outputs

    @torch.no_grad()
    def forward(self, inputs: List[Dict]) -> List[torch.Tensor]:
        # TODO
        pixel_values = [x['pixel_values'] for x in inputs]
        outputs = torch.stack(pixel_values, dim=0).to(self.vision_tower.device,
                                                      dtype=torch.half)
        outputs = self.encode_images(outputs)

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
