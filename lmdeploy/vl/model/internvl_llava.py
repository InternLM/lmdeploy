# Copyright (c) OpenMMLab. All rights reserved.

import warnings
from contextlib import contextmanager
from typing import Dict, List

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.llava import VISION_MODELS, LlavaVisionModel
from lmdeploy.vl.model.utils import rewrite_ctx

from .utils import disable_logging, disable_transformers_logging

logger = get_logger('lmdeploy')


def check_llava_install():
    try:
        from llava.model.multimodal_encoder.clip_encoder import InternVisionModel  # noqa: F401
    except ImportError:
        raise ImportError(
            'To use LlavaVLModel, please install llava by '
            '`pip install git+https://github.com/OpenGVLab/InternVL#subdirectory=internvl_chat_llava --no-deps`')


def _intern_vision_model__from_pretrained(vision_tower_name: str):
    logger.info(f'init empty InternVisionModel: {vision_tower_name}')
    from llava.model.multimodal_encoder.intern_vit_6b.modeling_intern_vit import InternVisionConfig, InternVisionModel
    config = InternVisionConfig.from_pretrained(vision_tower_name)
    model = InternVisionModel._from_config(config)
    model.requires_grad_(False)
    return model


def _intern_vl_model__from_pretrained(vision_tower_name: str):
    logger.info(f'init empty InternVLModel: {vision_tower_name}')

    from llava.model.multimodal_encoder.internvl_14b.modeling_internvl import InternVLConfig, InternVLModel

    config = InternVLConfig.from_pretrained(vision_tower_name)
    model = InternVLModel._from_config(config)
    model.requires_grad_(False)
    return model


@contextmanager
def init_empty_vit():
    """Skip download vision model if possible."""
    origin_func_path = [
        'llava.model.multimodal_encoder.intern_vit_6b.modeling_intern_vit.InternVisionModel.from_pretrained',  # noqa: E501
        'llava.model.multimodal_encoder.internvl_14b.modeling_internvl.InternVLModel.from_pretrained',  # noqa: E501
    ]
    rewrite_func = [_intern_vision_model__from_pretrained, _intern_vl_model__from_pretrained]
    with rewrite_ctx(origin_func_path, rewrite_func):
        yield


@VISION_MODELS.register_module()
class InternVLLlavaVisionModel(LlavaVisionModel):
    """Llava visual model."""

    @classmethod
    def match(cls, config: AutoConfig):
        """Check whether the config match the model."""
        arch = config.architectures[0] if config.architectures else None
        if arch == 'LlavaLlamaForCausalLM':
            mm_vision_tower = getattr(config, 'mm_vision_tower', '')
            if 'OpenGVLab' in mm_vision_tower:
                return True
        return False

    def build_preprocessor(self):
        return super().build_preprocessor()

    def build_model(self):
        """Build the vision part of a VLM model when backend is turbomind, or
        load the whole VLM model when `self.with_llm==True`"""
        check_llava_install()
        # currently, only support llava llama
        from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM  # noqa
        self.config = LlavaConfig.from_pretrained(self.model_path)
        assert self.config.model_type in ['llava', 'llava_llama'], \
            'currently, only support llava llama'

        # init empty model, skip layer initialization
        from accelerate import init_empty_weights
        with init_empty_weights(), warnings.catch_warnings(), \
                disable_transformers_logging():
            warnings.simplefilter('ignore')
            self.config.quantization_config = {}  # disable vision part quantization
            model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
            self.vl_model = model
            if not self.with_llm:
                del model.lm_head
                del model.model.embed_tokens
                del model.model.layers
                del model.model.norm

            with init_empty_vit():
                vision_tower = model.get_vision_tower()
                vision_tower.is_loaded = False
                vision_tower.load_model()
            crop_size = vision_tower.image_processor.crop_size['height']
            image_size = vision_tower.config.image_size
            patch_size = vision_tower.config.patch_size
            if crop_size != image_size:
                vision_tower.vision_tower.resize_pos_embeddings(image_size, crop_size, patch_size)
                vision_tower.vision_tower.embeddings.image_size = crop_size
                vision_tower.config.image_size = crop_size
                vision_tower.image_processor.crop_size = dict(height=crop_size, width=crop_size)
                vision_tower.image_processor.size = dict(shortest_edge=crop_size)

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(model=model,
                                         max_memory=self.max_memory,
                                         checkpoint=self.model_path,
                                         device_map='auto' if not self.with_llm else {'': 'cpu'},
                                         no_split_module_classes=['InternVisionEncoderLayer'],
                                         dtype=torch.half)

        self.model = model.model.eval()
        self.vision_tower = model.model.vision_tower.eval()
        self.mm_projector = model.model.mm_projector.eval()

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refer to `super().preprocess() for spec."""
        return super().preprocess(messages)

    @torch.no_grad()
    def forward(self, messages: List[Dict], max_batch_size: int = 1) -> List[Dict]:
        """Extract image feature. ONLY implement it when the backend is
        turbomind engine.

        Args:
            messages(List[Dict]): the outputs of `preprocess`
            max_batch_size(int): the max batch size when forwarding vision
                model
        Return:
            the message list with forwarding results included
        """
        inputs = [x['content'] for x in messages if x['role'] == 'preprocess']
        inputs = inputs[0]
        outputs = []
        for idx in range(0, len(inputs), max_batch_size):
            pixel_values = [x['pixel_values'] for x in inputs[idx:idx + max_batch_size]]
            split_sizes = [x.shape[0] for x in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            pixel_values = pixel_values.to(device=self.vision_tower.device, dtype=torch.float16)
            logger.info(f'vision forward shape: {pixel_values.shape}')
            if pixel_values.ndim == 5:
                feats = self.encode_images(pixel_values)
                feats = torch.split(feats, split_sizes, dim=0)
                feats = [x.flatten(0, 1) for x in feats]
            else:
                feats = self.encode_images(pixel_values)
                feats = [x for x in feats]
            outputs.extend(feats)
        messages.append(dict(role='forward', content=outputs))
        return messages
