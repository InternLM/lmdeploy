# Copyright (c) OpenMMLab. All rights reserved.

import warnings
from contextlib import contextmanager
from typing import List, Union

import torch
from PIL.Image import Image
from transformers import AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VisonModel
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


class InternVLLlavaVisionModel(VisonModel):
    """Llava visual model."""

    def __init__(self, model_path, with_llm: bool = False):
        self.with_llm = with_llm
        self.model_path = model_path
        # check llava install
        check_llava_install()
        self.build_model()

    def build_model(self):
        """build model & load weights."""

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
            model = AutoModelForCausalLM.from_config(self.config,
                                                     trust_remote_code=True)
            if not self.with_llm:
                del model.lm_head
                del model.model.embed_tokens
                del model.model.layers
                del model.model.norm
            else:
                self.vl_model = model

            with init_empty_vit():
                vision_tower = model.get_vision_tower()
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

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map='auto' if not self.with_llm else {'': 'cpu'},
                no_split_module_classes=['InternVisionEncoderLayer'],
                dtype=torch.half)

        self.model = model.model.eval()
        self.vision_tower = model.model.vision_tower.eval()
        self.mm_projector = model.model.mm_projector.eval()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """encode images."""
        image_features = self.vision_tower(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def preprocess(
            self,
            images: List[Image]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """preprocess."""
        # TODO: gpu processor
        from llava.mm_utils import process_images
        images = [x.convert('RGB') for x in images]
        image_processor = self.vision_tower.image_processor
        outputs = process_images(images, image_processor, self.config)
        return outputs

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        images = self.preprocess(images)
        if isinstance(images, list):
            images = [
                x.to(self.vision_tower.device, dtype=torch.float16)
                for x in images
            ]
        else:
            images = images.to(self.vision_tower.device, dtype=torch.float16)

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)
            image_features = [x for x in image_features]
        return image_features
