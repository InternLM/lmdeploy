# Copyright (c) OpenMMLab. All rights reserved.
# Modified from
# https://github.com/haotian-liu/LLaVA.git
import warnings
from contextlib import contextmanager
from typing import Dict, List

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging, rewrite_ctx

logger = get_logger('lmdeploy')


def check_llava_install():
    """check llava install."""
    try:
        import llava  # noqa: F401
    except ImportError:
        raise ImportError(
            'To use LlavaVLModel, please install llava by '
            'pip install git+https://github.com/haotian-liu/LLaVA.git --no-deps'  # noqa: E501
        )


def _clip_vision_tower_load_model(self, **kwargs):
    logger.info(f'CLIPVisionTower.load_model: {self.vision_tower_name}')
    from transformers import (CLIPImageProcessor, CLIPVisionConfig,
                              CLIPVisionModel)
    self.image_processor = CLIPImageProcessor.from_pretrained(
        self.vision_tower_name)
    config = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
    self.vision_tower = CLIPVisionModel._from_config(config=config)
    self.vision_tower.requires_grad_(False)
    self.is_loaded = True


@contextmanager
def init_llava_vision_tower(config):
    """skip download vision model if possible."""
    if getattr(config, 'unfreeze_mm_vision_tower', False):
        origin_func_path = [
            'llava.model.multimodal_encoder.clip_encoder.CLIPVisionTower.load_model'  # noqa: E501
        ]
        rewrite_func = [_clip_vision_tower_load_model]
        with rewrite_ctx(origin_func_path, rewrite_func):
            yield
    else:
        yield


@VISION_MODELS.register_module()
class LlavaVisionModel(VisonModel):
    """Llava visual model."""

    @classmethod
    def match(cls, config: AutoConfig):
        """check whether the config match the model."""
        arch = config.architectures[0]
        if arch in ['LlavaLlamaForCausalLM', 'LlavaMistralForCausalLM']:
            # internvl-llava has vision_tower of OpenGVLab/xxx
            mm_vision_tower = getattr(config, 'mm_vision_tower', '')
            # yi-vl has projector type of xxx_Norm
            projector_type = getattr(config, 'mm_projector_type', 'linear')
            if '_Norm' in projector_type:
                return False
            if 'OpenGVLab' in mm_vision_tower:
                return False
            return True
        return False

    def build_preprocessor(self):
        check_llava_install()

        self.arch = self.hf_config.architectures[0]
        if self.arch == 'LlavaLlamaForCausalLM':
            from llava.model.language_model.llava_llama import LlavaConfig
            self.config = LlavaConfig.from_pretrained(self.model_path)
            assert self.config.model_type in ['llava', 'llava_llama'], \
                f'expect model_type llava and llava_llama '\
                f'but got {self.config.model_type}'
        elif self.arch == 'LlavaMistralForCausalLM':
            from llava.model.language_model.llava_mistral import \
                LlavaMistralConfig
            self.config = LlavaMistralConfig.from_pretrained(self.model_path)
        else:
            assert 0, f'unsupported arch {self.arch}'

        from accelerate import init_empty_weights

        # init empty model, skip layer initialization
        with init_empty_weights(), warnings.catch_warnings(), \
                init_llava_vision_tower(self.config):
            warnings.simplefilter('ignore')
            self.config.quantization_config = {
            }  # disable vision part quantization
            self.model = AutoModelForCausalLM.from_config(
                self.config, trust_remote_code=True)
            self.image_processsor = self.model.model.vision_tower.image_processor  # noqa

    def build_model(self):
        """load vision model's weights."""
        if not self.with_llm:
            # remove the LLM part from llava model.
            # Instead, Load the LLM part to turbomind engine
            del self.model.lm_head
            del self.model.model.embed_tokens
            del self.model.model.layers
            del self.model.model.norm
        else:
            self.vl_model = self.model

        # init empty vision_tower, the embedding layer in CLIPVisionModel
        # can't init right under init_empty_weights
        with init_llava_vision_tower(self.config):
            vision_tower = self.model.get_vision_tower()
            vision_tower.is_loaded = False
            vision_tower.load_model()
            # for llava-v1.5, the vit is not in llm ckpt
            vision_tower.to(dtype=torch.half)

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(
                model=self.model,
                max_memory=self.max_memory,
                checkpoint=self.model_path,
                device_map='auto' if not self.with_llm else {'': 'cpu'},
                no_split_module_classes=['CLIPEncoderLayer'],
                dtype=torch.half)

        self.model = self.model.model.eval()
        self.vision_tower = self.model.model.vision_tower.half().eval()
        self.mm_projector = self.model.model.mm_projector.half().eval()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """encode images."""
        image_features = self.vision_tower(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """get images and their corresponding preprocess parameters from
        messages, and perform preprocessing."""
        from llava.mm_utils import process_images
        outputs = []
        for item in messages[-1]['content']:
            item_type = item['type']
            if item_type == 'image':
                image = item['image'].convert('RGB')
                pixel_values = process_images(image, self.image_processor,
                                              self.config)
                outputs.append(
                    dict(pixel_values=pixel_values,
                         image_tokens=1,
                         image_token_id=1,
                         image_size=image.size))
        return outputs

    @torch.no_grad()
    def forward(self, inputs: List[Dict]) -> List[torch.Tensor]:
        image_sizes = [x['image_size'] for x in inputs]
        pixel_values = [
            x['pixel_values'].to(device=self.vision_tower.device,
                                 dtype=torch.float16) for x in inputs
        ]
        pixel_values = torch.cat(pixel_values, dim=0)
        if pixel_values.ndim == 5:
            from llava.model.llava_arch import (get_anyres_image_grid_shape,
                                                unpad_image)
            image_features = self.encode_images(pixel_values)
            image_features = torch.split(image_features, 1, dim=0)
            logger.error(f'image feature size: {image_features.shape}')
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type',
                                          'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio',
                                         'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.vision_tower.num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = \
                                get_anyres_image_grid_shape(
                                    image_sizes[image_idx],
                                    self.config.image_grid_pinpoints,
                                    self.vision_tower.config.image_size)
                            image_feature = image_feature.view(
                                num_patch_height, num_patch_width, height,
                                width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1,
                                                                  2).flatten(
                                                                      2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(
                                    *image_feature.shape[:-1], 1).to(
                                        image_feature.device)),
                                                      dim=-1)
                            image_feature = image_feature.flatten(1,
                                                                  2).transpose(
                                                                      0, 1)
                        else:
                            image_feature = image_feature.permute(
                                0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat(
                            (base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (image_feature,
                                 self.model.image_newline[None].to(
                                     image_feature.device)),
                                dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError('Unexpected mm_patch_merge_type: '
                                 f'{self.config.mm_patch_merge_type}')
        else:
            image_features = self.encode_images(pixel_values)
            image_features = torch.split(image_features, 1, dim=0)
            image_features = [x.squeeze() for x in image_features]
        return image_features

    @classmethod
    def proc_messages(cls, messages, chat_template, sequence_start):
        """apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            n_images = [
                1 for item in message['content'] if item['type'] == 'image'
            ]
            n_images = sum(n_images)
            content = [
                item['text'] for item in message['content']
                if item['type'] == 'text'
            ]
            content = f'<img>{IMAGE_TOKEN * n_images}</img>\n' + content[0]
            prompt_messages.append(dict(role='user', content=content))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template,
                                                 sequence_start)
        return super().to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer,
                                      sequence_start)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template,
                                                 sequence_start)
        return super().to_turbomind_aux(messages, prompt, IMAGE_TOKEN,
                                        tokenizer, sequence_start)
