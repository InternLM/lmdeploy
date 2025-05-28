# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/haotian-liu/LLaVA.git

import ast
import math
import warnings
from contextlib import contextmanager
from typing import Dict, List

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.llava_hf import VISION_MODELS, LlavaHfVisionModel
from lmdeploy.vl.model.utils import disable_logging, rewrite_ctx

logger = get_logger('lmdeploy')


def check_llava_install():
    """Check llava install."""
    try:
        import llava  # noqa: F401
    except ImportError:
        raise ImportError('To use LlavaVLModel, please install llava by '
                          '`pip install git+https://github.com/haotian-liu/LLaVA.git --no-deps`'  # noqa: E501
                          )


def _clip_vision_tower_load_model(self, **kwargs):
    logger.info(f'CLIPVisionTower.load_model: {self.vision_tower_name}')
    from transformers import CLIPVisionConfig, CLIPVisionModel

    config = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
    self.vision_tower = CLIPVisionModel._from_config(config=config)
    self.vision_tower.requires_grad_(False)
    self.is_loaded = True


@contextmanager
def init_llava_vision_tower(config):
    """Skip download vision model if possible."""
    if getattr(config, 'unfreeze_mm_vision_tower', False):
        origin_func_path = [
            'llava.model.multimodal_encoder.clip_encoder.CLIPVisionTower.load_model'  # noqa: E501
        ]
        rewrite_func = [_clip_vision_tower_load_model]
        with rewrite_ctx(origin_func_path, rewrite_func):
            yield
    else:
        yield


def select_best_resolution(original_size, possible_resolutions):
    """Selects the best resolution from a list of possible resolutions based on
    the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """  # noqa
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution
                                                               and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """Resize and pad an image to a target resolution while maintaining aspect
    ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """  # noqa
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def process_anyres_image(image, processor, grid_pinpoints):
    """Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """  # noqa
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [
        processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0] for image_patch in image_patches
    ]
    return torch.stack(image_patches, dim=0)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, 'image_aspect_ratio', None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == 'anyres':
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


@VISION_MODELS.register_module()
class LlavaVisionModel(LlavaHfVisionModel):
    """Llava visual model."""

    @classmethod
    def match(cls, config: AutoConfig):
        """Check whether the config match the model."""
        arch = config.architectures[0] if config.architectures else None
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
        from transformers import CLIPImageProcessor
        self.image_processor = CLIPImageProcessor.from_pretrained(self.hf_config.mm_vision_tower)
        config = AutoConfig.from_pretrained(self.hf_config.mm_vision_tower)
        image_size = config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.n_token_per_image = (image_size // patch_size)**2
        if self.hf_config.mm_vision_select_feature == 'cls_patch':
            self.n_token_per_image += 1

    def build_model(self):
        """Build the vision part of a VLM model when backend is turbomind, or
        load the whole VLM model when `self.with_llm==True`"""
        check_llava_install()

        self.arch = self.hf_config.architectures[0]
        model = None
        if self.arch == 'LlavaLlamaForCausalLM':
            from llava.model.language_model.llava_llama import LlavaConfig
            self.config = LlavaConfig.from_pretrained(self.model_path)
            assert self.config.model_type in ['llava', 'llava_llama'], \
                f'expect model_type llava and llava_llama '\
                f'but got {self.config.model_type}'
        elif self.arch == 'LlavaMistralForCausalLM':
            from llava.model.language_model.llava_mistral import LlavaMistralConfig
            self.config = LlavaMistralConfig.from_pretrained(self.model_path)
        else:
            assert 0, f'unsupported arch {self.arch}'

        from accelerate import init_empty_weights

        # init empty model, skip layer initialization
        with init_empty_weights(), warnings.catch_warnings(), \
                init_llava_vision_tower(self.config):
            warnings.simplefilter('ignore')
            self.config.quantization_config = {}  # disable vision part quantization
            model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)

        self.vl_model = model
        if not self.with_llm:
            # remove the LLM part from llava model.
            del model.lm_head
            del model.model.embed_tokens
            del model.model.layers
            del model.model.norm

        # init empty vision_tower, the embedding layer in CLIPVisionModel
        # can't init right under init_empty_weights
        with init_llava_vision_tower(self.config):
            vision_tower = model.get_vision_tower()
            vision_tower.is_loaded = False
            vision_tower.load_model()
            # for llava-v1.5, the vit is not in llm ckpt
            vision_tower.to(dtype=torch.half)

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(model=model,
                                         max_memory=self.max_memory,
                                         checkpoint=self.model_path,
                                         device_map='auto' if not self.with_llm else {'': 'cpu'},
                                         no_split_module_classes=['CLIPEncoderLayer'],
                                         dtype=torch.half)

        self.model = model.model.eval()
        self.vision_tower = model.model.vision_tower.half().eval()
        self.mm_projector = model.model.mm_projector.half().eval()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images."""
        image_features = self.vision_tower(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refer to `super().preprocess() for spec."""
        images = self.collect_images(messages)
        outputs = []
        for image, params in images:
            image = image.convert('RGB')
            pixel_values = process_images([image], self.image_processor, self.config)
            outputs.append(
                dict(pixel_values=pixel_values,
                     image_size=image.size,
                     image_tokens=self.n_token_per_image,
                     image_token_id=self.image_token_id))
        messages.append(dict(role='preprocess', content=outputs))
        return messages

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

        from llava.model.llava_arch import get_anyres_image_grid_shape, unpad_image

        inputs = [x['content'] for x in messages if x['role'] == 'preprocess']
        inputs = inputs[0]
        outputs = []
        for idx in range(0, len(inputs), max_batch_size):
            image_sizes = [x['image_size'] for x in inputs[idx:idx + max_batch_size]]
            pixel_values = [x['pixel_values'] for x in inputs[idx:idx + max_batch_size]]
            if pixel_values[0].ndim == 5:
                split_sizes = [x.shape[1] for x in pixel_values]
                pixel_values = torch.cat([x for x in pixel_values], dim=1)
                logger.info(f'vision forward shape: {pixel_values.shape}')
                pixel_values = pixel_values.squeeze(0)
                pixel_values = pixel_values.to(device=self.vision_tower.device, dtype=torch.float16)
                feats = self.encode_images(pixel_values)
                feats = torch.split(feats, split_sizes, dim=0)
                mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
                image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
                if mm_patch_merge_type == 'flat':
                    outputs.expand([x.flatten(0, 1) for x in feats])
                elif mm_patch_merge_type.startswith('spatial'):
                    for img_idx, feat in enumerate(feats):
                        if feat.shape[0] > 1:
                            base_feat = feat[0]
                            feat = feat[1:]
                            height = self.vision_tower.num_patches_per_side
                            width = self.vision_tower.num_patches_per_side
                            assert height * width == base_feat.shape[0]
                            if image_aspect_ratio == 'anyres':
                                num_patch_width, num_patch_height = \
                                    get_anyres_image_grid_shape(
                                        image_sizes[img_idx],
                                        self.config.image_grid_pinpoints,
                                        self.vision_tower.config.image_size)
                                feat = feat.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                raise NotImplementedError
                            if 'unpad' in mm_patch_merge_type:
                                feat = feat.permute(4, 0, 2, 1, 3).contiguous()
                                feat = feat.flatten(1, 2).flatten(2, 3)
                                feat = unpad_image(feat, image_sizes[img_idx])
                                feat = torch.cat((feat, self.model.image_newline[:, None, None].expand(
                                    *feat.shape[:-1], 1).to(feat.device)),
                                                 dim=-1)
                                feat = feat.flatten(1, 2).transpose(0, 1)
                            else:
                                feat = feat.permute(0, 2, 1, 3, 4).contiguous()
                                feat = feat.flatten(0, 3)
                            feat = torch.cat((base_feat, feat), dim=0)
                        else:
                            feat = feat[0]
                            if 'unpad' in mm_patch_merge_type:
                                feat = torch.cat((feat, self.model.image_newline[None].to(feat.device)), dim=0)
                        outputs.append(feat)
                else:
                    raise ValueError('Unexpected mm_patch_merge_type: '
                                     f'{self.config.mm_patch_merge_type}')
            else:
                pixel_values = torch.cat(pixel_values, dim=0)
                pixel_values = pixel_values.to(device=self.vision_tower.device, dtype=torch.float16)
                logger.info(f'vision forward shape: {pixel_values.shape}')
                feats = self.encode_images(pixel_values)
                outputs.extend([x for x in feats])
        messages.append(dict(role='forward', content=outputs))
        return messages
