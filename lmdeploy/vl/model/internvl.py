# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

import torch
from PIL.Image import Image
from transformers import AutoModel, CLIPImageProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    """copy from https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image,
                       min_num=1,
                       max_num=6,
                       image_size=448,
                       use_thumbnail=False):
    """copy from https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


@VISION_MODELS.register_module()
class InternVLVisionModel(VisonModel):
    """InternVL vision model."""

    _arch = 'InternVLChatModel'

    def build_model(self):
        """Load model."""
        from accelerate import init_empty_weights
        with init_empty_weights():
            config = self.hf_config
            # transformers below 4.37.0 may raise error about flash_attn
            config.llm_config.attn_implementation = 'eager'
            model = AutoModel.from_config(config, trust_remote_code=True)
            if not self.with_llm:
                del model.language_model
            else:
                self.vl_model = model
            model.half()

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map='auto' if not self.with_llm else {'': 'cpu'},
                max_memory=self.max_memory,
                no_split_module_classes=['InternVisionEncoderLayer'],
                dtype=torch.half)

        # We need eval mode to freeze the weights in model, thus,
        # avoid randomness in inference.
        self.model = model.eval()
        self.config = config

        if getattr(self.config, 'dynamic_image_size', False):
            logger.info('using InternVL-Chat-V1-5 vision preprocess')
            MEAN = (0.485, 0.456, 0.406)
            STD = (0.229, 0.224, 0.225)
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            input_size = self.config.vision_config.image_size
            self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')
                         if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size),
                         interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
            self._forward_func = self._forward_v1_5
        else:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                self.model_path)
            self._forward_func = self._forward

    def _preprocess_v1_5(self, images: List[Image], params: List[Dict] = None):
        if params is not None:
            assert len(images) == len(
                params), 'different length of images and params'
        else:
            params = [{}] * len(images)

        image_res = {'low': 6, 'medium': 12, 'high': 24}

        outputs = []
        for image, param in zip(images, params):
            max_num = param.get('max_dynamic_patch')
            if max_num is None or not isinstance(max_num, int):
                res_key = param.get('detail', 'default')
                max_num = image_res.get(res_key, self.config.max_dynamic_patch)
            out = dynamic_preprocess(
                image,
                min_num=self.config.min_dynamic_patch,
                max_num=max_num,
                image_size=self.config.vision_config.image_size,
                use_thumbnail=self.config.use_thumbnail)
            out = [self.transform(x) for x in out]
            out = torch.stack(out)  # (patch) x c x h x w
            outputs.append(out)
        return outputs

    def _forward_v1_5(self, images: List[Image], params: List[Dict] = None):
        """forward for internvl-chat-v1-5."""
        outputs = self._preprocess_v1_5(images, params)
        split = [x.shape[0] for x in outputs]
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.to(self.model.device, dtype=torch.float16)
        outputs = self.model.extract_feature(outputs)
        outputs = torch.split(outputs, split, dim=0)
        outputs = [x.reshape(-1, x.shape[-1]) for x in outputs]
        return outputs

    def _forward(self, images: List[Image], params: List[Dict] = None):
        """forward for internvl-chat-v1-1, internvl-chat-v1-2."""
        pixel_values = self.image_processor(images=images,
                                            return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(self.model.device, dtype=torch.float16)
        outputs = self.model.extract_feature(pixel_values)
        outputs = torch.split(outputs, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs

    @torch.no_grad()
    def forward(self,
                images: List[Image],
                params: List[Dict] = None) -> List[torch.Tensor]:
        """forward."""
        images = [x.convert('RGB') for x in images]
        return self._forward_func(images, params)
