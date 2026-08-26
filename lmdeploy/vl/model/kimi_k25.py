# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.base import VISION_MODELS, MultimodalSpecialTokens, VisionModel
from lmdeploy.vl.model.preprocess_utils import get_mm_items_offset


@VISION_MODELS.register_module()
class KimiK25VisionModel(VisionModel):
    """Kimi-K2.5/K2.6 image preprocessor for the PyTorch engine.

    The fixed Kimi-K2.6 processor emits one ``<|media_pad|>`` token per
    image, packed NaViT patches, and a unified ``grid_thws`` tensor.  LMDeploy
    must expand each placeholder before scheduling so that every projected
    vision feature owns one token slot.
    """

    _arch = [
        'KimiK25ForConditionalGeneration',
        'Kimi_K25ForConditionalGeneration',
    ]

    _IMAGE_TOKEN = '<|media_pad|>'
    _PATCH_SIZE = 14
    _MERGE_KERNEL = (2, 2)
    _INTEGER_DTYPES = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }

    def build_preprocessor(self, trust_remote_code: bool = False):
        """Build and validate the snapshot-owned Kimi processor."""
        if self.backend not in ('', None, 'pytorch'):
            raise ValueError('Kimi-K2.6 image preprocessing currently supports only the PyTorch engine.')

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = self.processor.tokenizer
        token_id = tokenizer.convert_tokens_to_ids(self._IMAGE_TOKEN)
        config_token_id = getattr(self.hf_config, 'media_placeholder_token_id', None)
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f'Kimi processor does not define a valid {self._IMAGE_TOKEN} token id.')
        if config_token_id is None or token_id != config_token_id:
            raise ValueError(
                'Kimi media placeholder token mismatch: '
                f'tokenizer={token_id}, config={config_token_id}.')

        vision_config = getattr(self.hf_config, 'vision_config', None)
        patch_size = getattr(vision_config, 'patch_size', None)
        merge_kernel = getattr(vision_config, 'merge_kernel_size', None)
        if patch_size != self._PATCH_SIZE:
            raise ValueError(
                f'Unsupported Kimi vision patch size {patch_size}; expected {self._PATCH_SIZE}.')
        if isinstance(merge_kernel, int):
            merge_kernel = (merge_kernel, merge_kernel)
        if merge_kernel is None or tuple(merge_kernel) != self._MERGE_KERNEL:
            raise ValueError(
                f'Unsupported Kimi merge kernel {merge_kernel}; expected {self._MERGE_KERNEL}.')

        self.image_token = self._IMAGE_TOKEN
        self.image_token_id = token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_token,
            image_token_id=self.image_token_id,
        )

    @staticmethod
    def _collect_images(messages: list[dict]) -> list[Image.Image]:
        """Collect images in prompt order and reject unsupported media."""
        if not isinstance(messages, list):
            raise TypeError(f'Kimi messages must be a list, got {type(messages).__name__}.')

        images = []
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise TypeError(
                    f'Kimi message {message_index} must be a dict, got {type(message).__name__}.')
            content = message.get('content')
            if content is None or isinstance(content, str):
                continue
            if not isinstance(content, list):
                raise TypeError(
                    f'Kimi message {message_index} content must be text or a list, '
                    f'got {type(content).__name__}.')

            for item_index, item in enumerate(content):
                if not isinstance(item, dict):
                    raise TypeError(
                        f'Kimi content item {message_index}:{item_index} must be a dict, '
                        f'got {type(item).__name__}.')
                item_type = item.get('type')
                if item_type == 'text':
                    continue
                if item_type != Modality.IMAGE:
                    raise ValueError(
                        'Kimi M5 image frontend does not support modality '
                        f'{item_type!r} at {message_index}:{item_index}.')
                image = item.get('data')
                if not isinstance(image, Image.Image):
                    raise TypeError(
                        'Kimi image data must be a PIL.Image.Image after media parsing, '
                        f'got {type(image).__name__} at {message_index}:{item_index}.')
                images.append(image)
        return images

    @staticmethod
    def _single_input_ids(value: Any) -> list[int]:
        """Convert one processor sequence to a validated Python token list."""
        if isinstance(value, torch.Tensor):
            if value.ndim == 2:
                if value.shape[0] != 1:
                    raise ValueError(
                        f'Kimi image preprocessing supports one prompt at a time, got shape {tuple(value.shape)}.')
                value = value[0]
            if value.ndim != 1:
                raise ValueError(f'Kimi input_ids must be rank 1 or [1, S], got shape {tuple(value.shape)}.')
            value = value.tolist()

        if not isinstance(value, list):
            raise TypeError(f'Kimi input_ids must be a list or tensor, got {type(value).__name__}.')
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in value):
            raise ValueError('Kimi input_ids must contain only non-negative integer token ids.')
        return list(value)

    def _expand_media_tokens(self, input_ids: list[int],
                             image_tokens: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
        """Expand raw media placeholders and return final exclusive spans."""
        token_tensor = torch.tensor(input_ids, dtype=torch.long)
        raw_offsets = get_mm_items_offset(token_tensor, self.image_token_id)
        if len(raw_offsets) != len(image_tokens):
            raise ValueError(
                'Kimi image placeholder count must match image count: '
                f'got {len(raw_offsets)} placeholder spans for {len(image_tokens)} images.')

        expanded_ids = []
        cursor = 0
        for image_index, ((start, end), expected_tokens) in enumerate(zip(raw_offsets, image_tokens)):
            actual_tokens = end - start
            if actual_tokens not in (1, expected_tokens):
                raise ValueError(
                    f'Kimi image {image_index} placeholder span has {actual_tokens} tokens; '
                    f'expected one raw placeholder or {expected_tokens} expanded tokens.')
            expanded_ids.extend(input_ids[cursor:start])
            expanded_ids.extend([self.image_token_id] * expected_tokens)
            cursor = end
        expanded_ids.extend(input_ids[cursor:])

        expanded_offsets = get_mm_items_offset(
            torch.tensor(expanded_ids, dtype=torch.long),
            self.image_token_id,
        )
        expected_offsets = len(image_tokens)
        if len(expanded_offsets) != expected_offsets:
            raise ValueError(
                'Kimi expanded media spans are not independently addressable: '
                f'got {len(expanded_offsets)} spans for {expected_offsets} images.')
        for image_index, ((start, end), expected_tokens) in enumerate(
                zip(expanded_offsets, image_tokens)):
            if end - start != expected_tokens:
                raise ValueError(
                    f'Kimi expanded image {image_index} span has {end - start} tokens; '
                    f'expected {expected_tokens}.')
        return expanded_ids, expanded_offsets

    def _validate_vision_outputs(
        self,
        pixel_values: Any,
        grid_thws: Any,
        image_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Validate the fixed snapshot's packed image output."""
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(
                f'Kimi pixel_values must be a tensor, got {type(pixel_values).__name__}.')
        if pixel_values.ndim != 4 or tuple(pixel_values.shape[1:]) != (
                3, self._PATCH_SIZE, self._PATCH_SIZE):
            raise ValueError(
                'Kimi pixel_values must have shape [P, 3, 14, 14], '
                f'got {tuple(pixel_values.shape)}.')
        if not pixel_values.is_floating_point():
            raise TypeError(f'Kimi pixel_values must be floating point, got {pixel_values.dtype}.')
        if not torch.isfinite(pixel_values).all().item():
            raise ValueError('Kimi pixel_values contain NaN or Inf.')

        if not isinstance(grid_thws, torch.Tensor):
            raise TypeError(f'Kimi grid_thws must be a tensor, got {type(grid_thws).__name__}.')
        if grid_thws.dtype not in self._INTEGER_DTYPES:
            raise TypeError(f'Kimi grid_thws must use an integer dtype, got {grid_thws.dtype}.')
        if grid_thws.ndim != 2 or grid_thws.shape[1] != 3:
            raise ValueError(f'Kimi grid_thws must have shape [N, 3], got {tuple(grid_thws.shape)}.')
        if grid_thws.shape[0] != image_count:
            raise ValueError(
                f'Kimi grid_thws count must match image count: {grid_thws.shape[0]} != {image_count}.')

        grid_thws = grid_thws.detach().to(device='cpu').contiguous()
        image_tokens = []
        patch_count = 0
        for image_index, (t, h, w) in enumerate(grid_thws.tolist()):
            if t != 1:
                raise ValueError(
                    f'Kimi M5 image frontend requires t=1, got grid {t, h, w} for image {image_index}.')
            if h <= 0 or w <= 0:
                raise ValueError(f'Kimi image grid dimensions must be positive, got {t, h, w}.')
            if h % self._MERGE_KERNEL[0] or w % self._MERGE_KERNEL[1]:
                raise ValueError(
                    f'Kimi image grid {t, h, w} is not divisible by merge kernel {self._MERGE_KERNEL}.')
            patch_count += t * h * w
            image_tokens.append(h * w // (self._MERGE_KERNEL[0] * self._MERGE_KERNEL[1]))

        if patch_count != pixel_values.shape[0]:
            raise ValueError(
                'Kimi pixel patch rows must equal sum(t*h*w): '
                f'{pixel_values.shape[0]} != {patch_count}.')
        return pixel_values.contiguous(), grid_thws, image_tokens

    def preprocess(
        self,
        messages: list[dict],
        input_prompt: str | list[int],
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Preprocess one text/image request into LMDeploy multimodal items."""
        if mm_processor_kwargs:
            raise ValueError(
                'Kimi M5 image frontend does not support mm_processor_kwargs overrides.')
        if not isinstance(input_prompt, (str, list)):
            raise TypeError(
                f'Kimi input_prompt must be text or token ids, got {type(input_prompt).__name__}.')

        images = self._collect_images(messages)
        if not images:
            if isinstance(input_prompt, str):
                text_outputs = self.processor.tokenizer(
                    input_prompt,
                    return_tensors='pt',
                )
                input_ids = self._single_input_ids(text_outputs['input_ids'])
            else:
                input_ids = self._single_input_ids(input_prompt)
            input_ids, offsets = self._expand_media_tokens(input_ids, [])
            assert not offsets
            return dict(input_ids=input_ids, multimodal=[])

        medias = [dict(type='image', image=image) for image in images]
        if isinstance(input_prompt, str):
            processor_outputs = self.processor(
                medias=medias,
                text=input_prompt,
                return_tensors='pt',
            )
            input_ids = self._single_input_ids(processor_outputs.get('input_ids'))
        else:
            processor_outputs = self.processor.image_processor.preprocess(
                medias,
                return_tensors='pt',
            )
            input_ids = self._single_input_ids(input_prompt)

        pixel_values = self._postprocess_mm_output(
            processor_outputs.get('pixel_values'),
            getattr(self, 'mm_feature_dtype', None),
        )
        pixel_values, grid_thws, image_tokens = self._validate_vision_outputs(
            pixel_values,
            processor_outputs.get('grid_thws'),
            len(images),
        )
        input_ids, offsets = self._expand_media_tokens(input_ids, image_tokens)

        multimodal = []
        patch_start = 0
        for grid_thw, offset, token_count in zip(grid_thws, offsets,
                                                  image_tokens):
            patch_count = int(torch.prod(grid_thw).item())
            patch_end = patch_start + patch_count
            image_pixels = pixel_values[patch_start:patch_end].clone()
            multimodal.append(
                dict(
                    modality=Modality.IMAGE,
                    pixel_values=image_pixels,
                    grid_thws=grid_thw.unsqueeze(0).clone(),
                    offset=offset,
                    image_token_id=self.image_token_id,
                    image_tokens=token_count,
                ))
            patch_start = patch_end

        if patch_start != pixel_values.shape[0]:
            raise RuntimeError(
                f'Kimi image split consumed {patch_start} patches, expected {pixel_values.shape[0]}.')
        return dict(input_ids=input_ids, multimodal=multimodal)
