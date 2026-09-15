# Copyright (c) OpenMMLab. All rights reserved.
"""Multimodal frontend for GLM-5.3-Flash."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, AutoTokenizer

from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.base import (
    VISION_MODELS,
    MultimodalSpecialTokens,
    VisionModel,
)
from lmdeploy.vl.model.preprocess_utils import (
    get_expanded_mm_items,
    get_override_size,
)


def _processor_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Translate GLM-5 token budgets to the GLM-4V pixel contract."""
    config = dict(config)
    for key in ('image_processor_type', 'video_processor_type',
                'patch_expand_factor'):
        config.pop(key, None)
    min_tokens = config.pop('min_image_tokens', None)
    max_tokens = config.pop('max_image_tokens', None)
    patch_size = int(config.get('patch_size', 14))
    merge_size = int(config.get('merge_size', 2))
    temporal_patch_size = int(config.get('temporal_patch_size', 2))
    pixels_per_token = temporal_patch_size * (patch_size * merge_size)**2
    if min_tokens is not None or max_tokens is not None:
        config['size'] = {
            'shortest_edge': int(min_tokens or 1) * pixels_per_token,
            'longest_edge': int(max_tokens or min_tokens) * pixels_per_token,
        }
    return config


@VISION_MODELS.register_module()
class GLM5NextVisionModel(VisionModel):
    """Prepare GLM-5.3 images/videos for the native PyTorch vision tower."""

    _arch = ['Glm5NextForConditionalGeneration']
    # Match GLM/SGLang's video contract before the HF processor: sample at
    # 2 FPS, cap at 2048 source frames, and complete temporal pairs.
    default_media_io_kwargs = {
        'video': {
            'fps': 2.0,
            'num_frames': 2048,
            'sampling_strategy': 'glm',
        },
    }

    @classmethod
    def match(cls, config):
        arch = config.architectures[0] if config.architectures else None
        return arch in cls._arch and getattr(config, 'vision_config', None) is not None

    def _build_compat_processor(self, trust_remote_code: bool):
        """Build from public GLM-4V components on pre-GLM-5 Transformers."""
        from transformers.models.glm4v.image_processing_glm4v import (
            Glm4vImageProcessor,
        )
        from transformers.models.glm4v.processing_glm4v import Glm4vProcessor
        from transformers.models.glm4v.video_processing_glm4v import (
            Glm4vVideoProcessor,
        )

        config_path = Path(self.model_path) / 'processor_config.json'
        processor_config = json.loads(config_path.read_text())
        image_processor = Glm4vImageProcessor(
            **_processor_kwargs(processor_config['image_processor']))
        video_processor = Glm4vVideoProcessor(
            **_processor_kwargs(processor_config['video_processor']))
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=trust_remote_code)
        return Glm4vProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=getattr(tokenizer, 'chat_template', None),
        )

    def build_preprocessor(self, trust_remote_code: bool = False):
        processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=trust_remote_code)
        if not (hasattr(processor, 'image_processor')
                and hasattr(processor, 'video_processor')):
            processor = self._build_compat_processor(trust_remote_code)
        self.processor = processor

        self.image_token = processor.image_token
        self.video_token = processor.video_token
        self.image_token_id = int(self.hf_config.image_token_id)
        configured_video_token_id = int(self.hf_config.video_token_id)
        self.input_video_token_id = configured_video_token_id

        # GLM video prompts use <|video|> before processing, then expand each
        # frame to an image-token span.  Detect this contract rather than
        # hard-coding it, so a future native GLM-5 processor can use a distinct
        # post-tokenization video id.
        frame_builder = getattr(processor, 'replace_frame_token_id', None)
        frame_text = frame_builder(0, 1) if callable(frame_builder) else ''
        self.video_token_id = (self.image_token_id
                               if self.image_token in frame_text else
                               configured_video_token_id)
        self._shared_video_token = self.video_token_id == self.image_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_token,
            video_token=self.video_token,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
        )

    @staticmethod
    def _next_span(input_ids: torch.Tensor, cursor: int, token_id: int,
                   length: int) -> tuple[tuple[int, int], int]:
        while cursor < len(input_ids) and int(input_ids[cursor]) != token_id:
            cursor += 1
        end = cursor + length
        if end > len(input_ids) or not torch.all(input_ids[cursor:end] == token_id):
            raise ValueError(
                f'cannot locate a contiguous multimodal span of {length} tokens')
        return (cursor, end), end

    def _shared_token_offsets(
        self,
        input_ids: torch.Tensor,
        mm_items: list[tuple[Modality, Any, dict]],
        collected: dict[Modality, dict[str, Any]],
    ) -> None:
        """Recover mixed image/video ownership when both use image tokens."""
        merge_length = self.processor.image_processor.merge_size**2
        image_index = 0
        video_index = 0
        cursor = 0
        image_offsets = []
        video_offsets = []
        for modality, _, _ in mm_items:
            if modality == Modality.IMAGE:
                grid = collected[Modality.IMAGE]['image_grid_thw'][image_index]
                length = int(torch.as_tensor(grid).prod().item()) // merge_length
                span, cursor = self._next_span(input_ids, cursor,
                                               self.image_token_id, length)
                image_offsets.append(span)
                image_index += 1
            elif modality == Modality.VIDEO:
                grid = collected[Modality.VIDEO]['video_grid_thw'][video_index]
                t, h, w = torch.as_tensor(grid).tolist()
                length = int(h * w) // merge_length
                for _ in range(int(t)):
                    span, cursor = self._next_span(input_ids, cursor,
                                                   self.image_token_id, length)
                    video_offsets.append(span)
                video_index += 1
        if Modality.IMAGE in collected:
            collected[Modality.IMAGE]['offset'] = image_offsets
        if Modality.VIDEO in collected:
            collected[Modality.VIDEO]['offset'] = video_offsets

    def _expand_raw_input_ids(
        self,
        input_prompt: list[int],
        outputs: dict[str, Any],
        raw_videos: list[Any],
        video_metadatas: list[Any],
    ) -> torch.Tensor:
        """Expand raw image/video placeholder IDs with official GLM text."""
        tokenizer = self.processor.tokenizer
        image_grids = outputs.get('image_grid_thw')
        video_grids = outputs.get('video_grid_thw')

        image_replacements: list[list[int]] = []
        if image_grids is not None:
            for image_idx in range(len(image_grids)):
                replacement = self.processor.replace_image_token(
                    outputs, image_idx=image_idx)
                image_replacements.append(
                    tokenizer.encode(replacement, add_special_tokens=False))

        video_replacements: list[list[int]] = []
        if video_grids is not None:
            from transformers.video_utils import make_batched_metadata

            metadata = make_batched_metadata(
                raw_videos, video_metadata=video_metadatas)
            video_inputs = {
                'video_grid_thw': video_grids,
                'video_metadata': metadata,
            }
            for video_idx in range(len(video_grids)):
                replacement = self.processor.replace_video_token(
                    video_inputs, video_idx=video_idx)
                video_replacements.append(
                    tokenizer.encode(replacement, add_special_tokens=False))

        image_count = input_prompt.count(self.image_token_id)
        video_count = input_prompt.count(self.input_video_token_id)
        if image_count != len(image_replacements):
            raise ValueError(
                'raw GLM-5 input image placeholders do not match images: '
                f'{image_count} placeholders for {len(image_replacements)} images.'
            )
        if video_count != len(video_replacements):
            raise ValueError(
                'raw GLM-5 input video placeholders do not match videos: '
                f'{video_count} placeholders for {len(video_replacements)} videos.'
            )

        image_index = 0
        video_index = 0
        expanded: list[int] = []
        for token in input_prompt:
            if token == self.image_token_id:
                expanded.extend(image_replacements[image_index])
                image_index += 1
            elif token == self.input_video_token_id:
                expanded.extend(video_replacements[video_index])
                video_index += 1
            else:
                expanded.append(token)
        return torch.tensor(expanded, dtype=torch.long)

    def preprocess(self,
                   messages: list[dict],
                   input_prompt: str | list[int],
                   mm_processor_kwargs: dict[str, Any] | None = None):
        mm_items = self.collect_multimodal_items(messages)
        modalities = {item[0] for item in mm_items}
        is_raw_video = (not isinstance(input_prompt, str)
                        and Modality.VIDEO in modalities)
        is_mixed = {Modality.IMAGE, Modality.VIDEO} <= modalities
        if not self._shared_video_token or not (is_mixed or is_raw_video):
            return super().preprocess(messages, input_prompt,
                                      mm_processor_kwargs)

        raw_images = [item[1] for item in mm_items
                      if item[0] == Modality.IMAGE]
        raw_videos = [item[1] for item in mm_items
                      if item[0] == Modality.VIDEO]
        video_metadatas = [item[2].get('video_metadata') for item in mm_items
                           if item[0] == Modality.VIDEO]
        mm_processor_kwargs = mm_processor_kwargs or {}
        kwargs: dict[str, Any] = {}
        if raw_images:
            kwargs['images'] = raw_images
        if raw_videos:
            kwargs['videos'] = raw_videos
            kwargs['videos_kwargs'] = {
                'video_metadata': video_metadatas,
                'do_resize': True,
                'do_sample_frames': False,
            }
        image_size = get_override_size(self.processor.image_processor,
                                       mm_processor_kwargs.get('image'),
                                       modality='image')
        if image_size is not None:
            kwargs['images_kwargs'] = {'size': image_size}
        video_size = get_override_size(self.processor.video_processor,
                                       mm_processor_kwargs.get('video'),
                                       modality='video')
        if video_size is not None:
            kwargs['videos_kwargs']['size'] = video_size

        input_text = input_prompt if isinstance(input_prompt, str) else ''
        outputs = self.processor(text=[input_text],
                                 padding=True,
                                 return_tensors='pt',
                                 **kwargs)
        collected: dict[Modality, dict[str, Any]] = {}
        for name, value in outputs.items():
            modality = self.ATTR_NAME_TO_MODALITY.get(name)
            if modality not in (Modality.IMAGE, Modality.VIDEO):
                continue
            collected.setdefault(modality, {})
            if name in self.FEATURE_NAMES:
                value = self._postprocess_mm_output(
                    value, getattr(self, 'mm_feature_dtype', None))
                name = 'feature'
            collected[modality][name] = value

        input_ids = (outputs['input_ids'].flatten()
                     if isinstance(input_prompt, str) else
                     self._expand_raw_input_ids(input_prompt, outputs,
                                                raw_videos,
                                                video_metadatas))
        self._shared_token_offsets(input_ids, mm_items, collected)
        expanded = get_expanded_mm_items(collected, self.mm_tokens)
        return dict(input_ids=input_ids.tolist(), multimodal=expanded)
