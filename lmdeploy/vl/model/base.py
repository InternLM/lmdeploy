# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
from abc import ABC, abstractmethod
from itertools import groupby
from typing import Any

import numpy as np
import torch
from mmengine import Registry
from transformers import AutoConfig, AutoTokenizer

from lmdeploy.archs import get_model_arch
from lmdeploy.vl.constants import Modality

VISION_MODELS = Registry('vision_model')


class VisionModel(ABC):
    """Visual model which extract image feature."""
    _arch: str | list[str] = None

    def __init__(self,
                 model_path: str,
                 with_llm: bool = False,
                 max_memory: dict[int, int] = None,
                 hf_config: AutoConfig = None,
                 backend: str = ''):
        """init."""
        self.model_path = model_path
        self.with_llm = with_llm
        self.max_memory = max_memory
        self.backend = backend
        if hf_config is None:
            _, hf_config = get_model_arch(model_path)
        self.hf_config = hf_config
        self.image_token_id = self.get_pad_token_id(model_path, hf_config) or 0

        # mapping from attribute names to modality types
        self.ATTR_NAME_TO_MODALITY = {
            # image-related attributes
            'pixel_values': Modality.IMAGE,
            'image_sizes': Modality.IMAGE,
            'image_grid_thw': Modality.IMAGE,
            'image_attention_mask': Modality.IMAGE,
            'image_emb_mask': Modality.IMAGE,
            'images_spatial_crop': Modality.IMAGE,
            'images_crop': Modality.IMAGE,
            'has_local_crops': Modality.IMAGE,
            'has_images': Modality.IMAGE,
            'tgt_size': Modality.IMAGE,
            'image_grid_hws': Modality.IMAGE,
            'aspect_ratio_ids': Modality.IMAGE,
            'aspect_ratio_mask': Modality.IMAGE,
            'num_patches': Modality.IMAGE,
            'patch_pixel_values': Modality.IMAGE,
            'block_sizes': Modality.IMAGE,
            # audio-related attributes
            'audio_features': Modality.AUDIO,
            'audio_feature_lens': Modality.AUDIO,
            'input_features': Modality.AUDIO,
            'input_features_mask': Modality.AUDIO,
            'audio_attention_mask': Modality.AUDIO,
            'feature_attention_mask': Modality.AUDIO,
            # video-related attributes
            'pixel_values_videos': Modality.VIDEO,
            'second_per_grid_ts': Modality.VIDEO,
            'video_grid_thw': Modality.VIDEO,
        }

        # name of the feature filed
        self.FEATURE_NAMES = [
            'pixel_values',
            'pixel_values_videos',
            'audio_features',
            'input_features',
        ]

    @staticmethod
    def get_mm_items_offset(
        input_ids: torch.Tensor, mm_token_id: int
    ) -> list[tuple[int, int]]:
        """
        Get a set of range for mm_items from input_ids
        Example:
            input_ids = [1, 2, 3, 3, 3, 4, 3, 3]
            mm_token_id = 3
            return result = [(2,4),(6,7)]
        """
        mask = input_ids == mm_token_id
        start_positions = (mask & ~torch.roll(mask, 1)).nonzero(as_tuple=True)[0]
        end_positions = (mask & ~torch.roll(mask, -1)).nonzero(as_tuple=True)[0]
        return list(zip(start_positions.tolist(), end_positions.tolist()))

    def get_pad_token_id(self, model_path, hf_config):
        """Get pad_token_id from hf_config or tokenizer."""
        pad_token_id = getattr(hf_config, 'pad_token_id', None)
        if pad_token_id is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                pad_token_id = getattr(tokenizer, 'pad_token_id', None)
            except Exception as e:
                print(e)
                pass
        return pad_token_id

    @abstractmethod
    def build_preprocessor(self, ):
        """Build the preprocessor.

        NOTE: When the derived class implements this method, try not to
        introduce the upper stream model repo as a thirdparty package
        """
        raise NotImplementedError()

    def build_model(self, ):
        """Build the vision part of a VLM model when backend is turbomind.

        But when `with_llm=True`, load the whole VLM model
        """
        if self.backend == 'turbomind' or self.with_llm:
            raise NotImplementedError()

    # adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/mm_utils.py
    def _get_expanded_mm_items(self, collected_mm_items):
        """Hf processor outputs produced bundled data for multiple
        images/videos we need to expand them into per-image/video entries for
        better cache locality and fine-grained scheduling."""
        expanded_mm_items = []
        for modality, item in collected_mm_items.items():
            is_bundled = item.get('offset', None) is not None and len(item['offset']) > 1

            # non-bundled case
            if not is_bundled:
                expanded_mm_items.append(
                    dict(
                        modality=modality,
                        pixel_values=item['feature'],
                        image_grid_thw=item['image_grid_thw'][0],
                        offset=item['offset'][0],
                        image_token_id=self.image_token_id
                        )
                    )
                continue

            # bundled case
            num_items = len(item['offset'])
            if modality == Modality.IMAGE:
                image_grid_thw = item['image_grid_thw']
                grid_len = image_grid_thw.shape[0]

                patches_per_item = []
                for grid in image_grid_thw:
                    grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                    patches_per_item.append(int(torch.prod(grid_tensor).item()))

                cumulative = torch.cumsum(
                    torch.tensor(patches_per_item, dtype=torch.long), dim=0
                )
                slice_indices = [0] + cumulative.tolist()

                # expand each image into a separate item
                for i in range(num_items):
                    start_idx, end_idx = slice_indices[i], slice_indices[i + 1]
                    # TODO: may compute mm mask and remove mm token id inputs
                    expanded_mm_items.append(
                        dict(
                            modality=modality,
                            pixel_values=item['feature'][start_idx:end_idx],
                            image_grid_thw=image_grid_thw[i],
                            offset=item['offset'][i],
                            image_token_id=self.image_token_id,
                        )
                    )
            elif modality == Modality.VIDEO:
                video_grid_thw = item['video_grid_thw']

                # video_grid_thw shape: [num_videos, 3] where each row is [T, H, W]
                # When T > 1, item.offsets contains frames (num_items = total frames)
                # grid_len = num_videos, num_items = sum(T for each video) = total frames
                grid_len = video_grid_thw.shape[0]
                num_videos = grid_len

                # calculate total frames and frames per video
                frames_per_video = []
                total_frames = 0
                for i in range(num_videos):
                    grid = video_grid_thw[i]
                    if isinstance(grid, torch.Tensor):
                        T = int(grid[0].item())  # T is the first element [T, H, W]
                    else:
                        grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                        T = int(grid_tensor[0].item())
                    frames_per_video.append(T)
                    total_frames += T

                # num_items should equal total_frames when T > 1
                if num_items != total_frames:
                    expanded_mm_items.append(item)
                    continue

                # calculate patches per video: T * H * W for each video
                patches_per_video = []
                for i in range(num_videos):
                    grid = video_grid_thw[i]
                    if isinstance(grid, torch.Tensor):
                        patches_per_video.append(int(torch.prod(grid).item()))
                    else:
                        grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                        patches_per_video.append(int(torch.prod(grid_tensor).item()))

                # calculate cumulative patches to get slice indices for each video
                cumulative = torch.cumsum(
                    torch.tensor(patches_per_video, dtype=torch.long), dim=0
                )
                slice_indices = [0] + cumulative.tolist()

                # group frames by video, calculate frame indices for each video
                frame_start_indices = [0]
                for i in range(num_videos):
                    frame_start_indices.append(
                        frame_start_indices[-1] + frames_per_video[i]
                    )

                # expand each video into a separate item
                for video_idx in range(num_videos):
                    start, end = (
                        slice_indices[video_idx],
                        slice_indices[video_idx + 1],
                    )
                    frame_start, frame_end = (
                        frame_start_indices[video_idx],
                        frame_start_indices[video_idx + 1],
                    )

                    # import pdb; pdb.set_trace()
                    # expand each frame into a separate item, not sure good or no
                    t, h, w = video_grid_thw[video_idx].tolist()
                    for frame_idx in range(t):
                        video_feature = item['feature'][start:end]
                        # FIXME: grid_thw [1, h, w] is only for qwen3vl
                        expanded_mm_items.append(
                            dict(
                                modality=modality,
                                pixel_values_videos=video_feature[frame_idx * h * w:(frame_idx + 1) * h * w],
                                video_grid_thw=torch.tensor([1, h, w]),
                                offset=item['offset'][frame_start:frame_end][frame_idx],
                                video_token_id=self.video_token_id,
                            )
                        )

        return expanded_mm_items

    def preprocess(self,
                   messages: list[dict],
                   input_text: str,
                   mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:
        """Refer to `super().preprocess()` for spec."""

        kwargs = mm_processor_kwargs or {}
        mm_items = self.collect_multimodal_items(messages)

        # TODO: get kwargs from params
        # TODO: careful about mm processor kwargs, in mixed modality case
        # may need to treat each modality differently
        raw_images, raw_videos, video_metadatas = [], [], []
        for modality, data, params in mm_items:
            if modality == Modality.IMAGE:
                raw_images.append(data)
            elif modality == Modality.VIDEO:
                raw_videos.append(data)
                video_metadatas.append(params['video_metadata'])
            else:
                raise ValueError(f'unsupported modality {modality}')

        # get kwrags for processor
        if raw_images:
            kwargs['images'] = raw_images
        if raw_videos:
            kwargs['videos'] = raw_videos
            kwargs['video_metadata'] = video_metadatas
            # leave resize to hf processor
            # sample frames is done in video loader, avoid duplication
            kwargs['do_resize'] = True
            kwargs['do_sample_frames'] = False
            kwargs['return_metadata'] = True

        # process raw items with hf processor
        processor_outputs = self.processor(
            text=[input_text],
            padding=True,
            return_tensors='pt',
            **kwargs,
        )
        input_ids = processor_outputs['input_ids'].flatten()

        # collect from processor outputs and categorized by modality
        collected_mm_items: dict[Modality, dict[str, Any]] = {}
        for attr_name, value in processor_outputs.items():
            if attr_name == 'input_ids':
                continue

            current_modality = self.ATTR_NAME_TO_MODALITY.get(attr_name)

            if current_modality:
                if current_modality not in collected_mm_items:
                    collected_mm_items[current_modality] = {}

                if attr_name in self.FEATURE_NAMES:
                    attr_name = 'feature'

                collected_mm_items[current_modality][attr_name] = value

        # compute offsets for all items
        for modality, item in collected_mm_items.items():
            mm_token_id = self.mm_tokens.get_token_id_by_modality(modality)
            item['offset'] = self.get_mm_items_offset(
                input_ids=input_ids,
                mm_token_id=mm_token_id,
            )

        # expand bundled hf processor outputs into per-image/video entry
        expanded_mm_items = self._get_expanded_mm_items(collected_mm_items)

        # import pdb; pdb.set_trace()
        return input_ids.tolist(), expanded_mm_items

    def has_input_ids(self, messages: list[dict]) -> bool:
        """Check whether the messages contain input_ids directly.

        Args:
            messages (list[dict]): a list of message, which is supposed to be
                the output of `preprocess`
        Returns:
            bool: whether the messages contain input_ids directly
        """
        users = [x['content'] for x in messages if x['role'] == 'user']
        return len(users) == 1 and isinstance(users[0], list) and isinstance(users[0][0].get('text', ''), list)

    def forward(self, messages: list[dict], max_batch_size: int = 1) -> list[dict]:
        """Extract image feature. ONLY implement it when the backend is
        turbomind engine.

        Args:
            messages(list[dict]): the outputs of `preprocess`
            max_batch_size(int): the max batch size when forwarding vision
                model
        Return:
            the message list with forwarding results included, which is
            determined by the derived classes
        """
        if self.backend == 'turbomind':
            raise NotImplementedError()

    # def to_pytorch(self, messages, chat_template, tokenizer, sequence_start, chat_template_kwargs=None, **kwargs):
    #     """Pack the preprocessing results in a format compatible with what is
    #     required by pytorch engine. ONLY implement it when the backend is
    #     pytorch engine.

    #     Args:
    #         messages(list[dict]): the output of `preprocess`
    #         chat_template: the chat template defined in `lmdeploy/model.py`
    #         tokenzer: the tokenizer model
    #         sequence_start: starting flag of a sequence
    #         chat_template_kwargs: additional arguments for chat template
    #             processing, such as `add_vision_id` and `enable_thinking`
    #     """
    #     if self.backend == 'pytorch':
    #         raise NotImplementedError()

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start, chat_template_kwargs=None, **kwargs):
        """Pack the forwarding results in a format compatible with what is
        required by turbomind engine. ONLY implement it when the backend is
        turbomind engine.

        Args:
            messages(list[dict]): the output of `preprocess`
            chat_template: the chat template defined in `lmdeploy/model.py`
            tokenzer: the tokenizer model
            sequence_start: starting flag of a sequence
            chat_template_kwargs: additional arguments for chat template
                processing, such as `add_vision_id` and `enable_thinking`
        """
        if self.backend == 'turbomind':
            raise NotImplementedError()

    @staticmethod
    def collect_multimodal_items(messages):
        """Gather all multimodal items along with their respective parameters
        from the messages and compile them into a single list.

        Args:
            messages (list[dict]): a list of message
        Returns:
            list[tuple[Modality, Any, dict]]: a list of (modality, data, params) for each multimodal item
        """
        multimodal_items = []
        for message in messages:
            content = message['content']
            if not isinstance(content, list):
                continue

            for x in content:
                if not isinstance(x, dict):
                    continue

                modality = x.get('type')
                if modality is None or modality == 'text':
                    continue

                data = x.get('data')
                params = {k: v for k, v in x.items() if k not in ['type', 'data']}
                multimodal_items.append((modality, data, params))

        return multimodal_items

    @staticmethod
    def IMAGE_TOKEN_included(messages):
        """Check whether the IMAGE_TOKEN is included in the messages.

        Args:
            messages (list[dict]): a list of message
        Returns:
            bool: whether the IMAGE_TOKEN is included in the messages
        """
        for message in messages:
            role, content = message['role'], message['content']
            if role != 'user':
                continue
            if isinstance(content, str) and '<IMAGE_TOKEN>' in content:
                return True
            elif isinstance(content, list):
                content = [x['text'] for x in content if x['type'] == 'text']
                if any('<IMAGE_TOKEN>' in x for x in content):
                    return True
        return False

    def to_pytorch_with_input_ids(self, messages):
        """Pack the preprocessing results in a format compatible with what is
        required by pytorch engine when input_ids are provided directly.

        Args:
            messages(list[dict]): the output of `preprocess`
        """
        # collect all preprocessing result from messages
        preps = [x['content'] for x in messages if x['role'] == 'preprocess']
        assert len(preps) == 1
        preps = preps[0]

        _input_ids = messages[0]['content'][0]['text']
        segs = []
        for k, g in groupby(_input_ids, lambda x: x == self.image_token_id):
            if not k:
                segs.append(list(g))
            else:
                segs.extend([[]] * (len(list(g)) - 1))
        if _input_ids[0] == self.image_token_id:
            segs = [[]] + segs
        if _input_ids[-1] == self.image_token_id:
            segs = segs + [[]]

        assert self.image_token_id == preps[0]['image_token_id']
        assert len(segs) == len(preps) + 1, (f'the number of image token id {self.image_token_id} is not equal '
                                             f'to input images, {len(segs) - 1} vs {len(preps)}')
        input_ids = []
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(preps):
                preps[i - 1].update(offset=len(input_ids))
                image_tokens = preps[i - 1]['image_tokens']
                input_ids.extend([self.image_token_id] * image_tokens)
            input_ids.extend(seg)

        return dict(prompt=None, input_ids=input_ids, multimodal=preps)

    def to_turbomind_aux(self, messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start):
        """Auxiliary function to pack the forwarding results in a format
        compatible with what is required by turbomind engine.

        Args:
            messages(list[dict]): the output of `preprocess`
            prompt(str): the prompt after applying chat template
            IMAGE_TOKEN(str): a placeholder where image tokens will be
                inserted
            tokenzer: the tokenizer model
            sequence_start: starting flag of a sequence
        """
        # collect image features from messages
        features = [x['content'] for x in messages if x['role'] == 'forward']
        features = features[0]
        features = [x.cpu() for x in features]
        # split prompt into segments and validate data
        segs = prompt.split(IMAGE_TOKEN)
        assert len(segs) == len(features) + 1, (f'the number of {IMAGE_TOKEN} is not equal '
                                                f'to input images, {len(segs) - 1} vs {len(features)}')

        # tokenizer prompt, and get input_embeddings and input_embedding_ranges
        input_ids = []
        begins = []
        ends = []
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(features):
                image_dim = features[i - 1].shape[0]
                begins.append(len(input_ids))
                ends.append(begins[-1] + image_dim)
                input_ids.extend([self.image_token_id] * image_dim)
            seg_ids = tokenizer.encode(seg, add_bos=((i == 0) and sequence_start))
            input_ids.extend(seg_ids)
        ranges = np.stack([begins, ends], axis=1).tolist()
        return dict(prompt=prompt, input_ids=input_ids, input_embeddings=features, input_embedding_ranges=ranges)

    @classmethod
    def match(cls, config: AutoConfig):
        """Check whether the config match the model."""
        arch = config.architectures[0] if config.architectures else None
        if arch and (arch == cls._arch or arch in cls._arch):
            return True
        return False


@dataclasses.dataclass
class MultomodalSpecialTokens:
    image_token: str | list[str] | None = None
    video_token: str | list[str] | None = None
    audio_token: str | list[str] | None = None

    image_token_id: int | None = None
    video_token_id: int | None = None
    audio_token_id: int | None = None

    def get_token_id_by_modality(self, modality: Modality) -> int | None:
        """Get token ID for a given modality."""
        return {
            Modality.IMAGE: self.image_token_id,
            Modality.VIDEO: self.video_token_id,
            Modality.AUDIO: self.audio_token_id,
        }.get(modality)
