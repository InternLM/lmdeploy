# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
from abc import ABC, abstractmethod
from itertools import groupby
from typing import Any

import numpy as np
from mmengine import Registry
from transformers import AutoConfig, AutoTokenizer

from lmdeploy.archs import get_model_arch
from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.preprocess_utils import (
    get_expanded_input_ids,
    get_expanded_mm_items,
    get_mm_items_offset,
    get_override_size,
)

VISION_MODELS = Registry('vision_model')


class VisionModel(ABC):
    """Visual model which extract image feature."""
    _arch: str | list[str] = None

    # mapping from processor output attribute names to modality types
    ATTR_NAME_TO_MODALITY = {
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
        # time series-related attributes
        'ts_values': Modality.TIME_SERIES,
        'ts_sr': Modality.TIME_SERIES,
        'ts_lens': Modality.TIME_SERIES,
    }

    # processor output attributes that carry the main feature tensor
    FEATURE_NAMES = [
        'pixel_values',
        'pixel_values_videos',
        'audio_features',
        'input_features',
        'ts_values',
    ]

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

    def preprocess(self,
                   messages: list[dict],
                   input_prompt: str | list[int],
                   mm_processor_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Preprocess multimodal data and return a dict with ``input_ids`` and
        multimodal features.

        New-style models inherit this implementation. Legacy models override with `def preprocess(self, messages)`.
        """

        mm_items = self.collect_multimodal_items(messages)

        raw_images, raw_videos, video_metadatas = [], [], []
        raw_time_series, sampling_rates = [], []
        for modality, data, params in mm_items:
            if modality == Modality.IMAGE:
                raw_images.append(data)
            elif modality == Modality.VIDEO:
                raw_videos.append(data)
                video_metadatas.append(params.get('video_metadata', None))
            elif modality == Modality.TIME_SERIES:
                raw_time_series.append(data)
                sampling_rates.append(params.get('sampling_rate', None))
            else:
                raise ValueError(f'unsupported modality {modality}')

        # get kwargs for processor
        kwargs = {}
        images_kwargs = {}
        videos_kwargs = {}
        mm_processor_kwargs = mm_processor_kwargs or {}
        if raw_images:
            kwargs['images'] = raw_images
            image_size = get_override_size(self.processor.image_processor,
                                           mm_processor_kwargs.get('image'),
                                           modality='image')
            if image_size is not None:
                images_kwargs['size'] = image_size
        if raw_videos:
            kwargs['videos'] = raw_videos
            videos_kwargs['video_metadata'] = video_metadatas
            # perform resize in hf processor, while sample frames has been done in video loader
            videos_kwargs['do_resize'] = True
            videos_kwargs['do_sample_frames'] = False
            video_size = get_override_size(self.processor.video_processor,
                                           mm_processor_kwargs.get('video'),
                                           modality='video')
            if video_size is not None:
                videos_kwargs['size'] = video_size
        if images_kwargs:
            kwargs['images_kwargs'] = images_kwargs
        if videos_kwargs:
            kwargs['videos_kwargs'] = videos_kwargs
        if raw_time_series:
            assert hasattr(self, 'time_series_processor'), \
                'time series processor is not defined for time series input'
            assert not raw_images and not raw_videos, \
                'time series is not compatible with image/video input'
            self.tokenizer = self.processor.tokenizer
            time_series_processor = self.time_series_processor
            kwargs['time_series'] = raw_time_series
            kwargs['sampling_rate'] = sampling_rates

        # process raw items with hf processor
        input_text = input_prompt if isinstance(input_prompt, str) else ''
        processor_outputs = (time_series_processor if raw_time_series else self.processor)(
            text=[input_text],
            padding=True,
            return_tensors='pt',
            **kwargs,
        )

        # collect items from hf processor outputs and categorized by modality for lmdeploy to consume
        collected_mm_items: dict[Modality, dict[str, Any]] = {}
        for attr_name, value in processor_outputs.items():
            current_modality = self.ATTR_NAME_TO_MODALITY.get(attr_name)
            if current_modality:
                if current_modality not in collected_mm_items:
                    collected_mm_items[current_modality] = {}

                if attr_name in self.FEATURE_NAMES:
                    attr_name = 'feature'

                collected_mm_items[current_modality][attr_name] = value

        # get input_ids, expand multimodal tokens only when we receive input ids from /generate endpoint
        if isinstance(input_prompt, str):
            input_ids = processor_outputs['input_ids'].flatten()
        else:
            input_ids = get_expanded_input_ids(input_prompt, collected_mm_items, self.processor, self.mm_tokens)

        # compute offsets for all items
        for modality, item in collected_mm_items.items():
            mm_token_id = self.mm_tokens.get_token_id_by_modality(modality)
            item['offset'] = get_mm_items_offset(input_ids=input_ids, mm_token_id=mm_token_id)

        # expand bundled hf processor outputs into per-image/video entry for lmdeploy to consume
        expanded_mm_items = get_expanded_mm_items(collected_mm_items, self.mm_tokens)

        return dict(input_ids=input_ids.tolist(), multimodal=expanded_mm_items)

    @staticmethod
    def has_input_ids(messages: list[dict]) -> bool:
        """Check whether the messages contain input_ids directly.

        This is True when the first (and only) user message content is a list
        whose first item carries a ``text`` field that is itself a list of
        token ids (i.e. the output of the ``/generate`` endpoint rather than a
        plain text prompt).

        Args:
            messages (list[dict]): a list of message, which is supposed to be
                the output of `preprocess`
        Returns:
            bool: whether the messages contain input_ids directly
        """
        user_contents = [x['content'] for x in messages if x['role'] == 'user']
        if len(user_contents) != 1:
            return False
        content = user_contents[0]
        if not isinstance(content, list) or not content:
            return False
        first_item = content[0]
        return isinstance(first_item, dict) and isinstance(first_item.get('text'), list)

    @staticmethod
    def get_input_prompt(messages: list[dict], chat_template, sequence_start: bool,
                         chat_template_kwargs: dict | None = None) -> str | list[int]:
        """Return the input prompt for the preprocessor.

        When the messages already carry embedded token ids (from the
        ``/generate`` endpoint), extract and return them directly.
        Otherwise, render the messages through *chat_template* to produce a
        plain-text prompt string.

        Args:
            messages: Preprocessed message list.
            chat_template: Chat template used to render a text prompt.
            sequence_start: Whether this is the start of a new sequence.
            chat_template_kwargs: Extra kwargs forwarded to ``messages2prompt``.
        Returns:
            A list of token ids when input_ids are embedded, otherwise a str.
        """
        if VisionModel.has_input_ids(messages):
            return messages[0]['content'][0]['text']
        return chat_template.messages2prompt(messages, sequence_start, **(chat_template_kwargs or {}))

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

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start, chat_template_kwargs=None, **kwargs):
        """Pack the preprocessing results in a format compatible with what is
        required by pytorch engine. ONLY implement it when the backend is
        pytorch engine.

        Args:
            messages(list[dict]): the output of `preprocess`
            chat_template: the chat template defined in `lmdeploy/model.py`
            tokenzer: the tokenizer model
            sequence_start: starting flag of a sequence
            chat_template_kwargs: additional arguments for chat template
                processing, such as `add_vision_id` and `enable_thinking`
        """
        if self.backend == 'pytorch':
            raise NotImplementedError()

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

    def to_pytorch_aux(self, messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start):
        """Auxiliary function to pack the preprocessing results in a format
        compatible with what is required by pytorch engine.

        Args:
            messages(list[dict]): the output of `preprocess`
            prompt(str): the prompt after applying chat template
            IMAGE_TOKEN(str): a placeholder where image tokens will be
                inserted
            tokenzer: the tokenizer model
            sequence_start: starting flag of a sequence
        """
        # collect all preprocessing result from messages
        preps = [x['content'] for x in messages if x['role'] == 'preprocess']
        assert len(preps) == 1
        preps = preps[0]

        # split prompt into segments and validate data
        segs = prompt.split(IMAGE_TOKEN)
        assert len(segs) == len(preps) + 1, (f'the number of {IMAGE_TOKEN} is not equal '
                                             f'to input images, {len(segs) - 1} vs {len(preps)}')

        # calculate the image token offset for each image
        input_ids = []
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(preps):
                preps[i - 1].update(offset=len(input_ids))
                image_tokens = preps[i - 1]['image_tokens']
                assert self.image_token_id == preps[i - 1]['image_token_id']
                input_ids.extend([self.image_token_id] * image_tokens)
            token_ids = tokenizer.encode(seg, add_bos=((i == 0) and sequence_start))
            input_ids.extend(token_ids)

        return dict(prompt=prompt, input_ids=input_ids, multimodal=preps)

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
class MultimodalSpecialTokens:
    image_token: str | None = None
    video_token: str | None = None
    audio_token: str | None = None
    ts_token: str | None = None

    image_token_id: int | None = None
    video_token_id: int | None = None
    audio_token_id: int | None = None
    ts_token_id: int | None = None

    def get_token_id_by_modality(self, modality: Modality) -> int | None:
        """Get token ID for a given modality."""
        return {
            Modality.IMAGE: self.image_token_id,
            Modality.VIDEO: self.video_token_id,
            Modality.AUDIO: self.audio_token_id,
            Modality.TIME_SERIES: self.ts_token_id,
        }.get(modality)
