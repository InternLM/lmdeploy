# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np
import torch
from mmengine import Registry
from transformers import AutoConfig

from lmdeploy.archs import get_model_arch

VISION_MODELS = Registry('vision_model')


class VisonModel(ABC):
    """Visual model which extract image feature."""
    _arch: Union[str, List[str]] = None

    def __init__(self,
                 model_path: str,
                 with_llm: bool = False,
                 max_memory: Dict[int, int] = None,
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

    @abstractmethod
    def build_preprocessor(self, ):
        raise NotImplementedError()

    @abstractmethod
    def build_model(self, ):
        """build model."""
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """preprocess multimodal data in the messages, of which only the last
        item includes the mulitmodal data.

        Args:
            message(Dict): multimodal data in a dict, which is as follows:
            [
                {'role': 'user', 'content': 'user prompt'},
                {'role': 'assisant', 'content': 'AI reponse'},
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': 'string',
                        },
                        {
                            'type': 'image',
                            'image': pillow.Image,
                            'key1': value1,
                            ...
                        },
                        {
                            'type': 'image',
                            'image': pillow.Image,
                            'key1': value1,
                            ...
                        },
                        ...
                    ]
                }
            ]
        Returns:
            the preprocessing results in a list. list[i] is a dict, referring
            to the preprocessing result of an image. The dict acts like:
            {
                'pixel_values': torch.Tensor,
                'others_output_by_image_preprocessing': torch.Tensor or else,
                ...,
                'image_tokens': int, # the number of tokens that the corresponding image encoded,
                'image_token_id': int, #
            }
        """  # noqa
        raise NotImplementedError()

    @abstractmethod
    def forward(self, inputs: List[Dict]) -> List[torch.Tensor]:
        """extract image feature.

        Args:
            inputs: the outputs of `preprocess`
        Return:
            A list of torch.Tensor. Each one represents the feature of an
                image
        """
        raise NotImplementedError()

    @abstractmethod
    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        """"""
        raise NotImplementedError()

    @abstractmethod
    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start):
        """"""
        raise NotImplementedError()

    @classmethod
    def to_pytorch_aux(cls, messages, prompt, IMAGE_TOKEN, tokenizer,
                       sequence_start):
        """"""
        # collect all preprocessing result from messages
        preps = [
            message.pop('preprocess') for message in messages
            if 'preprocess' in message.keys()
        ]
        # flatten the list
        preps = list(itertools.chain(*preps))

        # split prompt into segments and validate data
        segs = prompt.split(IMAGE_TOKEN)
        assert len(segs) == len(preps) + 1, (
            f'the number of {IMAGE_TOKEN} is not equal '
            f'to input images, {len(segs) - 1} vs {len(preps)}')

        # calculate the image token offset for each image
        input_ids = []
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(preps):
                preps[i - 1].update(offset=len(input_ids))
                image_tokens = preps[i - 1]['image_tokens']
                image_token_id = preps[i - 1]['image_token_id']
                input_ids.extend([image_token_id] * image_tokens)
            token_ids = tokenizer.encode(seg,
                                         add_bos=((i == 0) and sequence_start))
            input_ids.extend(token_ids)

        return dict(prompt=prompt, input_ids=input_ids, multimodal=preps)

    @classmethod
    def to_turbomind_aux(cls, messages, prompt, IMAGE_TOKEN, tokenizer,
                         sequence_start):
        # collect image features from messages
        features = [
            message.pop('forward') for message in messages
            if 'forward' in message.keys()
        ]
        # flatten the list
        features = list(itertools.chain(*features))
        features = [x.cpu().numpy() for x in features]

        # split prompt into segments and validate data
        segs = prompt.split(IMAGE_TOKEN)
        assert len(segs) == len(features) + 1, (
            f'the number of {IMAGE_TOKEN} is not equal '
            f'to input images, {len(segs) - 1} vs {len(features)}')

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

    @classmethod
    def match(cls, config: AutoConfig):
        """check whether the config match the model."""
        arch = config.architectures[0]
        if arch == cls._arch or arch in cls._arch:
            return True
        return False
