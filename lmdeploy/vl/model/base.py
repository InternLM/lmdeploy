# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from itertools import groupby
from typing import Dict, List, Union

import numpy as np
from mmengine import Registry
from transformers import AutoConfig, AutoTokenizer

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

    @abstractmethod
    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Preprocess multimodal data in the messages.

        The derived class,
        i.e., a specific vision model, takes the charge of image preprocessing
        and the result management.
        It can integrate the result into the messages list, or insert it to
        the individual image item.
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
                {....}
            ]
        Returns:
            the message list with preprocessing results included, which is
            determined by the derived classes
        """  # noqa
        raise NotImplementedError()

    def has_input_ids(self, messages: List[Dict]) -> bool:
        """Check whether the messages contain input_ids directly.

        Args:
            messages (List[Dict]): a list of message, which is supposed to be
                the output of `preprocess`
        Returns:
            bool: whether the messages contain input_ids directly
        """
        users = [x['content'] for x in messages if x['role'] == 'user']
        return len(users) == 1 and isinstance(users[0], List) and isinstance(users[0][0].get('text', ''), List)

    def forward(self, messages: List[Dict], max_batch_size: int = 1) -> List[Dict]:
        """Extract image feature. ONLY implement it when the backend is
        turbomind engine.

        Args:
            messages(List[Dict]): the outputs of `preprocess`
            max_batch_size(int): the max batch size when forwarding vision
                model
        Return:
            the message list with forwarding results included, which is
            determined by the derived classes
        """
        if self.backend == 'turbomind':
            raise NotImplementedError()

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        """Pack the preprocessing results in a format compatible with what is
        required by pytorch engine. ONLY implement it when the backend is
        pytorch engine.

        Args:
            messages(List[Dict]): the output of `preprocess`
            chat_template: the chat template defined in `lmdeploy/model.py`
            tokenzer: the tokenizer model
            sequence_start: starting flag of a sequence
        """
        if self.backend == 'pytorch':
            raise NotImplementedError()

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        """Pack the forwarding results in a format compatible with what is
        required by turbomind engine. ONLY implement it when the backend is
        turbomind engine.

        Args:
            messages(List[Dict]): the output of `preprocess`
            chat_template: the chat template defined in `lmdeploy/model.py`
            tokenzer: the tokenizer model
            sequence_start: starting flag of a sequence
        """
        if self.backend == 'turbomind':
            raise NotImplementedError()

    @staticmethod
    def collect_images(messages):
        """Gather all images along with their respective parameters from the
        messages and compile them into a single list. Each image is converted
        to RGB color space.

        Args:
            messages (List[Tuple[Image, Dict]]): a list of images with their
                corresponding parameters
        """  # noqa
        images = []
        for message in messages:
            content = message['content']
            if not isinstance(content, List):
                continue
            images.extend([(x['image'], {
                k: v
                for k, v in x.items() if k not in {'type', 'image'}
            }) for x in content if x['type'] == 'image'])
        return images

    def to_pytorch_with_input_ids(self, messages):
        """Pack the preprocessing results in a format compatible with what is
        required by pytorch engine when input_ids are provided directly.

        Args:
            messages(List[Dict]): the output of `preprocess`
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
            messages(List[Dict]): the output of `preprocess`
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
            messages(List[Dict]): the output of `preprocess`
            prompt(str): the prompt after applying chat template
            IMAGE_TOKEN(str): a placeholder where image tokens will be
                inserted
            tokenzer: the tokenizer model
            sequence_start: starting flag of a sequence
        """
        # collect image features from messages
        features = [x['content'] for x in messages if x['role'] == 'forward']
        features = features[0]
        features = [x.cpu().numpy() for x in features]
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
