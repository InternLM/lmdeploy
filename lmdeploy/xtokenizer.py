# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABC, abstractmethod

import numpy as np
from mmengine import Registry

from .model import MODELS, SamplingParam

XTOKENIZERS = Registry('xtokenizer', locations=['lmdeploy.xtokenizer'])


class BaseModel(ABC):
    """Base model."""

    def __init__(self,
                 session_len=2048,
                 top_p=0.8,
                 top_k=None,
                 temperature=0.8,
                 repetition_penalty=1.0,
                 stop_words=None,
                 capability='chat',
                 **kwargs):
        self.session_len = session_len
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.stop_words = stop_words
        self.capability = capability

    @abstractmethod
    def query2ids(self, query, sequence_start=True, **kwargs):
        """Return input ids and padding offsets."""

    @abstractmethod
    def messages2ids(self, messages, sequence_start=True, **kwargs):
        """Return input ids and padding offsets.

        user content should be str or (str, int) or [str, int]
        """

    @property
    def sampling_param(self):
        return SamplingParam(top_p=self.top_p,
                             top_k=self.top_k,
                             temperature=self.temperature,
                             repetition_penalty=self.repetition_penalty)


@XTOKENIZERS.register_module(name='qwen-vl')
@XTOKENIZERS.register_module(name='qwen-vl-chat')
class QwenVL(BaseModel):
    """Qwen VL tokenizer."""

    def __init__(self,
                 tokenizer=None,
                 session_len=8192,
                 top_p=0.3,
                 top_k=0,
                 temperature=1.0,
                 im_start='<|im_start|>',
                 im_end='<|im_end|>',
                 system='You are a helpful assistant.',
                 stop_words=['<|im_end|>'],
                 img_start_id=151857,
                 img_end_id=151858,
                 **kwargs):
        super().__init__(**kwargs)
        self.session_len = session_len
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.im_start = im_start
        self.im_end = im_end
        self.system = system
        self.stop_words = stop_words
        self.img_start_id = img_start_id
        self.img_end_id = img_end_id
        self.model = MODELS.get('qwen-7b')(im_start=im_start,
                                           im_end=im_end,
                                           system=system)
        self.tokenizer = tokenizer

    def _construct_query(self, query):
        if isinstance(query, str):
            return query
        query, nimg = query
        text = ''
        for i in range(nimg):
            text += f'Picture {i + 1}:<img>placeholder</img>\n'
        text += query
        return text

    def _get_image_offsets(self, input_ids):
        input_ids = np.array(input_ids)
        offsets = np.where(input_ids == self.img_start_id)[0] + 1
        return offsets.tolist()

    def query2ids(self, query, sequence_start=True, **kwargs):
        text = self._construct_query(query)
        decorated_text = self.model.decorate_prompt(
            text, sequence_start=sequence_start)
        input_ids = self.tokenizer.encode(decorated_text)
        offsets = self._get_image_offsets(input_ids)
        return input_ids, offsets

    def messages2ids(self, messages, sequence_start=True, **kwargs):
        if isinstance(messages, str) or isinstance(messages, (tuple, list)):
            return self.query2ids(messages, sequence_start)
        messages_cp = copy.deepcopy(messages)
        for message in messages_cp:
            msg_role = message['role']
            if msg_role == 'user':
                message['content'] = self._construct_query(message['content'])
        decorated_text = self.model.messages2prompt(
            messages_cp, sequence_start=sequence_start)
        input_ids = self.tokenizer.encode(decorated_text)
        offsets = self._get_image_offsets(input_ids)
        return input_ids, offsets
