# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import numpy as np

from lmdeploy.serve.turbomind.utils import (Postprocessor, XPreprocessor,
                                            prepare_tensor)
from lmdeploy.xtokenizer import XTOKENIZERS

from .chatbot import Chatbot, Session, StatusCode, filter_suffix, get_logger


class ImChatbot(Chatbot):

    MODEL_REGISTRY = XTOKENIZERS

    def _init_prepost_processor(self):
        tritonserver_addr = self.tritonserver_addr
        self.preprocess = XPreprocessor(tritonserver_addr)
        self.postprocess = Postprocessor(tritonserver_addr)

    def _init_cfg(self, **model_kwargs):
        super()._init_cfg(**model_kwargs)
        self.img_start_id = self.model.img_start_id
        self.img_end_id = self.model.img_end_id

    def stream_infer(self,
                     session_id: int,
                     prompt: str,
                     image_embs: List[np.array] = None,
                     request_id: str = '',
                     request_output_len: int = None,
                     sequence_start: bool = False,
                     sequence_end: bool = False,
                     *args,
                     **kwargs):
        """Start a new round conversion of a session.

        Args:
            session_id (int): the identical id of a session
            prompt (str): user's prompt in this round conversation
            image_embs (List[np.array]): image embedding features in this
                round conversation
            request_id (str): the identical id of this round conversation
            request_output_len (int): the expected generated token numbers
            sequence_start (bool): start flag of a session
            sequence_end (bool): end flag of a session
        Returns:
            iterator: The generated content by chatbot
        """

        assert isinstance(session_id, int), \
            f'INT session id is required, but got {type(session_id)}'

        logger = get_logger(log_level=self.log_level)
        logger.info(f'session {session_id}, request_id {request_id}, '
                    f'request_output_len {request_output_len}')

        if self._session is None:
            sequence_start = True
            self._session = Session(session_id=session_id)
        elif self._session.status == 0:
            logger.error(f'session {session_id} has been ended. Please set '
                         f'`sequence_start` be True if you want to restart it')
            yield StatusCode.TRITON_SESSION_CLOSED, '', 0
            return

        self.cfg.update(**kwargs)
        self._session.status = 1
        self._session.request_id = request_id
        self._session.response = ''
        self._session.response_ids = []
        preseq_length = self._session.sequence_length

        if self._session.history_ids is None or sequence_start:
            self._session.history_ids = []
            self._session.image_embs = []
            self._session.image_offsets = []

        self._session.prompt = prompt
        input_ids, _, image_offsets = self.preprocess(
            self._session.prompt,
            sequence_start=sequence_start,
            num_image=len(image_embs) if image_embs is not None else 0)

        for status, res, tokens in self._stream_infer(
                self._session,
                input_ids,
                request_output_len,
                sequence_start,
                sequence_end,
                image_embs=image_embs,
                image_offsets=image_offsets):
            if status == StatusCode.TRITON_STREAM_END:  # remove stop_words
                res = filter_suffix(res, self.model.stop_words)
            if status.value < 0:
                break
            else:
                yield status, res, tokens
        if status.value == 0:
            self._session.history_ids.extend(input_ids.flatten().tolist() +
                                             self._session.response_ids)
            if image_embs is not None:
                self._session.image_embs.extend(image_embs)
                self._session.image_offsets.extend(
                    (image_offsets + preseq_length).tolist())
        else:
            yield status, res, tokens

    def _create_input(self,
                      session,
                      cfg,
                      input_ids,
                      input_lengths,
                      request_output_len,
                      sequence_start,
                      sequence_end,
                      preseq_length,
                      cancel,
                      image_embs=None,
                      image_offsets=None,
                      **kwargs):
        inputs = super()._create_input(session, cfg, input_ids, input_lengths,
                                       request_output_len, sequence_start,
                                       sequence_end, preseq_length, cancel)
        if image_embs is not None:
            image_embs = [x.squeeze()[None] for x in image_embs]
            image_embs = np.concatenate(image_embs, axis=0)[None]
            image_embs = image_embs.astype(np.float16)
            inputs += [
                prepare_tensor('image_embs', image_embs),
                prepare_tensor('image_offsets', image_offsets),
            ]
        return inputs

    def resume(self, session_id: int, *args, **kwargs):
        """Resume a session by sending the history conversations to triton
        inference server. After resuming, users can continue chatting with
        chatbot.

        Args:
            session_id (int): the identical id of a session

        Returns:
            int: 0: success, -1: session not found
        """
        assert isinstance(session_id, int), \
            f'INT session id is required, but got {type(session_id)}'

        logger = get_logger(log_level=self.log_level)
        logger.info(f'resume session: {session_id}')

        if self._session is None:
            logger.error(
                f"session {session_id} doesn't exist. It cannot be recovered")
            return StatusCode.TRITON_SESSION_INVALID_ARG
        if self._session.session_id != session_id:
            logger.error(
                f'you cannot resume session {session_id}, because this '
                f'session is {self._session.session_id}')
            return StatusCode.TRITON_SESSION_INVALID_ARG

        self._session.status = 1
        self._session.sequence_length = 0
        input_ids = self._session.history_ids
        image_embs = self._session.image_embs
        image_offsets = self._session.image_offsets

        for status, _, _ in self._stream_infer(self._session,
                                               input_ids=input_ids,
                                               request_output_len=0,
                                               sequence_start=True,
                                               sequence_end=False,
                                               image_embs=image_embs,
                                               image_offsets=image_offsets):
            if status.value < 0:
                break

        return status

    def infer(self,
              session_id: int,
              prompt: str,
              image_embs: List[np.array] = None,
              request_id: str = '',
              request_output_len: int = None,
              sequence_start: bool = False,
              sequence_end: bool = False,
              *args,
              **kwargs):
        """Start a new round conversion of a session. Return the chat
        completions in non-stream mode.

        Args:
            session_id (int): the identical id of a session
            prompt (str): user's prompt in this round conversation
            image_embs (List[np.array]): image embedding features in this
                round conversation
            request_id (str): the identical id of this round conversation
            request_output_len (int): the expected generated token numbers
            sequence_start (bool): start flag of a session
            sequence_end (bool): end flag of a session
        Returns:
            tuple(Status, str, int): status, text/chat completion,
            generated token number
        """
        assert isinstance(session_id, int), \
            f'INT session id is required, but got {type(session_id)}'

        logger = get_logger(log_level=self.log_level)
        logger.info(f'session {session_id}, request_id {request_id}, '
                    f'request_output_len {request_output_len}')

        if self._session is None:
            sequence_start = True
            self._session = Session(session_id=session_id)
        elif self._session.status == 0:
            logger.error(f'session {session_id} has been ended. Please set '
                         f'`sequence_start` be True if you want to restart it')
            return StatusCode.TRITON_SESSION_CLOSED, '', 0

        self.cfg.update(**kwargs)
        self._session.status = 1
        self._session.request_id = request_id
        self._session.response = ''
        self._session.response_ids = []
        preseq_length = self._session.sequence_length

        if self._session.history_ids is None or sequence_start:
            self._session.history_ids = []
            self._session.image_embs = []
            self._session.image_offsets = []
        self._session.prompt = prompt
        input_ids, _, image_offsets = self.preprocess(
            self._session.prompt,
            sequence_start=sequence_start,
            num_image=len(image_embs) if image_embs is not None else 0)
        status, res, tokens = None, '', 0
        for status, res, tokens in self._stream_infer(
                self._session,
                input_ids,
                request_output_len,
                sequence_start,
                sequence_end,
                image_embs=image_embs,
                image_offsets=image_offsets):
            if status.value < 0:
                break
            if status == StatusCode.TRITON_STREAM_END:  # remove stop_words
                res = filter_suffix(res, self.model.stop_words)
        if status.value == 0:
            self._session.history_ids.extend(input_ids.flatten().tolist() +
                                             self._session.response_ids)
            if image_embs is not None:
                self._session.image_embs.extend(image_embs)
                self._session.image_offsets.extend(
                    (image_offsets + preseq_length).tolist())

        return status, res, tokens
