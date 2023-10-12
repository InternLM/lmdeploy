# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import queue
import random
import threading
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import List, Union

import google.protobuf.json_format
import mmengine
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.grpc.service_pb2 import ModelInferResponse

from lmdeploy.model import MODELS
from lmdeploy.serve.turbomind.utils import (Postprocessor, Preprocessor,
                                            prepare_tensor)
from lmdeploy.utils import filter_suffix


@dataclass
class Session:
    session_id: Union[int, str]
    request_id: str = ''
    histories: str = ''  # history conversations of the session
    sequence_length: int = 0  # the total generated token number in the session
    prompt: str = ''
    response: str = ''
    status: int = None  # status of the session


class StatusCode(Enum):
    TRITON_STREAM_END = 0  # end of streaming
    TRITON_STREAM_ING = 1  # response is in streaming
    TRITON_SESSION_READY = 2  # session is ready for inference
    TRITON_SERVER_ERR = -1  # triton server's error
    TRITON_SESSION_CLOSED = -2  # session has been closed
    TRITON_SESSION_OUT_OF_LIMIT = -3  # request length out of limit
    TRITON_SESSION_INVALID_ARG = -4  # invalid argument


def stream_callback(que, result, error):
    """callback function invoked by triton client."""
    if error:
        print(error)
        que.put(dict(errcode=StatusCode.TRITON_SERVER_ERR, errmsg=f'{error}'))
    else:
        que.put(result.get_response(as_json=True))


def get_logger(log_file=None, log_level=logging.INFO):
    """Return the logger."""
    from lmdeploy.utils import get_logger
    logger = get_logger('service.ft', log_file=log_file, log_level=log_level)
    return logger


class Chatbot:
    """Chatbot for LLaMA series models with turbomind as inference engine.

    Args:
        tritonserver_addr (str): communicating address '<ip>:<port>' of
            triton inference server
        model_name (str): name of the to-be-deployed mode
        log_level (int): the level of the log
        display (bool): display the generated text on consolo or not
        profile_generation (bool): profile token generation or not
    """

    def __init__(self,
                 tritonserver_addr: str,
                 model_name: str = '',
                 ignore_eos: bool = False,
                 log_level: int = logging.INFO,
                 display: bool = False,
                 profile_generation: bool = False,
                 profile_serving: bool = False,
                 **model_kwargs):
        self.tritonserver_addr = tritonserver_addr
        self.model_name = model_name
        if self.model_name == '':
            self.model_name = self._get_model_name()
        assert self.model_name in MODELS.module_dict.keys(), \
            f"'{self.model_name}' is not supported. " \
            f'The supported models are: {MODELS.module_dict.keys()}'
        self.model = MODELS.get(self.model_name)(**model_kwargs)
        self._session = None
        self.preprocess = Preprocessor(tritonserver_addr)
        self.postprocess = Postprocessor(tritonserver_addr)
        self.bos_id = self._get_bos()
        self.eos_id = self._get_eos()
        stop_words = self._stop_words(self.model.stop_words)
        bad_words = None
        if ignore_eos:
            stop_words = None
            bad_words = np.array([[[self.eos_id], [1]]], dtype=np.int32)
        self.cfg = mmengine.Config(
            dict(session_len=self.model.session_len,
                 top_p=self.model.top_p,
                 top_k=self.model.top_k,
                 temperature=self.model.temperature,
                 repetition_penalty=self.model.repetition_penalty,
                 stop_words=stop_words,
                 bad_words=bad_words))
        self.log_level = log_level
        self.display = display
        self.profile_generation = profile_generation
        self.profile_serving = profile_serving

    def stream_infer(self,
                     session_id: int,
                     prompt: str,
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

        self._session.status = 1
        self._session.request_id = request_id
        self._session.response = ''
        self.cfg.update(**kwargs)

        self._session.prompt = self._get_prompt(prompt, sequence_start)
        for status, res, tokens in self._stream_infer(self._session,
                                                      self._session.prompt,
                                                      request_output_len,
                                                      sequence_start,
                                                      sequence_end):
            if status == StatusCode.TRITON_STREAM_END:  # remove stop_words
                res = filter_suffix(res, self.model.stop_words)
            if status.value < 0:
                break
            else:
                yield status, res, tokens
        if status.value == 0:
            self._session.histories = \
                self._session.histories + self._session.prompt + \
                self._session.response
        else:
            yield status, res, tokens

    def end(self, session_id: int, *args, **kwargs):
        """end a session. Triton inference server will release the session's
        occupied resource when it is ended.

        Args:
            session_id (int): the identical id of a session

        Returns:
            int: 0: success, -1: session not found
        """
        assert isinstance(session_id, int), \
            f'INT session id is required, but got {type(session_id)}'

        logger = get_logger(log_level=self.log_level)
        logger.info(f'end session: {session_id}')

        if self._session is None:
            logger.error(
                f"session {session_id} doesn't exist. It cannot be ended")
            return StatusCode.TRITON_SESSION_INVALID_ARG
        if self._session.session_id != session_id:
            logger.error(f'you cannot end session {session_id}, because this '
                         f'session is {self._session.session_id}')
            return StatusCode.TRITON_SESSION_INVALID_ARG
        if self._session.status == 0:
            logger.warning(f'session {session_id} has already been ended')
            return StatusCode.TRITON_SESSION_CLOSED

        self._session.status = 0
        for status, _, _ in self._stream_infer(self._session,
                                               prompt='',
                                               request_output_len=0,
                                               sequence_start=False,
                                               sequence_end=True):
            if status.value < 0:
                break

        self.reset_session()
        return status

    def cancel(self, session_id: int, *args, **kwargs):
        """Cancel the session during generating tokens.

        Args:
            session_id (int): the identical id of a session

        Returns:
            int: 0: success, -1: session not found
        """
        assert isinstance(session_id, int), \
            f'INT session id is required, but got {type(session_id)}'
        logger = get_logger(log_level=self.log_level)
        logger.info(f'cancel session: {session_id}')

        if self._session is None:
            logger.error(
                f"session {session_id} doesn't exist. It cannot be cancelled")
            return StatusCode.TRITON_SESSION_INVALID_ARG
        if self._session.session_id != session_id:
            logger.error(
                f'you cannot cancel session {session_id}, because this '
                f'session is {self._session.session_id}')
            return StatusCode.TRITON_SESSION_INVALID_ARG
        if self._session.status == 0:
            logger.error(f'session {session_id} has already been ended. '
                         f'It cannot be cancelled')
            return StatusCode.TRITON_SESSION_CLOSED

        prev_session = self._session
        status, res = None, None
        for status, res, _ in self._stream_infer(self._session,
                                                 prompt='',
                                                 request_output_len=0,
                                                 sequence_start=False,
                                                 sequence_end=False,
                                                 cancel=True):
            if status.value < 0:
                break
        if status == StatusCode.TRITON_STREAM_END:
            logger.info(f'cancel session {session_id} successfully')
            if prev_session.histories:
                logger.warning(f'TODO: start to recover session {session_id}')
        else:
            logger.info(f'cancel session {session_id} failed: {res}')
        return status

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
        histories = self._session.histories
        for status, _, _ in self._stream_infer(self._session,
                                               prompt=histories,
                                               request_output_len=0,
                                               sequence_start=True,
                                               sequence_end=False):
            if status.value < 0:
                break

        self._session.histories = histories
        return status

    def infer(self,
              session_id: int,
              prompt: str,
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

        self._session.status = 1
        self._session.request_id = request_id
        self._session.response = ''

        self._session.prompt = self._get_prompt(prompt, sequence_start)
        status, res, tokens = None, '', 0
        for status, res, tokens in self._stream_infer(self._session,
                                                      self._session.prompt,
                                                      request_output_len,
                                                      sequence_start,
                                                      sequence_end):
            if status.value < 0:
                break
            if status == StatusCode.TRITON_STREAM_END:  # remove stop_words
                res = filter_suffix(res, self.model.stop_words)
        if status.value == 0:
            self._session.histories = \
                self._session.histories + self._session.prompt + \
                self._session.response
            return status, res, tokens
        else:
            return status, res, tokens

    def reset_session(self):
        """reset session."""
        self._session = None

    @property
    def session(self):
        """get session."""
        return self._session

    @session.setter
    def session(self, value):
        """set session."""
        self._session = value

    def _get_model_name(self):
        with grpcclient.InferenceServerClient(
                self.tritonserver_addr) as client:
            model_config = client.get_model_config(model_name='turbomind',
                                                   as_json=True)
            return model_config['config']['parameters']['model_name'][
                'string_value']

    def _get_bos(self):
        """return bos token id."""
        token_ids, _ = self.preprocess('<BOS>')
        return token_ids[0][0]

    def _get_eos(self):
        """return eos token id."""
        token_ids, _ = self.preprocess('<EOS>')
        return token_ids[0][0]

    def _stop_words(self, stop_words: List[str]):
        """return stop-words' token ids."""
        if stop_words is None:
            return None
        assert isinstance(stop_words, List) and \
               all(isinstance(elem, str) for elem in stop_words), \
               f'stop_words must be a list but got {type(stop_words)}'
        # each id in stop_words represents a stop word
        # refer to https://github.com/fauxpilot/fauxpilot/discussions/165 for
        # detailed explanation about turbomind's stop_words
        stop_words = [
            int(self.preprocess(stop_word)[0][0][-1])
            for stop_word in stop_words
        ]
        assert isinstance(stop_words, List) and \
               all(isinstance(elem, int) for elem in stop_words), \
               'invalid stop_words'
        stop_word_offsets = range(1, len(stop_words) + 1)
        stop_words = np.array([[stop_words,
                                stop_word_offsets]]).astype(np.int32)
        return stop_words

    def _get_prompt(self, prompt: str, sequence_start: bool):
        """return the concatenated prompt according to the model's chat
        template."""
        if self.profile_generation or self.profile_serving:
            return prompt
        return self.model.get_prompt(prompt, sequence_start)

    def _stream_infer(self,
                      session: Session,
                      prompt: str,
                      request_output_len: int = 512,
                      sequence_start: bool = True,
                      sequence_end: bool = False,
                      cancel: bool = False):
        """communicate with inference server to chat, or cancel a session, or
        end a session.

        Args:
            session (Session): an instance of a session
            prompt (str): the concatenated prompt
            request_output_len (int): the max number of tokens to be generated
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
            cancel (bool): indicator for cancelling the session
        Yields:
            tuple: status, text, generated token number
        """
        logger = get_logger(log_level=self.log_level)
        logger.info(f'session {session.session_id}, '
                    f'request id {session.request_id}, '
                    f'request_output_len {request_output_len}, '
                    f'start {sequence_start}, '
                    f'end {sequence_end}, cancel {cancel}')

        assert request_output_len is None or \
               isinstance(request_output_len, int), \
               f'request_output_len is supposed to be None or int, ' \
               f'but got {type(request_output_len)}'

        if sequence_start:
            logger.info(f'session {session.session_id}, clear history since '
                        f'sequence_start is True')
            session.histories = ''
            session.sequence_length = 0

        input_ids, input_lengths = self.preprocess(prompt)
        input_tokens = input_lengths.squeeze()
        if self.profile_generation:
            yield StatusCode.TRITON_STREAM_ING, \
                  'ignore preprocessing during profiling generation', 0
        if request_output_len is None:
            request_output_len = max(
                128,
                self.cfg.session_len - session.sequence_length - input_tokens)

        if input_tokens + request_output_len + \
                session.sequence_length > self.cfg.session_len:
            errmsg = f'session {session.session_id}, ' \
                     f'out of max sequence length {self.cfg.session_len}, ' \
                     f'#input tokens {input_tokens}, ' \
                     f'history tokens {session.sequence_length}, ' \
                     f'request length {request_output_len}'
            logger.warning(errmsg)
            yield StatusCode.TRITON_SESSION_OUT_OF_LIMIT, errmsg, 0
            return

        logger.info(f'session {session.session_id}, '
                    f'max length: {self.cfg.session_len}, '
                    f'input tokens: {input_tokens}, '
                    f'request tokens: {request_output_len}, '
                    f'history tokens: {session.sequence_length}')

        preseq_length = session.sequence_length
        session.response = ''
        session.status = StatusCode.TRITON_SESSION_READY

        que = queue.Queue()
        producer = threading.Thread(target=self._stream_producer,
                                    args=(self.tritonserver_addr, session, que,
                                          self.cfg, input_ids, input_lengths,
                                          request_output_len, sequence_start,
                                          sequence_end, preseq_length, cancel))
        producer.start()
        for status, res, n_token in self.stream_consumer(
                self.postprocess, que, session, input_tokens, preseq_length,
                cancel, logger, self.display, self.profile_generation,
                self.eos_id):
            yield status, res, n_token

        producer.join()
        self._session = que.get()
        curseq_length = self._session.sequence_length
        logger.info(f'session {session.session_id}, pre seq_len '
                    f'{preseq_length}, cur seq_len {curseq_length}, '
                    f'diff {curseq_length - preseq_length}')

    @staticmethod
    def _stream_producer(tritonserver_addr, session, que, cfg, input_ids,
                         input_lengths, request_output_len, sequence_start,
                         sequence_end, preseq_length, cancel):
        """Send a request to the triton inference server.

        Args:
            tritonserver_addr (str): the communication address of the inference
                server
            session (Session): an instance of a session
            que (multiprocessing.Queue): response queue
            cfg (dict): parameters for sampling
            input_ids (numpy.ndarray): token ids of input prompt
            input_lengths (numpy.ndarray): length of input_ids
            request_output_len (int): the max number of tokens to be generated
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
            preseq_length (int): the history sequence length
            cancel (bool): indicator for cancelling the session
        """
        request_output_len = np.full(input_lengths.shape,
                                     request_output_len).astype(np.uint32)

        callback = partial(stream_callback, que)
        with grpcclient.InferenceServerClient(tritonserver_addr) as client:
            inputs = [
                prepare_tensor('input_ids', input_ids),
                prepare_tensor('input_lengths', input_lengths),
                prepare_tensor('request_output_len', request_output_len),
                prepare_tensor('runtime_top_p',
                               cfg.top_p * np.ones((1, 1), dtype=np.float32)),
                prepare_tensor(
                    'temperature',
                    cfg.temperature * np.ones((1, 1), dtype=np.float32)),
                prepare_tensor(
                    'repetition_penalty',
                    cfg.repetition_penalty * np.ones(
                        (1, 1), dtype=np.float32)),
                prepare_tensor('step',
                               preseq_length * np.ones((1, 1), dtype=np.int32))
            ]
            if cfg.top_k is not None:
                inputs += prepare_tensor(
                    'runtime_top_k',
                    cfg.top_k * np.ones((1, 1), dtype=np.uint32)),
            if cfg.stop_words is not None:
                inputs += [prepare_tensor('stop_words_list', cfg.stop_words)]
            if cfg.bad_words is not None:
                inputs += [prepare_tensor('bad_words_list', cfg.bad_words)]

            inputs += [
                prepare_tensor(
                    'session_len',
                    cfg.session_len *
                    np.ones([input_ids.shape[0], 1], dtype=np.uint32)),
                prepare_tensor('START', (1 if sequence_start else 0) * np.ones(
                    (1, 1), dtype=np.int32)),
                prepare_tensor('END', (1 if sequence_end else 0) * np.ones(
                    (1, 1), dtype=np.int32)),
                prepare_tensor(
                    'CORRID',
                    session.session_id * np.ones((1, 1), dtype=np.uint64)),
                prepare_tensor('STOP', (1 if cancel else 0) * np.ones(
                    (1, 1), dtype=np.int32))
            ]
            if sequence_start:
                random_seed = random.getrandbits(64)
                inputs += [
                    prepare_tensor(
                        'random_seed',
                        random_seed * np.ones((1, 1), dtype=np.uint64))
                ]
            client.start_stream(callback)
            client.async_stream_infer('turbomind',
                                      inputs,
                                      sequence_id=session.session_id,
                                      request_id=session.request_id,
                                      sequence_start=sequence_start,
                                      sequence_end=sequence_end)
        que.put(None)

    @staticmethod
    def stream_consumer(postprocess, res_queue, session, n_input_token,
                        preseq_length, cancel, logger, display,
                        profile_generation, eos_id):
        """Consume the response from the triton inference server.

        Args:
            postprocess (callable): postprocess function for
                the generated tokens
            res_queue (multiprocessing.Queue): response queue
            session (Session): an instance of a session
            n_input_token (int): token number of input prompt
            preseq_length (int): the history sequence length
            cancel (bool): indicator for cancelling the session
            logger (util.Logger):
            display (bool): display the text in the consolo interface or not
            profile_generation (bool): indicator for profiling token generation
            eos_id (int): eos token id

        Yields:
            tuple: status, text, generated token number
        """
        status, res, n_token = None, '', 0
        while True:
            result = res_queue.get()
            if result is None:
                status = StatusCode.TRITON_STREAM_END
                res = session.response
                session.status = StatusCode.TRITON_STREAM_END
                break
            if 'errcode' in result:
                logger.error(f'got error from turbomind, code '
                             f"{result['errcode']}, {result['errmsg']}, "
                             f'token {session.sequence_length}')
                session.sequence_length = preseq_length
                session.response = ''
                status = StatusCode.TRITON_SERVER_ERR
                res = f"{result['errcode']}, {result['errmsg']}"
                n_token = 0
                break
            if cancel:
                continue
            try:
                message = ModelInferResponse()
                google.protobuf.json_format.Parse(json.dumps(result), message)
                result = grpcclient.InferResult(message)
                sequence_length = result.as_numpy('sequence_length')
                output_ids = result.as_numpy('output_ids')

                session.sequence_length = sequence_length.squeeze()
                output_ids = output_ids.reshape((1, 1, output_ids.shape[-1]))
                output_ids = output_ids[:, :, n_input_token +
                                        preseq_length:sequence_length.squeeze(
                                        )]
                last_token_id = None if output_ids.shape[
                    -1] == 0 else output_ids[-1, -1, -1]
                if last_token_id == eos_id:
                    session.sequence_length = session.sequence_length - 1
                    output_ids = output_ids[:, :, :-1]

                if profile_generation:
                    yield (StatusCode.TRITON_STREAM_ING,
                           'postprocessing is ignored during profiling '
                           'token generation', output_ids.shape[-1])
                    continue
                output_str = postprocess(
                    output_ids, np.array([[n_token]], dtype=np.uint32))
                n_token = output_ids.shape[-1]
                text = output_str[0].decode()
                if display:
                    print(text, end='', flush=True)
                session.response += text
                yield (StatusCode.TRITON_STREAM_ING, session.response,
                       output_ids.shape[-1])
            except Exception as e:
                logger.error(f'catch exception: {e}')
                logger.error(
                    f'session {session.session_id}: prompt: {session.prompt}')

        # put session back to queue so that `_stream_infer` can update it in
        # `self.sessions`
        while not res_queue.empty():
            res_queue.get()
        res_queue.put(session)
        if display:
            print('\n')
        yield status, res, n_token
