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

from llmdeploy.serve.fastertransformer.utils import (Postprocessor,
                                                     Preprocessor,
                                                     prepare_tensor)


@dataclass
class Session:
    session_id: Union[int, str]
    request_id: str = ''
    prev: str = ''  # history of the session in text format
    round_prev: str = ''  # previous generated text in the current round
    sequence_length: int = 0  # the total generated token number in the session
    response: str = ''
    status: int = None  # status of the session


class StatusCode(Enum):
    TRITON_STREAM_END = 0  # end of streaming
    TRITON_STREAM_ING = 1  # response is in streaming
    TRITON_SERVER_ERR = -1  # triton server's error
    TRITON_SESSION_CLOSED = -2  # session has been closed
    TRITON_SESSION_OUT_OF_LIMIT = -3  # request length out of limit
    TRITON_SESSION_INVALID_ARG = -4  # invalid argument


def stream_callback(que, result, error):
    if error:
        print(error)
        que.put(dict(errcode=StatusCode.TRITON_SERVER_ERR, errmsg=f'{error}'))
    else:
        que.put(result.get_response(as_json=True))


def get_logger(log_file=None, log_level=logging.INFO):
    from .utils import get_logger
    logger = get_logger('service.ft', log_file=log_file, log_level=log_level)
    return logger


class Chatbot:
    """Chatbot for LLaMA series models with fastertransformer as inference
    engine.

    Args:
        tritonserver_addr (str): communicating address '<ip>:<port>' of
            triton inference server
        model_name (str): name of the to-be-deployed mode
        session_len (int): the maximum context length of the model
        top_p (float): If set to float < 1, only the smallest set of most
            probable tokens with probabilities that add up to top_p or higher
            are kept for generation.
        top_k (int): The number of the highest probability vocabulary tokens to
            keep for top-k-filtering
        temperature (float): to modulate the next token probability
        repetition_penalty (float): The parameter for repetition penalty.
            1.0 means no penalty
        stop_words (list): List of token ids that stops the generation
        bad_words (list): List of token ids that are not allowed to be
            generated.
        log_level (int): the level of the log
        display (bool): display the generated text on consolo or not
    """

    def __init__(self,
                 tritonserver_addr: str,
                 model_name: str,
                 session_len: int = 2048,
                 top_p: float = 1.0,
                 top_k: int = 40,
                 temperature: float = 1.0,
                 repetition_penalty: float = 1.0,
                 stop_words: List = None,
                 bad_words: List = None,
                 log_level: int = logging.INFO,
                 display: bool = False):
        self._session = None
        self.tritonserver_addr = tritonserver_addr
        self.model_name = model_name
        if stop_words is not None:
            stop_words = np.array(stop_words, dtype=np.int32)
        if bad_words is not None:
            bad_words = np.array(bad_words, dtype=np.int32)

        self.cfg = mmengine.Config(
            dict(session_len=session_len,
                 top_p=top_p,
                 top_k=top_k,
                 temperature=temperature,
                 repetition_penalty=repetition_penalty,
                 stop_words=stop_words,
                 bad_words=bad_words))
        self.preprocess = Preprocessor(tritonserver_addr)
        self.postprocess = Postprocessor(tritonserver_addr)
        self.log_level = log_level
        self.display = display

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

        prompt = self._get_prompt(prompt, sequence_start)
        for status, res, tokens in self._stream_infer(self._session, prompt,
                                                      request_output_len,
                                                      sequence_start,
                                                      sequence_end):
            yield status, res, tokens
        self._session.prev = self._session.prev + self._session.round_prev

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
            if status != StatusCode.TRITON_STREAM_END:
                return status
        return StatusCode.TRITON_STREAM_END

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
            if prev_session.prev:
                logger.warn(f'TODO: start to recover session {session_id}')
        else:
            logger.info(f'cancel session {session_id} failed: {res}')
        return status

    def _get_prompt(self, prompt: str, sequence_start: bool):
        if self.model_name == 'vicuna':
            if sequence_start:
                return f'USER: {prompt} ASSISTANT:'
            else:
                return f'</s>USER: {prompt} ASSISTANT:'
        else:
            return prompt

    def _stream_infer(self,
                      session: Session,
                      prompt: str,
                      request_output_len: int = 512,
                      sequence_start: bool = True,
                      sequence_end: bool = False,
                      cancel: bool = False):
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

        input_ids, input_lengths = self.preprocess(prompt)
        input_tokens = input_lengths.squeeze()

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
            yield StatusCode.TRITON_SESSION_OUT_OF_LIMIT, errmsg, 0
        logger.info(f'session {session.session_id}, '
                    f'input tokens: {input_tokens}, '
                    f'request tokens: {request_output_len}, '
                    f'history tokens: {session.sequence_length}')

        preseq_length = session.sequence_length
        session.round_prev = ''

        que = queue.Queue()
        producer = threading.Thread(target=self._stream_producer,
                                    args=(self.tritonserver_addr, session, que,
                                          self.cfg, input_ids, input_lengths,
                                          request_output_len, sequence_start,
                                          sequence_end, preseq_length, cancel))
        producer.start()
        for state, res, tokens in self.stream_consumer(self.postprocess, que,
                                                       session, preseq_length,
                                                       cancel, logger,
                                                       self.display):
            if state.value < 0:
                yield state, res, 0
            else:
                yield state, res, tokens - input_tokens
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
        request_output_len = np.full(input_lengths.shape,
                                     request_output_len).astype(np.uint32)

        callback = partial(stream_callback, que)
        with grpcclient.InferenceServerClient(tritonserver_addr) as client:
            inputs = [
                prepare_tensor('input_ids', input_ids),
                prepare_tensor('input_lengths', input_lengths),
                prepare_tensor('request_output_len', request_output_len),
                prepare_tensor('runtime_top_k',
                               cfg.top_k * np.ones((1, 1), dtype=np.uint32)),
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
            client.async_stream_infer('fastertransformer',
                                      inputs,
                                      sequence_id=session.session_id,
                                      request_id=session.request_id,
                                      sequence_start=sequence_start,
                                      sequence_end=sequence_end)
        que.put(None)

    @staticmethod
    def stream_consumer(postprocess, res_queue, session, preseq_length, cancel,
                        logger, display):

        def process_response(res):
            if session.ai_says is None:
                return res, True
            index = res.find(session.ai_says)
            if index == -1:
                return res, False
            res = res[index + len(session.ai_says):].replace(session.eoa, '')
            return res, True

        while True:
            result = res_queue.get()
            if result is None:
                yield StatusCode.TRITON_STREAM_END, session.response, \
                      session.sequence_length - preseq_length
                break
            if 'errcode' in result:
                logger.error(f'got error from fastertransformer, code '
                             f"{result['errcode']}, {result['errmsg']}, "
                             f'token {session.sequence_length}')
                session.sequence_length = preseq_length
                yield result['errcode'], result['errmsg'], 0
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
                sequence_length = sequence_length - preseq_length

                output_ids = output_ids.reshape((1, 1, output_ids.shape[-1]))
                sequence_length = sequence_length.reshape(
                    (1, sequence_length.shape[-1]))
                output_str = postprocess(output_ids[:, :, preseq_length:],
                                         sequence_length)
                text = output_str[0].decode()
                if display:
                    new_text = text[len(session.round_prev):]
                    print(new_text, end='', flush=True)
                session.round_prev = text
                yield (StatusCode.TRITON_STREAM_ING, session.response,
                       sequence_length.squeeze())
            except Exception as e:
                logger.error(f'catch exception: {e}')

        # put session back to queue so that `_stream_infer` can update it in
        # `self.sessions`
        while not res_queue.empty():
            res_queue.get()
        res_queue.put(session)
        if display:
            print('\n')
