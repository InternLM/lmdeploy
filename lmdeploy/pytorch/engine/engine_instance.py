# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List

from lmdeploy.messages import EngineOutput, GenerationConfig
from lmdeploy.utils import get_logger

from ..messages import SamplingParam
from .engine import Engine
from .request import RequestSender, RequestType, Response, ResponseType

logger = get_logger('lmdeploy')

InputMultiModalType = List[Dict[str, Any]]


def _check_resp(resp: Response, state: ResponseType, warning_msg: str = None):
    """check if response has state."""
    if isinstance(state, ResponseType):
        state = [state]
    ret = resp.type in state
    if not ret and warning_msg is not None:
        logger.warning(warning_msg)
    return ret


def _check_resp_success(resp: Response, warning_msg: str = None):
    """check if response success."""
    return _check_resp(resp, ResponseType.SUCCESS, warning_msg)


async def async_try_add_session(req_sender: RequestSender, session_id: int):
    """Add new session.

    Args:
        session_id (int): The session id to add.
    """
    resp = await req_sender.async_send(RequestType.ADD_SESSION,
                                       dict(session_id=session_id))
    _check_resp(resp, [ResponseType.SUCCESS, ResponseType.SESSION_REPEAT],
                (f'Can not add session {session_id} '
                 f'with error: {resp.type}'))


async def async_end(req_sender: RequestSender, session_id: int):
    """End the given session."""
    req_sender.send_async(RequestType.END_SESSION,
                          dict(session_id=session_id, response=False))


async def async_cancel(req_sender: RequestSender, session_id: int):
    """Stop current streaming inference."""
    resp = await req_sender.async_send(RequestType.STOP_SESSION,
                                       dict(session_id=session_id))
    _check_resp_success(resp, (f'Failed to cancel session: {session_id}. '
                               f'Error: {resp.type}.'))


def try_add_session(req_sender: RequestSender, session_id: int):
    """Add new session.

    Args:
        session_id (int): The session id to add.
    """
    resp = req_sender.send(RequestType.ADD_SESSION,
                           dict(session_id=session_id))
    _check_resp(resp, [ResponseType.SUCCESS, ResponseType.SESSION_REPEAT],
                (f'Can not add session {session_id} '
                 f'with error: {resp.type}'))


def end(req_sender: RequestSender, session_id: int):
    """End the given session."""
    req_sender.send_async(RequestType.END_SESSION,
                          dict(session_id=session_id, response=False))


def cancel(req_sender: RequestSender, session_id: int):
    """Stop current streaming inference."""
    resp = req_sender.send(RequestType.STOP_SESSION,
                           dict(session_id=session_id))
    _check_resp_success(resp, (f'Failed to cancel session: {session_id}. '
                               f'Error: {resp.type}.'))


class EngineInstance:
    """Instance of TurboMind.

    Args:
        engine (Engine): engine
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self.req_sender = engine.req_manager.build_sender()

        self.max_input_len = self.engine.max_session_len

    def __del__(self):
        """Destructor."""
        self.engine.req_manager.senders.pop(self.req_sender.sender_id)

    async def _async_try_add_session(self, session_id: int):
        """Add new session.

        Args:
            session_id (int): The session id to add.
        """
        return await async_try_add_session(self.req_sender, session_id)

    def _try_add_session(self, session_id: int):
        """Add new session.

        Args:
            session_id (int): The session id to add.
        """
        return try_add_session(self.req_sender, session_id)

    async def async_stream_infer(self,
                                 session_id: int,
                                 input_ids: List[int],
                                 gen_config: GenerationConfig = None,
                                 multimodal: InputMultiModalType = None,
                                 adapter_name: str = None,
                                 **kwargs):
        """Send stream inference request.

        Args:
            session_id (int): The session id.
            input_ids (List[int]): The input token ids.
            gen_config (GenerationConfig): The sampling parameters.
            adapter_name (str): The lora adapter name.

        Yields:
            int: Error flags. 0 if success.
            List[int]: The streaming output tokens.
            int: The number of the output tokens.
        """
        if len(input_ids) > self.max_input_len:
            yield EngineOutput(ResponseType.INPUT_LENGTH_ERROR, [], 0)
            return
        gen_config = gen_config or GenerationConfig()
        sampling_param = SamplingParam.from_gen_config(gen_config=gen_config)
        self.req_sender.send_async(RequestType.ADD_SESSION,
                                   dict(session_id=session_id, response=False))
        msg = dict(
            token_ids=input_ids,
            session_id=session_id,
            sampling_param=sampling_param,
            adapter_name=adapter_name,
            input_multimodals=multimodal,
        )
        resp = self.req_sender.send_async(RequestType.ADD_MESSAGE, msg)

        while True:
            resp = await self.req_sender.async_recv(resp)

            if resp.type == ResponseType.SUCCESS:
                token_ids = resp.data['token_ids'].tolist()
                yield EngineOutput(resp.type, token_ids, len(token_ids))
            elif resp.type == ResponseType.FINISH:
                token_ids = resp.data['token_ids'].tolist()
                yield EngineOutput(resp.type, token_ids, len(token_ids))
                break
            else:
                yield EngineOutput(resp.type, [], 0)
                break

    async def async_infer(self,
                          session_id: int,
                          input_ids: List[int] = None,
                          multimodal: InputMultiModalType = None,
                          gen_config: GenerationConfig = None,
                          **kwargs):
        """Send inference request.

        Args:
            session_id (int): The session id.
            input_ids (List[int]): The input token ids.
            gen_config (GenerationConfig): The sampling parameters.

        Returns:
            int: Error flags. 0 if success.
            List[int]: The streaming output tokens.
            int: The number of the output tokens.
        """
        token_ids = []
        async for outputs in self.async_stream_infer(session_id,
                                                     input_ids,
                                                     multimodal=multimodal,
                                                     gen_config=gen_config,
                                                     **kwargs):
            status, tmp_ids = outputs.status, outputs.token_ids
            if status not in [ResponseType.SUCCESS, ResponseType.FINISH]:
                return EngineOutput(status, token_ids, len(token_ids))
            token_ids = tmp_ids

        return EngineOutput(0, token_ids, len(token_ids))

    def stream_infer(self,
                     session_id: int,
                     input_ids: List[int],
                     multimodal: InputMultiModalType = None,
                     gen_config: GenerationConfig = None,
                     adapter_name: str = None,
                     **kwargs):
        """Send stream inference request.

        Args:
            session_id (int): The session id.
            input_ids (List[int]): The input token ids.
            gen_config (GenerationConfig): The sampling parameters.
            adapter_name (str): The lora adapter name.

        Yields:
            int: Error flags. 0 if success.
            List[int]: The streaming output tokens.
            int: The number of the output tokens.
        """
        if len(input_ids) > self.max_input_len:
            yield EngineOutput(ResponseType.INPUT_LENGTH_ERROR, [], 0)
            return

        def __call_async():
            """call async."""
            coro_gen = self.async_stream_infer(session_id,
                                               input_ids,
                                               multimodal=multimodal,
                                               gen_config=gen_config,
                                               adapter_name=adapter_name,
                                               **kwargs)
            while True:
                try:
                    yield self.req_sender.run_until_complete(
                        coro_gen.__anext__())
                except StopAsyncIteration:
                    break

        yield from __call_async()

    def infer(self,
              session_id: int,
              input_ids: List[int] = None,
              multimodal: InputMultiModalType = None,
              gen_config: GenerationConfig = None,
              **kwargs):
        """Send inference request.

        Args:
            session_id (int): The session id.
            input_ids (List[int]): The input token ids.
            gen_config (GenerationConfig): The sampling parameters.

        Returns:
            int: Error flags. 0 if success.
            List[int]: The streaming output tokens.
            int: The number of the output tokens.
        """
        token_ids = []
        for outputs in self.stream_infer(session_id,
                                         input_ids,
                                         multimodal=multimodal,
                                         gen_config=gen_config,
                                         **kwargs):
            status, tmp_ids = outputs.status, outputs.token_ids
            if status not in [ResponseType.SUCCESS, ResponseType.FINISH]:
                return EngineOutput(status, token_ids, len(token_ids))
            token_ids = tmp_ids

        return EngineOutput(0, token_ids, len(token_ids))

    async def async_end(self, session_id: int):
        """End the given session."""
        return await async_end(self.req_sender, session_id)

    def end(self, session_id: int):
        """End the given session."""
        return end(self.req_sender, session_id)

    async def async_cancel(self, session_id: int):
        """Stop current streaming inference."""
        return await async_cancel(self.req_sender, session_id)

    def cancel(self, session_id: int):
        """Stop current streaming inference."""
        return cancel(self.req_sender, session_id)

    def decode(self,
               input_ids,
               multimodal: List[InputMultiModalType] = None,
               steps: List[int] = None,
               sequence_start: bool = True,
               sequence_end: bool = True,
               adapter_names: List[str] = None):
        """Perform context decode on input tokens.

        Args:
            input_ids (numpy.ndarray): the batch of input token ids
            steps (List[int]): the offset of the k/v cache
            multimodal (List[InputMultiModalType]):
                multimodals inputs.
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
            adapter_names (List[str]): The name of the adapters.
        """
        from torch.nn.utils.rnn import pad_sequence
        logger.debug('Decoding logits.')
        batch_size = len(input_ids)

        def __add_messages(session_ids, input_ids, adapter_names,
                           input_multimodals):
            add_msgs = []
            sampling_param = SamplingParam(max_new_tokens=0)
            batch_size = len(input_ids)
            if input_multimodals is None:
                input_multimodals = [None] * batch_size
            for (session_id, token_id, adapter_name,
                 in_mm) in zip(session_ids, input_ids, adapter_names,
                               input_multimodals):
                if len(token_id) > self.max_input_len:
                    raise RuntimeError(
                        f'Expect input length<={self.max_input_len} '
                        f'but get {len(token_id)}')
                msg = dict(token_ids=token_id,
                           session_id=session_id,
                           sampling_param=sampling_param,
                           adapter_name=adapter_name,
                           input_multimodals=in_mm,
                           return_logits=True)
                add_msgs.append(msg)
            req_types = [RequestType.ADD_MESSAGE] * batch_size
            resps = self.req_sender.batched_send_async(req_types,
                                                       data=add_msgs)
            return resps

        if steps is not None:
            assert batch_size == len(steps)

        if adapter_names is not None:
            assert len(adapter_names) == batch_size
        else:
            adapter_names = [None] * batch_size

        session_ids = tuple(range(batch_size))
        if sequence_start:
            for sid in session_ids:
                self.req_sender.send(RequestType.END_SESSION,
                                     dict(session_id=sid))
                self._try_add_session(sid)

        resps = __add_messages(session_ids, input_ids, adapter_names,
                               multimodal)

        ret = []
        for resp in resps:
            resp = self.req_sender.recv(resp)
            assert resp.type == ResponseType.FINISH
            ret.append(resp.data['logits'])

        ret = pad_sequence(ret, True)

        if sequence_end:
            for sid in session_ids:
                self.end(sid)

        return ret
