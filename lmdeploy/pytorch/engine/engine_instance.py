# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List

from lmdeploy.messages import EngineOutput, GenerationConfig
from lmdeploy.utils import get_logger

from ..messages import SamplingParam
from .base import EngineInstanceBase
from .engine import Engine
from .request import RequestSender, RequestType, Response, ResponseType

logger = get_logger('lmdeploy')

InputMultiModalType = List[Dict[str, Any]]


def _check_resp(resp: Response, state: ResponseType, warning_msg: str = None):
    """Check if response has state."""
    if isinstance(state, ResponseType):
        state = [state]
    ret = resp.type in state
    if not ret and warning_msg is not None:
        logger.warning(warning_msg)
    return ret


def _check_resp_success(resp: Response, warning_msg: str = None):
    """Check if response success."""
    return _check_resp(resp, ResponseType.SUCCESS, warning_msg)


async def async_try_add_session(req_sender: RequestSender, session_id: int):
    """Add new session.

    Args:
        session_id (int): The session id to add.
    """
    resp = await req_sender.async_send(RequestType.ADD_SESSION, dict(session_id=session_id))
    _check_resp(resp, [ResponseType.SUCCESS, ResponseType.SESSION_REPEAT], (f'Can not add session {session_id} '
                                                                            f'with error: {resp.type}'))


async def async_cancel(req_sender: RequestSender, session_id: int):
    """Stop current streaming inference."""
    resp = await req_sender.async_send(RequestType.STOP_SESSION, dict(session_id=session_id))
    _check_resp_success(resp, (f'Failed to cancel session: {session_id}. '
                               f'Error: {resp.type}.'))


def try_add_session(req_sender: RequestSender, session_id: int):
    """Add new session.

    Args:
        session_id (int): The session id to add.
    """
    resp = req_sender.send(RequestType.ADD_SESSION, dict(session_id=session_id))
    _check_resp(resp, [ResponseType.SUCCESS, ResponseType.SESSION_REPEAT], (f'Can not add session {session_id} '
                                                                            f'with error: {resp.type}'))


def end(req_sender: RequestSender, session_id: int):
    """End the given session."""
    logger.debug(f'session[{session_id}] try end session.')
    req_sender.send_async(RequestType.END_SESSION, dict(session_id=session_id, response=False))


def cancel(req_sender: RequestSender, session_id: int):
    """Stop current streaming inference."""
    logger.debug(f'session[{session_id}] try end session.')
    resp = req_sender.send(RequestType.STOP_SESSION, dict(session_id=session_id))
    _check_resp_success(resp, (f'Failed to cancel session: {session_id}. '
                               f'Error: {resp.type}.'))


class EngineInstance(EngineInstanceBase):
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
            yield EngineOutput(ResponseType.INPUT_LENGTH_ERROR, [])
            return
        gen_config = gen_config or GenerationConfig()
        sampling_param = SamplingParam.from_gen_config(gen_config=gen_config)
        logger.debug(f'session[{session_id}] try add session.')
        self.req_sender.send_async(RequestType.ADD_SESSION, dict(session_id=session_id, response=False))
        msg = dict(
            token_ids=input_ids,
            session_id=session_id,
            sampling_param=sampling_param,
            adapter_name=adapter_name,
            input_multimodals=multimodal,
            migration_request=gen_config.migration_request,
            with_cache=gen_config.with_cache,
            preserve_cache=gen_config.preserve_cache,
        )
        logger.debug(f'session[{session_id}] add message: num_input_ids={len(input_ids)}.')
        resp = self.req_sender.send_async(RequestType.ADD_MESSAGE, msg)
        output_offset = 0

        while True:
            resp = await self.req_sender.async_recv(resp)

            cache_block_ids = resp.data.get('cache_block_ids', None) if resp.data else None
            req_metrics = resp.data.get('req_metrics', None) if resp.data else None
            logprobs = resp.data.pop('logprobs', None) if resp.data else None
            if resp.type == ResponseType.SUCCESS:
                token_ids = resp.data['token_ids'].tolist()
                num_ids = len(token_ids) - output_offset
                logger.debug(f'session[{session_id}] success: num_out_ids={num_ids}.')
                yield EngineOutput(resp.type,
                                   token_ids[output_offset:],
                                   cache_block_ids=cache_block_ids,
                                   req_metrics=req_metrics,
                                   logprobs=logprobs)
                output_offset = len(token_ids)
            elif resp.type == ResponseType.FINISH:
                resp_data = resp.data
                token_ids = resp_data['token_ids'].tolist()
                logits = resp_data['logits']
                num_ids = len(token_ids) - output_offset
                logger.debug(f'session[{session_id}] finish: num_out_ids={num_ids}.')
                yield EngineOutput(resp.type,
                                   token_ids[output_offset:],
                                   logits=logits,
                                   cache_block_ids=cache_block_ids,
                                   req_metrics=req_metrics,
                                   logprobs=logprobs)
                break
            else:
                logger.debug(f'session[{session_id}] failed.')
                yield EngineOutput(resp.type, [])
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
        async for outputs in self.async_stream_infer(session_id,
                                                     input_ids,
                                                     multimodal=multimodal,
                                                     gen_config=gen_config,
                                                     **kwargs):
            status = outputs.status
            if status not in [ResponseType.SUCCESS, ResponseType.FINISH]:
                return outputs

        return outputs

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

        def __call_async():
            """Call async."""
            coro_gen = self.async_stream_infer(session_id,
                                               input_ids,
                                               multimodal=multimodal,
                                               gen_config=gen_config,
                                               adapter_name=adapter_name,
                                               **kwargs)
            while True:
                try:
                    yield self.req_sender.run_until_complete(coro_gen.__anext__())
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
        return self.req_sender.run_until_complete(
            self.async_infer(session_id, input_ids, multimodal=multimodal, gen_config=gen_config, **kwargs))

    async def async_end(self, session_id: int):
        """End the given session."""
        return end(self.req_sender, session_id)

    def end(self, session_id: int):
        """End the given session."""
        return end(self.req_sender, session_id)

    async def async_cancel(self, session_id: int):
        """Stop current streaming inference."""
        return await async_cancel(self.req_sender, session_id)

    def cancel(self, session_id: int):
        """Stop current streaming inference."""
        return cancel(self.req_sender, session_id)
