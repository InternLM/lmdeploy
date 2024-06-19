# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from lmdeploy.messages import EngineGenerationConfig, EngineOutput
from lmdeploy.utils import get_logger

from ..messages import (InputEmbeddingRangeType, InputEmbeddings,
                        InputEmbeddingType, SamplingParam)
from .engine import Engine
from .request import RequestSender, RequestType, Response, ResponseType

logger = get_logger('lmdeploy')


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
    resp = await req_sender.async_send(RequestType.END_SESSION,
                                       dict(session_id=session_id))
    _check_resp_success(resp, (f'Failed to end session: {session_id}. '
                               f'Error: {resp.type}.'))


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
    resp = req_sender.send(RequestType.END_SESSION,
                           dict(session_id=session_id))
    _check_resp_success(resp, (f'Failed to end session: {session_id}. '
                               f'Error: {resp.type}.'))


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

        def __get_max_input_len(engine):
            """get max input len."""
            cache_config = engine.cache_config
            max_input_len = (cache_config.block_size *
                             cache_config.num_gpu_blocks)
            window_size = cache_config.window_size
            if window_size > 0 and window_size <= max_input_len:
                max_input_len = (1 << 63) - 1
            return max_input_len

        self.engine = engine
        self.req_sender = engine.req_manager.build_sender()

        self.max_input_len = __get_max_input_len(self.engine)

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

    async def async_stream_infer(
            self,
            session_id: int,
            input_ids: List[int],
            gen_config: EngineGenerationConfig = None,
            adapter_name: str = None,
            input_embeddings: InputEmbeddingType = None,
            input_embedding_ranges: InputEmbeddingRangeType = None,
            **kwargs):
        """Send stream inference request.

        Args:
            session_id (int): The session id.
            input_ids (List[int]): The input token ids.
            gen_config (EngineGenerationConfig): The sampling parameters.
            adapter_name (str): The lora adapter name.

        Yields:
            int: Error flags. 0 if success.
            List[int]: The streaming output tokens.
            int: The number of the output tokens.
        """
        if len(input_ids) > self.max_input_len:
            yield EngineOutput(ResponseType.INPUT_LENGTH_ERROR, [], 0)
            return
        gen_config = gen_config or EngineGenerationConfig()
        sampling_param = SamplingParam.from_gen_config(gen_config=gen_config)
        await async_try_add_session(self.req_sender, session_id)
        input_embeddings_new: List[InputEmbeddings] = None
        if input_embeddings is not None and len(input_embeddings) > 0:
            assert len(input_embeddings) == len(input_embedding_ranges)
            input_embeddings_new = [
                InputEmbeddings(emb, rg[0], rg[1])
                for emb, rg in zip(input_embeddings, input_embedding_ranges)
            ]
        msg = dict(token_ids=input_ids,
                   session_id=session_id,
                   sampling_param=sampling_param,
                   adapter_name=adapter_name,
                   input_embeddings=input_embeddings_new)
        req_id = await self.req_sender.async_send_async(
            RequestType.ADD_MESSAGE, msg)

        token_ids = []
        while True:
            resp = await self.req_sender.async_recv(req_id)

            if resp.req_id != req_id:
                continue
            if resp.type == ResponseType.SUCCESS:
                token_ids += resp.data['token_ids']
                yield EngineOutput(resp.type, token_ids, len(token_ids))
            elif resp.type == ResponseType.FINISH:
                token_ids += resp.data['token_ids']
                yield EngineOutput(resp.type, token_ids, len(token_ids))
                break
            else:
                yield EngineOutput(resp.type, [], 0)
                break

    async def async_infer(
            self,
            session_id: int,
            input_ids: List[int] = None,
            gen_config: EngineGenerationConfig = None,
            input_embeddings: InputEmbeddingType = None,
            input_embedding_ranges: InputEmbeddingRangeType = None,
            **kwargs):
        """Send inference request.

        Args:
            session_id (int): The session id.
            input_ids (List[int]): The input token ids.
            gen_config (EngineGenerationConfig): The sampling parameters.

        Returns:
            int: Error flags. 0 if success.
            List[int]: The streaming output tokens.
            int: The number of the output tokens.
        """
        token_ids = []
        async for outputs in self.async_stream_infer(
                session_id,
                input_ids,
                gen_config=gen_config,
                input_embeddings=input_embeddings,
                input_embedding_ranges=input_embedding_ranges,
                **kwargs):
            status, tmp_ids = outputs.status, outputs.token_ids
            if status not in [ResponseType.SUCCESS, ResponseType.FINISH]:
                return EngineOutput(status, token_ids, len(token_ids))
            token_ids = tmp_ids

        return EngineOutput(0, token_ids, len(token_ids))

    def stream_infer(self,
                     session_id: int,
                     input_ids: List[int],
                     gen_config: EngineGenerationConfig = None,
                     adapter_name: str = None,
                     input_embeddings: InputEmbeddingType = None,
                     input_embedding_ranges: InputEmbeddingRangeType = None,
                     **kwargs):
        """Send stream inference request.

        Args:
            session_id (int): The session id.
            input_ids (List[int]): The input token ids.
            gen_config (EngineGenerationConfig): The sampling parameters.
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
            coro_gen = self.async_stream_infer(
                session_id,
                input_ids,
                gen_config,
                adapter_name,
                input_embeddings=input_embeddings,
                input_embedding_ranges=input_embedding_ranges,
                **kwargs)
            while True:
                try:
                    yield self.req_sender.run_until_complete(
                        coro_gen.__anext__())
                except StopAsyncIteration:
                    break

        if not self.req_sender.is_thread_safe():
            yield from __call_async()
            return

        gen_config = gen_config or EngineGenerationConfig()
        sampling_param = SamplingParam.from_gen_config(gen_config=gen_config)
        try_add_session(self.req_sender, session_id)
        input_embeddings_new: List[InputEmbeddings] = None
        if input_embeddings is not None and len(input_embeddings) > 0:
            assert len(input_embeddings) == len(input_embedding_ranges)
            input_embeddings_new = [
                InputEmbeddings(emb, rg[0], rg[1])
                for emb, rg in zip(input_embeddings, input_embedding_ranges)
            ]
        msg = dict(
            token_ids=input_ids,
            session_id=session_id,
            sampling_param=sampling_param,
            adapter_name=adapter_name,
            input_embeddings=input_embeddings_new,
        )
        req_id = self.req_sender.send_async(RequestType.ADD_MESSAGE, msg)

        token_ids = []
        while True:
            resp = self.req_sender.recv(req_id)

            if resp.req_id != req_id:
                continue
            if resp.type == ResponseType.SUCCESS:
                token_ids += resp.data['token_ids']
                yield EngineOutput(resp.type, token_ids, len(token_ids))
            elif resp.type == ResponseType.FINISH:
                token_ids += resp.data['token_ids']
                yield EngineOutput(resp.type, token_ids, len(token_ids))
                break
            else:
                yield EngineOutput(resp.type, [], 0)
                break

    def infer(self,
              session_id: int,
              input_ids: List[int] = None,
              gen_config: EngineGenerationConfig = None,
              input_embeddings: InputEmbeddingType = None,
              input_embedding_ranges: InputEmbeddingRangeType = None,
              **kwargs):
        """Send inference request.

        Args:
            session_id (int): The session id.
            input_ids (List[int]): The input token ids.
            gen_config (EngineGenerationConfig): The sampling parameters.

        Returns:
            int: Error flags. 0 if success.
            List[int]: The streaming output tokens.
            int: The number of the output tokens.
        """
        token_ids = []
        for outputs in self.stream_infer(
                session_id,
                input_ids,
                gen_config=gen_config,
                input_embeddings=input_embeddings,
                input_embedding_ranges=input_embedding_ranges,
                **kwargs):
            status, tmp_ids = outputs.status, outputs.token_ids
            if status not in [ResponseType.SUCCESS, ResponseType.FINISH]:
                return EngineOutput(status, token_ids, len(token_ids))
            token_ids = tmp_ids

        return EngineOutput(0, token_ids, len(token_ids))

    async def async_batched_infer(
        self,
        session_ids: List[int],
        token_ids: List[List[int]] = None,
        gen_config: EngineGenerationConfig = None,
        adapter_names: List[str] = None,
        keep_cache: bool = False,
        input_embeddings: List[InputEmbeddingType] = None,
        input_embedding_ranges: List[InputEmbeddingRangeType] = None,
    ):
        """Send inference request.

        Args:
            session_ids (List[int]): The session id.
            token_ids (List[int]): The input token ids.
            gen_config (EngineGenerationConfig): The sampling parameters.
            adapter_names (List[str]): The name of the adapters.
            keep_cache (bool): Keep kv cache after infer.

        Returns:
            int: Error flags. 0 if success.
            List[int]: The streaming output tokens.
            int: The number of the output tokens.
        """
        batch_size = len(token_ids)
        assert len(session_ids) == batch_size
        if adapter_names is not None:
            assert len(adapter_names) == batch_size
        else:
            adapter_names = [None for _ in range(batch_size)]

        if input_embeddings is not None:
            assert len(input_embeddings) == batch_size
            assert len(input_embedding_ranges) == batch_size
        else:
            input_embeddings = [None] * batch_size
            input_embedding_ranges = [None] * batch_size

        async def _add_sessions(session_ids):
            for session_id in session_ids:
                await self._async_try_add_session(session_id)

        async def _add_messages(session_ids, token_ids, adapter_names,
                                input_embeddings, input_embedding_ranges):
            add_msgs = []
            sampling_param = SamplingParam.from_gen_config(gen_config)
            for session_id, token_id, adapter_name, input_emb, input_ranges in zip(  # noqa: E501
                    session_ids, token_ids, adapter_names, input_embeddings,
                    input_embedding_ranges):
                cur_input_embeddings: List[InputEmbeddings] = None
                if input_emb is not None and len(input_emb) > 0:
                    assert len(input_emb) == len(input_ranges)
                    cur_input_embeddings = [
                        InputEmbeddings(emb, rg[0], rg[1])
                        for emb, rg in zip(input_emb, input_ranges)
                    ]
                msg = dict(
                    token_ids=token_id,
                    session_id=session_id,
                    sampling_param=sampling_param,
                    adapter_name=adapter_name,
                    input_embeddings=cur_input_embeddings,
                )
                add_msgs.append(msg)
            req_types = [RequestType.ADD_MESSAGE] * batch_size
            req_ids = await self.req_sender.async_batched_send_async(
                req_types, data=add_msgs)
            return req_ids

        await _add_sessions(session_ids)
        req_ids = await _add_messages(session_ids, token_ids, adapter_names,
                                      input_embeddings, input_embedding_ranges)

        # receive messages
        req_idx_map = dict(zip(req_ids, range(len(req_ids))))
        output_token_ids = [list() for _ in req_ids]
        status = 0
        finish_count = batch_size
        while finish_count:
            resp = await self.req_sender.async_recv_any()
            if resp.req_id not in req_ids:
                continue
            idx = req_idx_map[resp.req_id]
            token_ids = output_token_ids[idx]
            if resp.type == ResponseType.SUCCESS:
                token_ids += resp.data['token_ids']
            elif resp.type == ResponseType.FINISH:
                token_ids += resp.data['token_ids']
                if not keep_cache:
                    session_id = session_ids[idx]
                    await self.async_end(session_id=session_id)
                finish_count -= 1
            else:
                logger.error(f'Unexpected response: {resp.type}')
                status = 1
                break

        output_token_len = [len(token_ids) for token_ids in output_token_ids]
        return EngineOutput(status, output_token_ids, output_token_len)

    def batched_infer(
        self,
        session_ids: List[int],
        token_ids: List[List[int]] = None,
        gen_config: EngineGenerationConfig = None,
        adapter_names: List[str] = None,
        keep_cache: bool = False,
        input_embeddings: List[InputEmbeddingType] = None,
        input_embedding_ranges: List[InputEmbeddingRangeType] = None,
    ):
        """batched infer."""
        coro = self.async_batched_infer(
            session_ids,
            token_ids,
            gen_config=gen_config,
            adapter_names=adapter_names,
            input_embeddings=input_embeddings,
            input_embedding_ranges=input_embedding_ranges,
            keep_cache=keep_cache)
        return self.req_sender.run_until_complete(coro)

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
               steps: List[int] = None,
               sequence_start: bool = True,
               sequence_end: bool = True,
               adapter_names: List[str] = None):
        """Perform context decode on input tokens.

        Args:
            input_ids (numpy.ndarray): the batch of input token ids
            steps (List[int]): the offset of the k/v cache
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
            adapter_names (List[str]): The name of the adapters.
        """
        from torch.nn.utils.rnn import pad_sequence
        logger.debug('Decoding logits.')
        batch_size = len(input_ids)

        def __add_messages(session_ids, input_ids, adapter_names):
            add_msgs = []
            sampling_param = SamplingParam(max_new_tokens=0)
            for session_id, token_id, adapter_name in zip(
                    session_ids, input_ids, adapter_names):
                if len(token_id) > self.max_input_len:
                    raise RuntimeError(
                        f'Expect input length<={self.max_input_len} '
                        f'but get {len(token_id)}')
                msg = dict(token_ids=token_id,
                           session_id=session_id,
                           sampling_param=sampling_param,
                           adapter_name=adapter_name,
                           return_logits=True)
                add_msgs.append(msg)
            req_types = [RequestType.ADD_MESSAGE] * batch_size
            req_ids = self.req_sender.batched_send_async(req_types,
                                                         data=add_msgs)
            return req_ids

        if steps is not None:
            assert batch_size == len(steps)

        if adapter_names is None:
            adapter_names = [None] * batch_size
        assert batch_size == len(adapter_names)

        session_ids = tuple(range(batch_size))
        if sequence_start:
            for sid in session_ids:
                self.req_sender.send(RequestType.END_SESSION,
                                     dict(session_id=sid))
                self._try_add_session(sid)

        req_ids = __add_messages(session_ids, input_ids, adapter_names)
        req_idx_map = dict(zip(req_ids, range(len(req_ids))))

        finish_count = batch_size
        ret = [None] * batch_size
        while finish_count > 0:
            resp = self.req_sender.recv_any()
            if resp.req_id not in req_ids:
                continue

            assert resp.type == ResponseType.FINISH
            idx = req_idx_map[resp.req_id]
            ret[idx] = resp.data['logits']
            finish_count -= 1

        ret = pad_sequence(ret, True)

        if sequence_end:
            for sid in session_ids:
                self.end(sid)

        return ret
