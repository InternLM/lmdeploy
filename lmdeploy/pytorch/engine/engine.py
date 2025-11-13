# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import copy
import gc
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from lmdeploy.messages import PytorchEngineConfig, RequestMetrics, ResponseType
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.conn.engine_conn import EngineP2PConnection
from lmdeploy.pytorch.disagg.conn.protocol import (DistServeConnectionRequest, DistServeDropConnectionRequest,
                                                   DistServeInitRequest)
from lmdeploy.pytorch.disagg.messages import MigrationExecutionBatch
from lmdeploy.utils import get_logger, get_max_batch_size, get_model, logging_timer

from ..adapter.adapter import AdapterManager
from ..config import BackendConfig, CacheConfig, DistConfig, MiscConfig, ModelConfig, SchedulerConfig
from ..messages import MessageStatus, SchedulerSequence, UpdateTokenMode
from ..model_inputs import ModelInputs, VisionModelInputs
from ..paging import Scheduler
from ..strategies import build_strategy_factory
from .base import EngineBase
from .engine_checker import EngineChecker
from .executor import build_executor
from .model_agent import BatchedOutputs
from .request import Request, RequestManager, RequestType, Response

logger = get_logger('lmdeploy')

SeqList = List[SchedulerSequence]

_EMPTY_TOKEN = np.empty((0, ), dtype=np.int64)


@dataclass
class InferOutput:
    """The output of the model inference."""

    session_id: int
    resp: Response
    token_ids: List[int]
    meta: Any = None
    finish: bool = False
    logits: torch.Tensor = None
    logprobs: torch.Tensor = None

    # send cache blocks back for migration in Disaggregated LLM Serving
    # when Prefill Engine is Done.
    cache_block_ids: List[int] = None

    # for logging
    req_metrics: RequestMetrics = None


def _tensorlize_block_offsets(block_offsets, dtype=torch.int32):
    """Tensorlize block_offsets."""
    # copy on numpy is faster than torch.nn.utils.rnn.pad_sequence
    batch_size = len(block_offsets)
    max_len = max([len(off) for off in block_offsets])
    out = np.zeros((batch_size, max_len), dtype=block_offsets[0].dtype)

    for idx, off in enumerate(block_offsets):
        off_len = len(off)
        out[idx, :off_len] = off
    return torch.as_tensor(out, dtype=dtype)


def _update_engine_config(engine_config: PytorchEngineConfig):
    """Update pytorch engine config."""
    # make sure engine exits
    if engine_config is None:
        engine_config = PytorchEngineConfig()
    else:
        engine_config = copy.deepcopy(engine_config)

    if engine_config.max_batch_size is None:
        engine_config.max_batch_size = get_max_batch_size(engine_config.device_type)

    if engine_config.dllm_block_length is not None:
        max_prefill_token_num = engine_config.max_prefill_token_num
        max_batch_size = engine_config.max_batch_size
        if max_batch_size * engine_config.dllm_block_length > max_prefill_token_num:
            engine_config.max_batch_size = max_prefill_token_num // engine_config.dllm_block_length
            logger.warning(f'Update max_batch_size to {engine_config.max_batch_size} '
                           f'since dllm_block_length({engine_config.dllm_block_length}) * max_batch_size '
                           f'({max_batch_size}) > max_prefill_token_num ({max_prefill_token_num}).')

    if engine_config.dp != 1:
        if engine_config.tp == 1 and engine_config.ep == 1:
            engine_config.dp = 1
            engine_config.dp_rank = 0

    return engine_config


def _build_scheduler_config(engine_config: PytorchEngineConfig):
    """Build scheduler config."""
    scheduler_config = SchedulerConfig(max_batches=engine_config.max_batch_size,
                                       max_session_len=engine_config.session_len,
                                       prefill_interval=engine_config.prefill_interval)
    return scheduler_config


def _build_cache_config(engine_config: PytorchEngineConfig):
    """Build cache config."""
    cache_config = CacheConfig(max_batches=engine_config.max_batch_size,
                               block_size=engine_config.block_size,
                               num_cpu_blocks=engine_config.num_cpu_blocks,
                               num_gpu_blocks=engine_config.num_gpu_blocks,
                               cache_max_entry_count=engine_config.cache_max_entry_count,
                               max_prefill_token_num=engine_config.max_prefill_token_num,
                               enable_prefix_caching=engine_config.enable_prefix_caching,
                               quant_policy=engine_config.quant_policy,
                               device_type=engine_config.device_type,
                               migration_backend=engine_config.migration_backend,
                               role=engine_config.role)
    return cache_config


def _build_backend_config(engine_config: PytorchEngineConfig):
    """Build backend config."""
    backend_config = BackendConfig(
        eager_mode=engine_config.eager_mode,
        device_type=engine_config.device_type,
    )
    return backend_config


def _build_dist_config(engine_config: PytorchEngineConfig):
    """Build dist config."""
    dist_config = DistConfig(dp=engine_config.dp,
                             tp=engine_config.tp,
                             ep=engine_config.ep,
                             dp_rank=engine_config.dp_rank,
                             enable_microbatch=engine_config.enable_microbatch,
                             enable_eplb=engine_config.enable_eplb)
    return dist_config


def _build_misc_config(engine_config: PytorchEngineConfig):
    """Build misc config."""
    misc_config = MiscConfig.from_engine_config(engine_config)
    return misc_config


def _build_seq_meta(cache_config: CacheConfig, strategy: Any):
    from lmdeploy.pytorch.messages import SequenceMeta

    seq_meta = SequenceMeta(cache_config.block_size, strategy=strategy)
    return seq_meta


class CounterEvent:

    def __init__(self):
        self._counter = 0
        self._event = asyncio.Event()

    async def wait(self):
        await self._event.wait()

    def is_set(self):
        return self._event.is_set()

    def set(self):
        if self._counter > 0:
            self._counter -= 1
        if self._counter == 0:
            self._event.set()

    def clear(self):
        if self._counter == 0 and self._event.is_set():
            self._event.clear()
        self._counter += 1


class RunableEventBase:
    """Runable event base."""

    async def wait(self, idx: int):
        """Wait event."""
        raise NotImplementedError('Not implemented.')

    def set(self, idx: int = None):
        """Set event."""
        raise NotImplementedError('Not implemented.')


class RunableEventAsnyc(RunableEventBase):
    """Awaitable async runable event."""

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.event = asyncio.Event()

    async def wait(self):
        """Wait event."""
        await self.event.wait()

    def set(self):
        """Set event."""
        if self.scheduler.has_unfinished():
            self.event.set()
        else:
            self.event.clear()


def build_runable_event(scheduler: Scheduler):
    """Build runable event."""
    return RunableEventAsnyc(scheduler)


class InputsMakerBase:

    def __init__(self, engine: 'Engine'):
        self.engine = engine
        self.scheduler_config = engine.scheduler_config
        self.executor = engine.executor

    def _make_forward_inputs(self, *args, **kwargs):
        """Make forward inputs."""
        return self.engine._make_forward_inputs(*args, **kwargs)

    async def send_next_inputs(self):
        """Send next input."""
        raise NotImplementedError('Not implemented.')

    async def prefetch_next_inputs(self):
        """prefetch."""
        raise NotImplementedError('Not implemented.')


class InputsMakerAsync(InputsMakerBase):

    def __init__(self, engine: 'Engine'):
        super().__init__(engine)
        self.scheduler = self.engine.scheduler
        self.forward_inputs = None

        self.dp = self.engine.dist_config.dp
        self.role = self.engine.cache_config.role

        self.next_is_prefill = True
        if self.dp == 1:
            self.do_prefill = self.do_prefill_default
        else:
            self.do_prefill = self.do_prefill_dp

    def do_prefill_dp(self):
        if self.role == EngineRole.Prefill:
            return True

        scheduler = self.scheduler

        if self.next_is_prefill:
            ret = scheduler.has_waiting()
        else:
            ret = not scheduler.has_running()
        return ret

    def do_prefill_default(self):
        # decoding if no waiting
        scheduler = self.scheduler
        if not scheduler.has_waiting():
            return False
        num_running = scheduler.num_running()
        num_waiting = scheduler.num_waiting()
        max_batches = self.scheduler_config.max_batches
        # prefill if too much waiting
        permitted_waiting = 4 if (self.engine.engine_config.role != EngineRole.Prefill) else 1
        if num_waiting >= permitted_waiting:
            return True
        # prefill if no enough running
        if num_running < max_batches * 0.5:
            return True
        # decoding
        return False

    async def _send_next_inputs_impl(self, prefill: bool = None, enable_empty: bool = False):
        forward_inputs = self._make_forward_inputs(prefill, enable_empty)
        if forward_inputs is None:
            return None, None
        next_running = forward_inputs.pop('running')
        inputs = forward_inputs['inputs']
        logger.debug(f'Sending forward inputs: {inputs.log_info()}')
        if logger.level <= logging.DEBUG:
            session_ids = [seq.session_id for seq in next_running]
            logger.debug(f'Forward session_ids: {session_ids}')
        self.next_is_prefill = inputs.is_decoding
        await self.executor.forward_async(forward_inputs)
        self.forward_inputs = forward_inputs
        return forward_inputs, next_running

    async def send_next_inputs(self):
        prefill = self.do_prefill()
        return await self._send_next_inputs_impl(prefill)

    async def prefetch_next_inputs(self):
        enable = False
        scheduler = self.scheduler
        prefill = self.do_prefill()
        if prefill:
            enable = True
        else:
            num_running = scheduler.num_running()
            is_decoding = self.forward_inputs['inputs'].is_decoding
            running_threshold = (self.scheduler_config.max_batches // 4) if is_decoding else 0

            if num_running > running_threshold:
                enable = True

        if enable:
            # send next forward
            logger.debug('Prefetching next forward inputs.')
            return await self._send_next_inputs_impl(prefill, True)
        else:
            return None, None


def build_inputs_maker(engine: 'Engine'):
    """Build inputs makers."""
    return InputsMakerAsync(engine)


class Engine(EngineBase):
    """The inference engine of lmdeploy pytorch.

    Args:
        model_path (str): The hugging face model path.
        engine_config (PytorchEngineConfig): The config of the Engine.
        trust_remote_code (bool): Trust remote code.
    """

    def __init__(self,
                 model_path: str,
                 engine_config: PytorchEngineConfig = None,
                 trust_remote_code: bool = True) -> None:
        # make sure engine config exist
        engine_config = _update_engine_config(engine_config)

        # frequently gc would cause latency spike
        # default threshold (700, 10, 10)
        # WARNING: I don't know if it is a good idea to put gc setting here.
        gc.set_threshold(10000, 100, 100)

        # dist args
        self.tp = engine_config.tp
        self.dp = engine_config.dp
        self.dp_rank = engine_config.dp_rank

        # download models and adapters
        if not os.path.exists(model_path):
            model_path = get_model(model_path, engine_config.download_dir, engine_config.revision)

        adapters = engine_config.adapters
        if adapters is not None and len(adapters) > 0:
            adapters = self._download_adapters(adapters, engine_config)

        # check environment
        checker = EngineChecker(model_path=model_path,
                                engine_config=engine_config,
                                trust_remote_code=trust_remote_code,
                                logger=logger)
        checker.handle()

        # build configs
        scheduler_config = _build_scheduler_config(engine_config)
        cache_config = _build_cache_config(engine_config)
        backend_config = _build_backend_config(engine_config)
        dist_config = _build_dist_config(engine_config)
        misc_config = _build_misc_config(engine_config)

        # build model agent
        self.executor = build_executor(model_path,
                                       cache_config=cache_config,
                                       backend_config=backend_config,
                                       dist_config=dist_config,
                                       misc_config=misc_config,
                                       adapters=adapters,
                                       device_type=engine_config.device_type,
                                       distributed_executor_backend=engine_config.distributed_executor_backend,
                                       dtype=engine_config.dtype)
        self.executor.init()

        # strategies
        self.strategy_factory = build_strategy_factory(self.model_config, self.executor.misc_config)
        self.sampling_strategy = self.strategy_factory.build_sampling_strategy()
        self.model_agent_strategy = self.strategy_factory.build_model_agent_strategy()
        self.engine_strategy = self.strategy_factory.build_engine_strategy(cache_config=cache_config,
                                                                           scheduler_config=scheduler_config)
        self.seq_strategy = self.strategy_factory.build_sequence_strategy()

        self.input_processor = self.executor.get_input_processor()
        cache_config = self.executor.cache_config
        self.adapter_manager = self._build_adapter_manager(adapters)
        self.seq_meta = _build_seq_meta(cache_config, strategy=self.seq_strategy)
        self.scheduler = Scheduler(scheduler_config, cache_config, seq_meta=self.seq_meta)

        # engine args
        self.model_path = model_path
        self.engine_config = engine_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.backend_config = backend_config
        self.dist_config = dist_config
        self.misc_config = self.executor.misc_config
        self.max_session_len = self._get_max_session_len()
        self.engine_config.num_cpu_blocks = self.cache_config.num_cpu_blocks
        self.engine_config.num_gpu_blocks = self.cache_config.num_gpu_blocks

        self.req_manager = self._bind_request_manager()

        # create main thread
        self._start_loop()
        self._loop_main = None

        # for PD Disaggregation
        # For migrating prefill request to decode engine
        self.migration_event: asyncio.Event = None
        # For backpressure prefill request when cache is full
        self.perfill_watermark_event: asyncio.Event = None

        self.engine_conn = EngineP2PConnection(self)

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        engine_config: PytorchEngineConfig = None,
                        trust_remote_code: bool = True,
                        **kwargs):
        """Lmdeploy python inference engine.

        Args:
            pretrained_model_name_or_path (str):
                It could be one of the following options:
                    - i) The model_id of a lmdeploy-quantized model hosted
                      inside a model repo on huggingface.co, such as
                      "InternLM/internlm-chat-20b-4bit",
                      "lmdeploy/llama2-chat-70b-4bit", etc.
                    - ii) The model_id of a model hosted inside a model repo
                      on huggingface.co, such as "InternLM/internlm-chat-7b",
                      "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                      and so on.
            engine_config (PytorchEngineConfig): Pytorch engine config.
            trust_remote_code (bool): Trust remote code
        """
        if engine_config is not None and engine_config.enable_mp_engine:
            from .mp_engine import build_mp_engine
            backend = engine_config.mp_engine_backend
            return build_mp_engine(backend=backend,
                                   model_path=pretrained_model_name_or_path,
                                   engine_config=engine_config,
                                   trust_remote_code=trust_remote_code)
        if len(kwargs) > 0:
            logger.debug(f'Get unexpected kwargs: {kwargs}')
        return cls(model_path=pretrained_model_name_or_path,
                   engine_config=engine_config,
                   trust_remote_code=trust_remote_code)

    def _download_adapters(self, adapters: Dict[str, str], engine_config: PytorchEngineConfig):
        """Download adapters."""
        download_dir = engine_config.download_dir
        revision = engine_config.revision
        new_adapters = dict()
        for name, path in adapters.items():
            if os.path.exists(path):
                new_adapters[name] = path
                continue
            new_path = get_model(path, download_dir=download_dir, revision=revision)
            new_adapters[name] = new_path

        return new_adapters

    def _build_adapter_manager(self, adapters):
        return AdapterManager(adapters)

    def _bind_request_manager(self):
        """Bind request manager."""
        req_manager = RequestManager()
        req_manager.bind_func(RequestType.ADD_SESSION, self._on_add_session)
        req_manager.bind_func(RequestType.STOP_SESSION, self._on_stop_session)
        req_manager.bind_func(RequestType.END_SESSION, self._on_end_session)
        req_manager.bind_func(RequestType.ADD_MESSAGE, self._on_add_message)
        return req_manager

    def _start_loop(self):
        """Start loop."""
        return self.req_manager.start_loop(self.async_loop)

    def _response(self, resp: Response, resp_type: ResponseType, data: Any = None, err_msg: str = ''):
        """response."""
        if resp.type == ResponseType.FINISH:
            return
        resp.type = resp_type
        resp.data = data
        resp.err_msg = err_msg
        self.req_manager.response(resp)

    def _get_max_session_len(self):
        """Get max session len."""
        session_len = self.scheduler_config.max_session_len
        max_tokens = (self.cache_config.num_gpu_blocks * self.cache_config.block_size)
        window_size = self.cache_config.window_size
        if window_size > 0 and window_size <= max_tokens:
            max_tokens = (1 << 63) - 1
        max_tokens -= self.cache_config.block_size
        if session_len is None:
            session_len = max_tokens
        else:
            session_len = min(max_tokens, session_len)
        return session_len

    def _on_add_session(self, reqs: List[Request], **kwargs):
        """On add session callback."""
        for req in reqs:
            session_id = req.data['session_id']
            resp = req.data.get('response', True)
            resp_type = ResponseType.SESSION_REPEAT
            if session_id not in self.scheduler.sessions:
                self.scheduler.add_session(session_id)
                resp_type = ResponseType.SUCCESS
            if resp:
                self._response(req.resp, resp_type)

    def _on_stop_session(self, reqs: List[Request], **kwargs):
        """On stop session callback."""
        for req in reqs:
            session_id = req.data['session_id']
            resp = req.data.get('response', True)
            resp_type = ResponseType.SESSION_NOT_EXIST
            if session_id in self.scheduler.sessions:
                self.scheduler.stop_session(session_id)
                session = self.scheduler.sessions[session_id]
                for seq in session.sequences.values():
                    _resp: Response = getattr(seq, 'resp', None)
                    if _resp is not None:
                        _resp.type = ResponseType.CANCEL
                        self.req_manager.response(_resp)
                resp_type = ResponseType.SUCCESS
            if resp:
                self._response(req.resp, resp_type)

    def _on_end_session(self, reqs: List[Request], **kwargs):
        """On end session callback."""
        for req in reqs:
            session_id = req.data['session_id']
            resp = req.data.get('response', True)
            resp_type = ResponseType.SESSION_NOT_EXIST
            if session_id in self.scheduler.sessions:
                msgs = list(self.scheduler.sessions[session_id].sequences.values())
                if len(msgs) > 0 and msgs[0].preserve_cache:
                    self.scheduler._set_message_status(msgs[0], MessageStatus.TO_BE_MIGRATED)
                else:
                    self.end_session(session_id)
                resp_type = ResponseType.SUCCESS
            if resp:
                self._response(req.resp, resp_type)

    def _on_add_message(self, reqs: List[Request], **kwargs):
        """On add message callback."""
        valid_reqs = []
        for req in reqs:
            req_data = req.data
            session_id = req_data['session_id']
            if self.scheduler and session_id not in self.scheduler.sessions:
                self._response(req.resp, ResponseType.SESSION_NOT_EXIST)
                continue
            valid_reqs.append(req)
            if req_data.get('input_multimodals', None) is None:
                continue
            elif self.input_processor is None:
                logger.warning('Do not support Multimodal inputs.')
                continue
            input_ids = req_data['token_ids']
            input_multimodals = req_data['input_multimodals']
            if len(input_multimodals) == 0:
                req_data['input_multimodals'] = None
                continue

            if self.engine_config.disable_vision_encoder:
                # ignore multimodal inputs
                req_data['input_multimodals'] = None
                logger.warning('Vision encoder has not been loaded, multimodal inputs will be ignored.')
                continue

            result = self.input_processor.preprocess_input(input_ids, input_multimodals)

            input_ids = result.input_ids
            input_multimodals = result.input_multimodals

            req_data['token_ids'] = input_ids
            req_data['input_multimodals'] = input_multimodals

        if len(valid_reqs) > 0:
            self._add_message(valid_reqs)

    def _add_message(self, reqs: List[Request]):

        def __update_max_new_tokens(msg):
            """Update max new tokens."""
            max_session_len = self.max_session_len
            sampling_param = msg.sampling_param
            max_new_tokens = sampling_param.max_new_tokens
            num_all_tokens = msg.num_valid_ids
            if max_new_tokens + num_all_tokens > max_session_len:
                logger.warning(
                    f'session[{msg.session_id}]: num tokens is larger than max session len {max_session_len}. '
                    f'Update max_new_tokens={max_session_len - num_all_tokens}.')
                sampling_param.max_new_tokens = max_session_len - num_all_tokens

        scheduler = self.scheduler
        for req in reqs:
            session_id = req.data['session_id']
            if scheduler is None:
                self._response(req.resp, ResponseType.SESSION_NOT_EXIST)
                continue
            session_id = req.data['session_id']
            sess = scheduler.sessions[session_id]
            # TODO: support 1 session n sequence
            sampling_param = req.data['sampling_param']
            if len(sess.sequences) == 0:
                migration_request = req.data.get('migration_request')
                assert len(req.data['token_ids']) > 0, ('Empty input is not allowed.')
                sess.add_sequence(req.data['token_ids'],
                                  sampling_param=sampling_param,
                                  adapter_name=req.data['adapter_name'],
                                  multimodals=req.data.get('input_multimodals'),
                                  input_embeddings=req.data.get('input_embeddings', ),
                                  migration_request=migration_request,
                                  resp_cache=req.data.get('with_cache'),
                                  preserve_cache=req.data.get('preserve_cache'))
                msg = next(iter(sess.sequences.values()))
                __update_max_new_tokens(msg)
                scheduler.add_sequence(msg)
                if migration_request:
                    self.scheduler._set_message_status(msg, MessageStatus.WAITING_MIGRATION)
                    self.migration_event.set()
            else:
                msg = next(iter(sess.sequences.values()))
                msg.update_token_ids(
                    req.data['token_ids'],
                    multimodals=req.data.get('input_multimodals'),
                    embeddings=req.data.get('input_embeddings'),
                    mode=UpdateTokenMode.INPUTS,
                )
                msg.sampling_param = sampling_param
                msg.status = MessageStatus.WAITING
                __update_max_new_tokens(msg)

            msg.resp = req.resp

    @property
    def model_config(self) -> ModelConfig:
        """Model config."""
        return self.executor.model_config

    @property
    def gpu_count(self):
        return self.tp * self.dp

    @property
    def torch_int_dtype(self):
        """Return int32 for cuda, int64 for others."""
        if self.executor.device_type == 'cuda':
            return torch.int32
        return torch.int64

    def _create_vision_model_inputs(self, messages: SeqList, model_inputs: ModelInputs):
        """Create vision model inputs."""
        batch_size = len(messages)

        def __get_vlm_embeddings():
            """Get vlm input embeddings and indexings."""
            max_q_seq_length = model_inputs.seq_length.max().item()
            input_embeddings = [[
                emb.embeddings if isinstance(emb.embeddings, torch.Tensor) else torch.as_tensor(emb.embeddings)
                for emb in msg.input_embeddings
            ] for msg in messages]
            input_embedding_ranges = [
                torch.tensor([[emb.start, emb.end] for emb in msg.input_embeddings]) for msg in messages
            ]
            input_embedding_indexing = torch.zeros((batch_size, max_q_seq_length), dtype=torch.bool)
            for msg_id, msg in enumerate(messages):
                num_history_ids = msg.num_history_ids
                for emb in msg.input_embeddings:
                    # make slice index relative to embeddings
                    emb_start = emb.start - num_history_ids
                    emb_end = emb.end - num_history_ids
                    input_embedding_indexing[msg_id][emb_start:emb_end] = True
            return (input_embeddings, input_embedding_indexing, input_embedding_ranges)

        def __has_values(input_multimodals):
            for input_mm in input_multimodals:
                for val in input_mm.values():
                    if len(val) > 0:
                        return True
            return False

        has_embedding = any([len(msg.history_embeddings) > 0 for msg in messages])
        if has_embedding:
            has_embedding = any([len(msg.input_embeddings) > 0 for msg in messages])

        has_multimodal = any([not msg.history_multimodals.empty() for msg in messages])
        input_multimodals = None
        if has_multimodal:
            input_multimodals = [msg.get_input_multimodals() for msg in messages]
            has_multimodal = __has_values(input_multimodals)
            if not has_multimodal:
                # no multimodal inputs
                input_multimodals = None

        if not has_embedding and not has_multimodal:
            # no vision inputs
            return None

        if has_embedding:
            # for inputs with embeddings
            (input_embeddings, input_embedding_indexing, input_embedding_ranges) = __get_vlm_embeddings()
        else:
            input_embeddings = None
            input_embedding_indexing = None
            input_embedding_ranges = None

        history_lengths = model_inputs.history_lengths
        vision_embedding_inputs = VisionModelInputs(history_lengths=history_lengths,
                                                    input_embeddings=input_embeddings,
                                                    input_embedding_indexing=input_embedding_indexing,
                                                    input_embedding_ranges=input_embedding_ranges,
                                                    input_multimodals=input_multimodals)
        return vision_embedding_inputs

    @torch.inference_mode()
    @logging_timer('CreateModelInputs', logger)
    def create_model_inputs(self, messages: SeqList, is_prefill: bool):
        """Create model inputs from messages.

        Args:
            messages (SeqList): The input messages.
        """
        batch_size = len(messages)
        # history lengths
        history_lengths = torch.tensor([msg.num_history_ids for msg in messages])

        # input ids
        token_ids = [msg.token_ids for msg in messages]
        input_ids = torch.as_tensor(np.concatenate(token_ids))[None]

        # seqlens
        is_decoding = not is_prefill
        if not is_decoding:
            seq_length = [len(tokens) for tokens in token_ids]
            seq_length = torch.tensor(seq_length, dtype=torch.long)
            max_q_seqlen = seq_length.max().item()
        else:
            max_q_seqlen = len(token_ids[0])
            seq_length = torch.full((batch_size, ), max_q_seqlen, dtype=torch.long)
        kv_seqlens = seq_length + history_lengths
        max_kv_seqlen = kv_seqlens.max().item()
        sum_kv_seqlen = kv_seqlens.sum().item()

        # block offsets
        block_offsets = self.scheduler.get_block_tables(messages)
        block_offsets = _tensorlize_block_offsets(block_offsets, dtype=self.torch_int_dtype)

        # num_ignored_history
        num_ignored_history = torch.tensor([msg.num_ignored_history for msg in messages])

        # model_metas
        model_metas = [msg.model_meta for msg in messages]

        # create model inputs for all required fields
        model_inputs = ModelInputs(
            input_ids=input_ids,
            seq_length=seq_length,
            history_lengths=history_lengths,
            block_offsets=block_offsets,
            is_decoding=is_decoding,
            num_ignored_history=num_ignored_history,
            max_q_seqlen=max_q_seqlen,
            max_kv_seqlen=max_kv_seqlen,
            sum_kv_seqlen=sum_kv_seqlen,
            model_metas=model_metas,
        )

        # adapters
        local_adapter_ids = None
        if self.adapter_manager.num_adapters() > 1:
            adapter_names = [msg.adapter_name for msg in messages]
            local_adapter_ids = self.adapter_manager.get_adapter_ids(adapter_names)
            local_adapter_ids = seq_length.new_tensor(local_adapter_ids)
            model_inputs.local_adapter_ids = local_adapter_ids

        # cross for mllama
        cross_length = torch.tensor([msg.num_cross for msg in messages])
        history_cross_length = torch.tensor([msg.num_history_cross for msg in messages])
        if (cross_length + history_cross_length).max().item() > 0:
            model_inputs.cross_length = cross_length
            model_inputs.history_cross_length = history_cross_length

        # vision inputs
        vision_model_inputs = self._create_vision_model_inputs(messages, model_inputs)
        model_inputs.vision_inputs = vision_model_inputs

        # ssm
        if len(self.cache_config.states_shapes) > 0:
            state_offsets = torch.tensor([msg.logical_state for msg in messages])
            model_inputs.state_offsets = state_offsets

        return model_inputs

    def update_running_migration(self, running: SeqList, next_token_ids: np.ndarray, stopped: torch.Tensor,
                                 model_metas: List[Dict[str, Any]]):
        """Update scheduler."""
        if model_metas is None:
            model_metas = [None] * len(running)
        for token, msg, stop, model_meta in zip(next_token_ids, running, stopped, model_metas):
            if msg.status != MessageStatus.MIGRATION_LOCKED:
                continue
            update_token = token

            # fill token
            msg.update_token_ids(update_token, model_meta=model_meta, mode=UpdateTokenMode.PREFILL)
            if stop:
                update_token = _EMPTY_TOKEN
                msg.update_token_ids(update_token, model_meta=model_meta, mode=UpdateTokenMode.PREFILL)
                msg.status = MessageStatus.STOPPED

    def _make_infer_outputs(
        self,
        batched_outputs: BatchedOutputs,
        running: SeqList,
        is_decoding: bool,
    ):
        """Make infer output."""
        new_token_timestamp = batched_outputs.new_token_timestamp
        logits = batched_outputs.logits
        logprobs = batched_outputs.logprobs

        if logprobs is not None:
            logprobs.vals = logprobs.vals.tolist()
            logprobs.indices = logprobs.indices.tolist()

        seq_length = [seq.num_token_ids for seq in running]
        is_run = [seq.status == MessageStatus.LOCKED for seq in running]
        self.seq_strategy.update_running(running=running, batched_outputs=batched_outputs, is_decoding=is_decoding)

        # generate output
        outputs: Dict[int, InferOutput] = dict()
        for idx, msg in enumerate(running):
            if not is_run[idx]:
                continue
            token_ids = msg.generated_ids
            finish = msg.status == MessageStatus.STOPPED or msg.status == MessageStatus.TO_BE_MIGRATED
            if not finish and len(token_ids) == 0:
                continue
            resp_data = msg.resp.data
            if resp_data is not None and len(resp_data.get('token_ids', [])) == len(token_ids):
                # no new tokens
                continue
            session_id = msg.session_id
            if msg.resp_cache:
                cache_block_ids = self.scheduler.block_manager.get_block_table(msg).tolist()
            else:
                cache_block_ids = None

            # logprobs
            num_logprobs = msg.sampling_param.num_logprobs
            cur_logprobs = None
            if num_logprobs >= 0:
                cur_logprobs = (logprobs.vals[idx][:num_logprobs + 1], logprobs.indices[idx][:num_logprobs + 1])

            req_metrics = RequestMetrics(new_token_timestamp, msg.engine_events)
            out = InferOutput(session_id=session_id,
                              resp=msg.resp,
                              finish=finish,
                              token_ids=token_ids,
                              cache_block_ids=cache_block_ids,
                              req_metrics=req_metrics,
                              logprobs=cur_logprobs)
            outputs[session_id] = out

            if msg.return_logits:
                outputs[session_id].logits = logits.split(seq_length)[idx]
        return outputs

    def _make_forward_inputs(self, prefill: bool, enable_empty: bool = False):
        """Make forward inputs."""

        def __need_logits(seqs: SeqList):
            """Need logits."""
            return any(seq.return_logits for seq in seqs)

        def __need_schedule_again(prefill: bool, scheduler_output):
            """Need schedule again."""
            # only reschedule when prefill
            if not prefill:
                return False
            # schedule decoding if no valid prefill reqs.
            if len(scheduler_output.running) > 0:
                return False
            # disable decoding for prefill role
            if (self.engine_config.role == EngineRole.Prefill):
                return False
            # disable decoding if no running reqs.
            if not self.scheduler.has_running():
                logger.warning('No running sequences for decoding scheduling after prefill scheduling.')
                return False
            return True

        scheduler = self.scheduler
        logger.debug(f'Make forward inputs with prefill={prefill}, enable_empty={enable_empty}')

        prealloc_size = self.engine_strategy.get_prealloc_size(not prefill)
        scheduler_output = scheduler.schedule(is_prefill=prefill, prealloc_size=prealloc_size)

        if enable_empty and len(scheduler_output.running) == 0:
            return None

        if __need_schedule_again(prefill, scheduler_output):
            prefill = False
            prealloc_size = self.engine_strategy.get_prealloc_size(not prefill)
            scheduler_output = scheduler.schedule(is_prefill=prefill, prealloc_size=prealloc_size)

        num_loops = self.engine_strategy.get_num_loops(not prefill)
        running = scheduler_output.running
        swap_in_map = scheduler_output.swap_in_map
        swap_out_map = scheduler_output.swap_out_map

        if len(running) == 0:
            return None

        # create inputs
        inputs = self.create_model_inputs(running, prefill)
        sampling_inputs = self.sampling_strategy.make_sampling_inputs(running)
        return_logits = __need_logits(running)
        extra_inputs = self.model_agent_strategy.make_extra_inputs(running)
        stopping_criteria = self.model_agent_strategy.make_stopping_criteria(running)

        sync_long_context = inputs.input_ids.numel() > self.cache_config.max_prefill_token_num

        return dict(
            running=running,
            inputs=inputs,
            swap_in_map=swap_in_map,
            swap_out_map=swap_out_map,
            loop_count=num_loops,
            sampling_inputs=sampling_inputs,
            stopping_criteria=stopping_criteria,
            return_logits=return_logits,
            is_dummy=False,
            sync_long_context=sync_long_context,
            extra_inputs=extra_inputs,
        )

    async def _await_forward_event(self, forward_event: asyncio.Event):
        """Await forward event."""
        await forward_event.wait()

    @torch.inference_mode()
    async def _async_loop_preprocess_message(self, forward_event: asyncio.Event, has_runable_event: RunableEventBase):
        """Preprocess msg."""
        while True:
            await self._await_forward_event(forward_event)
            await self.req_manager.step()
            has_runable_event.set()

    async def _async_loop_send_responses(self, que: asyncio.Queue, forward_event: asyncio.Event):
        """Send responses."""

        def __log_resps(outputs: List[InferOutput]):
            """Log resps."""
            if logger.level <= logging.DEBUG:
                session_ids = [out.session_id for out in outputs]
                logger.debug(f'Response sessions: {session_ids}')
            elif logger.level <= logging.INFO:
                logger.debug(f'Response: num_outputs={len(outputs)}.')

        def __send_resp(out: InferOutput):
            """Send response."""
            resp_type = (ResponseType.FINISH if out.finish else ResponseType.SUCCESS)
            logprobs = None if out.resp.data is None else out.resp.data.get('logprobs', None)
            self._response(out.resp,
                           resp_type,
                           data=dict(token_ids=out.token_ids,
                                     logits=out.logits,
                                     cache_block_ids=out.cache_block_ids,
                                     req_metrics=out.req_metrics,
                                     logprobs=logprobs))

        def __update_logprobs(step_outputs: List[InferOutput]):
            for out in step_outputs:
                cur_logprobs = out.logprobs
                if cur_logprobs is None:
                    continue

                if out.resp.data is None:
                    out.resp.data = dict()
                out.resp.data.setdefault('logprobs', [])

                # logprobs to dict
                vals = cur_logprobs[0]
                indices = cur_logprobs[1]
                cur_logprobs = dict(zip(indices, vals))
                logprobs = out.resp.data['logprobs']
                logprobs.append(cur_logprobs)

        def __send_resps(step_outputs: List[InferOutput]):
            """Send response callback."""
            __log_resps(step_outputs)
            __update_logprobs(step_outputs)

            is_done = set()
            for out in reversed(step_outputs):
                if out.session_id in is_done:
                    continue
                is_done.add(out.session_id)
                __send_resp(out)

        while True:
            num_outs = que.qsize()
            if num_outs > 0:
                resps = []
                for _ in range(num_outs):
                    resps += que.get_nowait().values()
            else:
                resps = (await que.get()).values()
            await self._await_forward_event(forward_event)
            __send_resps(resps)

    async def p2p_initialize(self, init_request: DistServeInitRequest):
        return await self.engine_conn.p2p_initialize(init_request)

    def p2p_connect(self, conn_request: DistServeConnectionRequest):
        return self.engine_conn.p2p_connect(conn_request)

    async def p2p_drop_connect(self, drop_conn_request: DistServeDropConnectionRequest):
        return self.engine_conn.p2p_drop_connect(drop_conn_request)

    @torch.inference_mode()
    async def _async_loop_migration(self, resp_que: asyncio.Queue, has_runable_event: asyncio.Event):
        """Async loop migration."""
        while True:
            migration_running = self.scheduler._schedule_migration()
            if not migration_running and not self.scheduler.has_migration_waiting():
                await self.migration_event.wait()
            elif migration_running:
                self.migration_event.clear()
                for msg in migration_running:
                    migration_execution_requests: List[Tuple[int, List[Tuple[int, int]]]] = []
                    migration_request = msg.migration_request
                    prefill_block_ids = migration_request.remote_block_ids
                    decode_block_ids = list(self.scheduler.block_manager.get_block_table(msg=msg))

                    if not migration_request.is_dummy_prefill:
                        assert len(prefill_block_ids) == len(decode_block_ids), (
                            f'#prefill block ids ({len(prefill_block_ids)}) must equal to '
                            f'#decode block ids ({len(decode_block_ids)})'
                            f'all id length: {len(msg.num_token_ids)}')
                        migration_execution_requests.append((
                            migration_request.remote_engine_id,
                            list(zip(prefill_block_ids, decode_block_ids)),
                        ))
                        migration_inputs = MigrationExecutionBatch(protocol=migration_request.protocol,
                                                                   requests=migration_execution_requests)
                        logger.info(f'migrating session: {msg.session_id} begin')
                        await self.executor.migrate(migration_inputs)
                        logger.info(f'migrating session: {msg.session_id} done')
                        await self.engine_conn.zmq_send(remote_engine_id=migration_request.remote_engine_id,
                                                        remote_session_id=migration_request.remote_session_id)

                # generate output
                outputs: Dict[int, InferOutput] = dict()
                self.scheduler.lock_running_migration(migration_running)
                for _, msg in enumerate(migration_running):
                    session_id = msg.session_id
                    msg.resp.type = ResponseType.SUCCESS
                    token_ids = [msg.migration_request.remote_token_id]
                    # MUST be a wall-clock time
                    new_token_timestamp = time.time()
                    req_metrics = RequestMetrics(new_token_timestamp, msg.engine_events)
                    out = InferOutput(
                        session_id=session_id,
                        resp=msg.resp,
                        finish=False,
                        token_ids=np.array(token_ids),
                        req_metrics=req_metrics,
                    )
                    outputs[session_id] = out
                    self.update_running_migration([msg], np.array([token_ids]), [False], [None])
                resp_que.put_nowait(outputs)
                self.scheduler.unlock_running_migration(migration_running)
                has_runable_event.set()
            else:
                # release coroutine for decoding
                await asyncio.sleep(.5)

    @torch.inference_mode()
    async def _async_loop_main(
        self,
        resp_que: asyncio.Queue,
        forward_event: asyncio.Event,
        has_runable_event: RunableEventBase,
        inputs_maker: InputsMakerBase,
    ):
        """Main loop of the engine.

        Each engine instance would communicate with the engine by queue.
        """
        scheduler = self.scheduler
        forward_inputs = None
        next_running = None

        while True:
            if next_running is None:
                if not scheduler.has_unfinished():
                    forward_event.set()
                    await has_runable_event.wait()
                    forward_event.clear()

                scheduler.collect_migration_done()
                forward_inputs, next_running = await inputs_maker.send_next_inputs()
                if next_running is None:
                    # TODO (JimyMa): add watermark check event instead of async sleep.
                    # self.perfill_watermark_event.wait()
                    logger.warning(f'no next prefill running request, Maybe cache is full, '
                                   f'free gpu cache blocks: {scheduler.block_manager.get_num_free_gpu_blocks()}, '
                                   f'total gpu cache blocks: {scheduler.block_manager.num_gpu_blocks}')
                    forward_event.set()
                    await asyncio.sleep(0.1)
                    forward_event.clear()
                    continue

            forward_event.set()
            num_loops = forward_inputs['loop_count']
            is_decoding = forward_inputs['inputs'].is_decoding
            running = next_running
            next_running = None
            scheduler.lock_running(running)
            for idx in range(num_loops):

                # pre-forward before get last token
                if idx == num_loops - 1:
                    scheduler.collect_migration_done()
                    forward_inputs, next_running = await inputs_maker.prefetch_next_inputs()

                # send output
                out = await self.executor.get_output_async()
                if out is not None:
                    step_outputs = self._make_infer_outputs(out, running=running, is_decoding=is_decoding)
                    resp_que.put_nowait(step_outputs)

                # lock forward event
                # make sure that prefetch forward would not wait for detokenize
                # WARNING: this might have side effect on the performance
                if idx == num_loops // 2:
                    forward_event.clear()

            scheduler.unlock_running(running)
            has_runable_event.set()

    @staticmethod
    def _add_loop_tasks_done_callback(tasks: List[asyncio.Task]):
        """Add loop tasks done callback."""

        def __task_callback(task: asyncio.Task) -> None:
            """Raise exception on finish."""
            task_name = task.get_name()
            try:
                task.result()
            except asyncio.CancelledError:
                logger.debug(f'Task <{task_name}> cancelled.')
                return
            except Exception:
                logger.exception(f'Task <{task_name}> failed')
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()

        for task in tasks:
            task.add_done_callback(__task_callback)

    def _loop_finally(self):
        """Finally process for dist."""
        logger.info('Cleanup executor.')
        self.executor.stop()
        self.executor.release()

    def update_params(self, request: Any):
        """Update params."""
        self.executor.update_params(request)

    def sleep(self, level: int = 1):
        """Sleep."""
        self.executor.sleep(level)

    def wakeup(self, tags: Optional[List[str]] = None):
        """Wakeup."""
        self.executor.wakeup(tags)

    async def async_loop(self):
        try:
            event_loop = asyncio.get_event_loop()

            # forward task
            forward_event = CounterEvent()

            # migration task
            self.migration_event = asyncio.Event()

            logger.info('Starting executor.')
            self.executor.start(forward_event)

            # preprocess task
            logger.info('Starting async task MainLoopPreprocessMessage.')
            has_runable_event = build_runable_event(self.scheduler)
            loop_msg_proc = event_loop.create_task(self._async_loop_preprocess_message(
                forward_event, has_runable_event),
                                                   name='MainLoopPreprocessMessage')

            # response task
            logger.info('Starting async task MainLoopResponse.')
            resp_que = asyncio.Queue()
            loop_send_resp = event_loop.create_task(self._async_loop_send_responses(resp_que, forward_event),
                                                    name='MainLoopResponse')

            loop_main = asyncio.current_task()
            loop_tasks: List[asyncio.Task] = [loop_main, loop_msg_proc, loop_send_resp]

            if self.engine_config.role != EngineRole.Hybrid:
                logger.info('Starting async task MigrationLoop.')
                loop_migration = event_loop.create_task(
                    self._async_loop_migration(resp_que, has_runable_event=has_runable_event),
                    name='MainLoopMigration',
                )
                loop_tasks.append(loop_migration)

            # binding done callback
            self._add_loop_tasks_done_callback(loop_tasks)
            self._loop_main = loop_main

            # main loop
            logger.info('Starting async task MainLoop.')
            inputs_maker = build_inputs_maker(self)
            await self._async_loop_main(resp_que=resp_que,
                                        forward_event=forward_event,
                                        has_runable_event=has_runable_event,
                                        inputs_maker=inputs_maker)
        except Exception as e:
            logger.exception(f'exception happened: {type(e)} {e}')
        finally:
            self._loop_finally()

    def close(self):
        if self.executor.device_type == 'cuda':
            # https://discuss.pytorch.org/t/how-to-delete-a-tensor-in-gpu-to-free-up-memory/48879/32
            # W/O this, repeatedly rebuilding and destroying engines within the same process
            # will cause more and more reserved CUDA memory.
            torch._C._cuda_clearCublasWorkspaces()
        if self._loop_main is not None:
            self._loop_main.cancel()
        else:
            self._loop_finally()

    def create_instance(self, cuda_stream_id=0):
        """Create a pytorch engine instance.

        Args:
            cuda_stream_id(int): identity of a cuda stream
        Returns:
            EngineInstance: an instance of pytorch engine
        """
        from .engine_instance import EngineInstance
        return EngineInstance(self)

    def start_loop(self):
        """Start engine loop."""
        if self.req_manager.is_loop_alive():
            return True
        self.req_manager.create_loop_task()
        return True

    def end_session(self, session_id: int):
        """End session."""
        if session_id in self.scheduler.sessions:
            self.sampling_strategy.on_session_end(session_id)
            self.scheduler.end_session(session_id)
            return True
        return False

    def get_model_config(self):
        return self.model_config

    def get_engine_config(self):
        return self.engine_config

    def get_schedule_metrics(self):
        return self.scheduler.schedule_metrics
