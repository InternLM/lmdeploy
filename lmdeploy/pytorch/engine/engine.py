# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import gc
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from lmdeploy.messages import PytorchEngineConfig, RequestMetrics, ResponseType, SpeculativeConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.conn.engine_conn import EngineP2PConnection
from lmdeploy.pytorch.disagg.conn.protocol import (DistServeConnectionRequest, DistServeDropConnectionRequest,
                                                   DistServeInitRequest)
from lmdeploy.utils import get_logger, get_model

from ..adapter.adapter import AdapterManager
from ..config import CacheConfig, ModelConfig
from ..messages import SchedulerSequence, UpdateTokenMode
from ..paging import Scheduler
from ..strategies import build_strategy_factory
from .base import EngineBase
from .config_builder import ConfigBuilder
from .engine_checker import EngineChecker
from .executor import build_executor
from .request import Request, RequestManager, RequestType, Response

logger = get_logger('lmdeploy')

SeqList = List[SchedulerSequence]


@dataclass
class InferOutput:
    """The output of the model inference."""

    session_id: int
    resp: Response
    token_ids: Union[np.ndarray, List[int]]
    meta: Any = None
    finish: bool = False
    logits: torch.Tensor = None
    logprobs: torch.Tensor = None

    # send cache blocks back for migration in Disaggregated LLM Serving
    # when Prefill Engine is Done.
    cache_block_ids: List[int] = None

    # for logging
    req_metrics: RequestMetrics = None

    # expert ids
    routed_experts: torch.Tensor = None


def _build_seq_meta(cache_config: CacheConfig, seq_strategy: Any, sampling_strategy: Any):
    from lmdeploy.pytorch.messages import SequenceMeta

    seq_meta = SequenceMeta(cache_config.block_size, strategy=seq_strategy, sampling_strategy=sampling_strategy)
    return seq_meta


def response_reqs(req_manager: RequestManager,
                  resp: Response,
                  resp_type: ResponseType,
                  data: Any = None,
                  err_msg: str = ''):
    """response."""
    if resp.type == ResponseType.FINISH:
        return
    resp.type = resp_type
    resp.data = data
    resp.err_msg = err_msg
    req_manager.response(resp)


class Engine(EngineBase):
    """The inference engine of lmdeploy pytorch.

    Args:
        model_path (str): The hugging face model path.
        engine_config (PytorchEngineConfig): The config of the Engine.
        trust_remote_code (bool): Trust remote code.
    """

    def __init__(
        self,
        model_path: str,
        engine_config: PytorchEngineConfig = None,
        trust_remote_code: bool = True,
        speculative_config: SpeculativeConfig = None,
    ) -> None:
        # make sure engine config exist
        engine_config = ConfigBuilder.update_engine_config(engine_config)

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
        scheduler_config = ConfigBuilder.build_scheduler_config(engine_config)
        cache_config = ConfigBuilder.build_cache_config(engine_config)
        backend_config = ConfigBuilder.build_backend_config(engine_config)
        dist_config = ConfigBuilder.build_dist_config(engine_config)
        misc_config = ConfigBuilder.build_misc_config(engine_config)
        # spec decode
        self.specdecode_config = ConfigBuilder.build_specdecode_config(model_path, speculative_config, engine_config,
                                                                       cache_config)

        # build model agent
        self.executor = build_executor(
            model_path,
            cache_config=cache_config,
            backend_config=backend_config,
            dist_config=dist_config,
            misc_config=misc_config,
            adapters=adapters,
            device_type=engine_config.device_type,
            distributed_executor_backend=engine_config.distributed_executor_backend,
            dtype=engine_config.dtype,
            specdecode_config=self.specdecode_config,
        )
        self.executor.init()

        # strategies
        self.strategy_factory = build_strategy_factory(self.model_config, self.executor.misc_config,
                                                       self.specdecode_config)
        self.sampling_strategy = self.strategy_factory.build_sampling_strategy()
        self.model_agent_strategy = self.strategy_factory.build_model_agent_strategy()
        self.engine_strategy = self.strategy_factory.build_engine_strategy(cache_config=cache_config,
                                                                           scheduler_config=scheduler_config)
        self.seq_strategy = self.strategy_factory.build_sequence_strategy()

        self.input_processor = self.executor.get_input_processor()
        cache_config = self.executor.cache_config
        self.adapter_manager = self._build_adapter_manager(adapters)
        self.seq_meta = _build_seq_meta(cache_config,
                                        seq_strategy=self.seq_strategy,
                                        sampling_strategy=self.sampling_strategy)
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
        self.req_manager.set_main_loop_func(self.async_loop)
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
                        speculative_config: SpeculativeConfig = None,
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
            return build_mp_engine(
                backend=backend,
                model_path=pretrained_model_name_or_path,
                engine_config=engine_config,
                trust_remote_code=trust_remote_code,
                speculative_config=speculative_config,
            )
        if len(kwargs) > 0:
            logger.debug(f'Get unexpected kwargs: {kwargs}')
        return cls(
            model_path=pretrained_model_name_or_path,
            engine_config=engine_config,
            trust_remote_code=trust_remote_code,
            speculative_config=speculative_config,
        )

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

    def _response(self, resp: Response, resp_type: ResponseType, data: Any = None, err_msg: str = ''):
        """response."""
        return response_reqs(self.req_manager, resp, resp_type, data, err_msg)

    def _get_max_session_len(self):
        """Get max session len."""
        session_len = self.scheduler_config.max_session_len
        num_gpu_blocks = self.cache_config.num_gpu_blocks - self.cache_config.num_reserved_gpu_blocks
        max_tokens = (num_gpu_blocks * self.cache_config.block_size)
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
                    msgs[0].state.finish()
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
            if self.engine_config.role == EngineRole.Prefill:
                sampling_param.max_new_tokens = 1
            elif max_new_tokens + num_all_tokens > max_session_len:
                logger.warning(
                    f'session[{msg.session_id}]: num tokens is larger than max session len {max_session_len}. '
                    f'Update max_new_tokens={max_session_len - num_all_tokens}.')
                sampling_param.max_new_tokens = max_session_len - num_all_tokens

        scheduler = self.scheduler
        for req in reqs:
            session_id = req.data['session_id']
            sess = scheduler.sessions.get(session_id, None)
            if sess is None:
                self._response(req.resp, ResponseType.SESSION_NOT_EXIST)
                continue
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
                if migration_request:
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
                msg.state.activate()

            __update_max_new_tokens(msg)
            msg.resp = req.resp

    @property
    def model_config(self) -> ModelConfig:
        """Model config."""
        return self.executor.model_config

    async def p2p_initialize(self, init_request: DistServeInitRequest):
        return await self.engine_conn.p2p_initialize(init_request)

    def p2p_connect(self, conn_request: DistServeConnectionRequest):
        return self.engine_conn.p2p_connect(conn_request)

    async def p2p_drop_connect(self, drop_conn_request: DistServeDropConnectionRequest):
        return self.engine_conn.p2p_drop_connect(drop_conn_request)

    def _loop_finally(self):
        """Finally process for dist."""
        logger.info('Cleanup executor.')
        self.migration_event = None
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
        engine_loop = None
        try:
            from lmdeploy.pytorch.engine.engine_loop import build_engine_loop
            self._loop_main = asyncio.current_task()
            event_loop = asyncio.get_event_loop()

            # create engine loop
            engine_loop = build_engine_loop(self)
            self.migration_event = engine_loop.migration_event
            forward_event = engine_loop.forward_event

            # start executor
            logger.info('Starting executor.')
            self.executor.start(forward_event)

            # start engine loop
            engine_loop.create_tasks(event_loop)
            await engine_loop.wait_tasks()
        except asyncio.CancelledError:
            logger.info('Engine main loop cancelled.')
            raise
        except BaseException:
            logger.exception('Engine main loop failed.')
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
            self.scheduler.end_session(session_id)
            return True
        return False

    def get_engine_config(self):
        return self.engine_config

    def get_schedule_metrics(self):
        return self.scheduler.schedule_metrics
