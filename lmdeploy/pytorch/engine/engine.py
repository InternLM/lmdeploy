# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from lmdeploy.messages import PytorchEngineConfig, ResponseType
from lmdeploy.utils import get_logger, get_max_batch_size, get_model, logging_timer

from ..adapter.adapter import AdapterManager
from ..config import BackendConfig, CacheConfig, ModelConfig, SchedulerConfig
from ..messages import MessageStatus, SchedulerSequence
from ..model_inputs import ModelInputs, VisionModelInputs
from ..paging import Scheduler
from .engine_checker import EngineChecker
from .executor import build_executor
from .logits_process import SamplingInputs
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


def _tensorlize_block_offsets(block_offsets, dtype=torch.int32):
    """tensorlize block_offsets."""
    from torch.nn.utils.rnn import pad_sequence
    block_offsets = [torch.from_numpy(off).to(dtype) for off in block_offsets]
    block_offsets = pad_sequence(block_offsets, batch_first=True)
    return block_offsets


def _build_scheduler_config(engine_config: PytorchEngineConfig):
    """build scheduler config."""
    scheduler_config = SchedulerConfig(max_batches=engine_config.max_batch_size,
                                       max_session_len=engine_config.session_len,
                                       prefill_interval=engine_config.prefill_interval)
    return scheduler_config


def _build_cache_config(engine_config: PytorchEngineConfig):
    """build cache config."""
    cache_config = CacheConfig(
        max_batches=engine_config.max_batch_size,
        block_size=engine_config.block_size,
        num_cpu_blocks=engine_config.num_cpu_blocks,
        num_gpu_blocks=engine_config.num_gpu_blocks,
        cache_max_entry_count=engine_config.cache_max_entry_count,
        max_prefill_token_num=engine_config.max_prefill_token_num,
        enable_prefix_caching=engine_config.enable_prefix_caching,
        quant_policy=engine_config.quant_policy,
        device_type=engine_config.device_type,
    )
    return cache_config


def _build_backend_config(engine_config: PytorchEngineConfig):
    """build backend config."""
    backend_config = BackendConfig(
        eager_mode=engine_config.eager_mode,
        device_type=engine_config.device_type,
    )
    return backend_config


class Engine:
    """The inference engine of lmdeploy pytorch.

    Args:
        model_path (str): The hugging face model path.
        tokenizer (lmdeploy.Tokenizer): an instance of lmdeploy.Tokenizer
        engine_config (PytorchEngineConfig): The config of the Engine.
        trust_remote_code (bool): Trust remote code.
    """

    def __init__(self,
                 model_path: str,
                 tokenizer: object,
                 engine_config: PytorchEngineConfig = None,
                 trust_remote_code: bool = True) -> None:
        # make sure engine exits
        if engine_config is None:
            engine_config = PytorchEngineConfig()
        else:
            engine_config = copy.deepcopy(engine_config)
        if engine_config.max_batch_size is None:
            engine_config.max_batch_size = get_max_batch_size(engine_config.device_type)

        tp = engine_config.tp
        dp = 1

        self.tokenizer = tokenizer
        self.tp = tp
        self.dp = dp

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

        # build model agent
        raw_tokenizer = None
        if tokenizer is not None:
            raw_tokenizer = tokenizer.model.model
        self.executor = build_executor(model_path,
                                       cache_config=cache_config,
                                       backend_config=backend_config,
                                       tokenizer=raw_tokenizer,
                                       adapters=adapters,
                                       tp=tp,
                                       dp=dp,
                                       device_type=engine_config.device_type,
                                       distributed_executor_backend=engine_config.distributed_executor_backend,
                                       dtype=engine_config.dtype)
        self.executor.init()

        self.input_processor = self.executor.get_input_processor()
        cache_config = self.executor.cache_config
        self.adapter_manager = self._build_adapter_manager(adapters)
        self.scheduler = Scheduler(scheduler_config, cache_config)

        # engine args
        self.model_path = model_path
        self.engine_config = engine_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.backend_config = backend_config
        self.max_session_len = self._get_max_session_len()

        self.req_manager = self._bind_request_manager()

        # create main thread
        self._start_loop()
        self._loop_main = None

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        tokenizer: object,
                        engine_config: PytorchEngineConfig = None,
                        trust_remote_code: bool = True,
                        **kwargs):
        """lmdeploy python inference engine.

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
            tokenizer (lmdeploy.Tokenizer): an instance of lmdeploy.Tokenizer
            engine_config (PytorchEngineConfig): Pytorch engine config.
            trust_remote_code (bool): Trust remote code
        """
        if len(kwargs) > 0:
            logger.debug(f'Get unexpected kwargs: {kwargs}')
        return cls(model_path=pretrained_model_name_or_path,
                   tokenizer=tokenizer,
                   engine_config=engine_config,
                   trust_remote_code=trust_remote_code)

    def _download_adapters(self, adapters: Dict[str, str], engine_config: PytorchEngineConfig):
        """download adapters."""
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
        """bind request manager."""
        req_manager = RequestManager()
        req_manager.bind_func(RequestType.ADD_SESSION, self._on_add_session)
        req_manager.bind_func(RequestType.STOP_SESSION, self._on_stop_session)
        req_manager.bind_func(RequestType.END_SESSION, self._on_end_session)
        req_manager.bind_func(RequestType.ADD_MESSAGE, self._on_add_message)
        return req_manager

    def _start_loop(self):
        """start loop."""
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
        """get max session len."""
        session_len = self.scheduler_config.max_session_len
        max_tokens = (self.cache_config.num_gpu_blocks * self.cache_config.block_size)
        window_size = self.cache_config.window_size
        if window_size > 0 and window_size <= max_tokens:
            max_tokens = (1 << 63) - 1
        if session_len is None:
            session_len = max_tokens
        else:
            session_len = min(max_tokens, session_len)
        return session_len

    def _on_add_session(self, reqs: List[Request], **kwargs):
        """on add session callback."""
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
        """on stop session callback."""
        for req in reqs:
            session_id = req.data['session_id']
            resp = req.data.get('response', True)
            resp_type = ResponseType.SESSION_NOT_EXIST
            if session_id in self.scheduler.sessions:
                self.scheduler.stop_session(session_id)
                session = self.scheduler.sessions[session_id]
                for seq in session.sequences.values():
                    resp: Response = getattr(seq, 'resp', None)
                    if resp is not None:
                        resp.type = ResponseType.FINISH
                        self.req_manager.response(resp)
                resp_type = ResponseType.SUCCESS
            if resp:
                self._response(req.resp, resp_type)

    def _on_end_session(self, reqs: List[Request], **kwargs):
        """on end session callback."""
        for req in reqs:
            session_id = req.data['session_id']
            resp = req.data.get('response', True)
            resp_type = ResponseType.SESSION_NOT_EXIST
            if session_id in self.scheduler.sessions:
                self.scheduler.end_session(session_id)
                resp_type = ResponseType.SUCCESS
            if resp:
                self._response(req.resp, resp_type)

    def _on_add_message(self, reqs: List[Request], **kwargs):
        """on add message callback."""
        for req in reqs:
            req_data = req.data
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
            result = self.input_processor.preprocess_input(input_ids, input_multimodals)

            input_ids = result.input_ids
            input_multimodals = result.input_multimodals

            req_data['token_ids'] = input_ids
            req_data['input_multimodals'] = input_multimodals

        if len(reqs) > 0:
            self._add_message(reqs)

    def _add_message(self, reqs: List[Request]):

        def __update_max_new_tokens(msg):
            """update max new tokens."""
            max_session_len = self.max_session_len
            sampling_param = msg.sampling_param
            sampling_param.max_new_tokens = min(sampling_param.max_new_tokens, max_session_len - msg.num_all_tokens())

        for req in reqs:
            session_id = req.data['session_id']
            if session_id not in self.scheduler.sessions:
                self._response(req.resp, ResponseType.SESSION_NOT_EXIST)
                continue
            session_id = req.data['session_id']
            sess = self.scheduler.sessions[session_id]
            # TODO: support 1 session n sequence
            sampling_param = req.data['sampling_param']
            return_logits = sampling_param.out_logits
            if len(sess.sequences) == 0:
                assert len(req.data['token_ids']) > 0, ('Empty input is not allowed.')
                sess.add_sequence(
                    req.data['token_ids'],
                    sampling_param=sampling_param,
                    adapter_name=req.data['adapter_name'],
                    return_logits=return_logits,
                    multimodals=req.data.get('input_multimodals'),
                    input_embeddings=req.data.get('input_embeddings'),
                )
                msg = next(iter(sess.sequences.values()))
                __update_max_new_tokens(msg)
                self.scheduler.add_sequence(msg)
            else:
                msg = next(iter(sess.sequences.values()))
                msg.update_token_ids(
                    req.data['token_ids'],
                    multimodals=req.data.get('input_multimodals'),
                    embeddings=req.data.get('input_embeddings'),
                )
                msg.num_new_tokens = 0
                msg.sampling_param = sampling_param
                msg.return_logits = return_logits
                msg.status = MessageStatus.WAITING
                __update_max_new_tokens(msg)

            msg.resp = req.resp

    @property
    def model_config(self) -> ModelConfig:
        """model config."""
        return self.executor.model_config

    @property
    def gpu_count(self):
        return self.tp * self.dp

    @property
    def torch_int_dtype(self):
        """return int32 for cuda, int64 for others."""
        if self.executor.device_type == 'cuda':
            return torch.int32
        return torch.int64

    @logging_timer('CreateModelInputs', logger)
    def create_model_inputs(self, messages: SeqList, is_prefill: bool):
        """create model inputs from messages.

        Args:
            messages (SeqList): The input messages.
        """
        history_lengths = [msg.history_len for msg in messages]
        history_lengths = torch.tensor(history_lengths)

        token_ids = [msg.token_ids for msg in messages]

        if isinstance(token_ids[0], int):
            token_ids = [token_ids]

        batch_size = len(messages)
        input_ids = torch.from_numpy(np.concatenate(token_ids))

        is_decoding = not is_prefill
        if not is_decoding:
            seq_length = [len(tokens) for tokens in token_ids]
            seq_length = torch.tensor(seq_length, dtype=torch.long)
        else:
            seq_length = torch.ones(batch_size, dtype=torch.long)
        max_q_seq_length = seq_length.max().item()

        block_offsets = self.scheduler.get_block_tables(messages)
        block_offsets = _tensorlize_block_offsets(block_offsets, dtype=self.torch_int_dtype)

        local_adapter_ids = None
        if self.adapter_manager.num_adapters() > 1:
            adapter_names = [msg.adapter_name for msg in messages]
            local_adapter_ids = self.adapter_manager.get_adapter_ids(adapter_names)
            local_adapter_ids = seq_length.new_tensor(local_adapter_ids)

        # add batch dim [bs=1, seq_len]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        num_ignored_history = [msg.num_ignored_history for msg in messages]
        num_ignored_history = torch.tensor(num_ignored_history)

        model_metas = [msg.model_meta for msg in messages]

        def __get_vlm_embeddings():
            """get vlm input embeddings and indexings."""
            input_embeddings = [[
                emb.embeddings if isinstance(emb.embeddings, torch.Tensor) else torch.from_numpy(emb.embeddings)
                for emb in msg.input_embeddings
            ] for msg in messages]
            input_embedding_ranges = [
                torch.tensor([[emb.start, emb.end] for emb in msg.input_embeddings]) for msg in messages
            ]
            input_embedding_indexing = torch.zeros((batch_size, max_q_seq_length), dtype=torch.bool)
            for msg_id, msg in enumerate(messages):
                for emb in msg.input_embeddings:
                    # make slice index relative to embeddings
                    emb_start = emb.start - msg.history_len
                    emb_end = emb.end - msg.history_len
                    input_embedding_indexing[msg_id][emb_start:emb_end] = True
            return (input_embeddings, input_embedding_indexing, input_embedding_ranges)

        # for inputs with embeddings
        history_image_nums = None
        history_image_token_lengths = None

        input_embeddings = None
        input_embedding_indexing = None
        input_embedding_ranges = None
        has_embedding = any([len(msg.input_embeddings) > 0 for msg in messages])
        if has_embedding:
            (input_embeddings, input_embedding_indexing, input_embedding_ranges) = __get_vlm_embeddings()

        input_multimodals = None
        has_multimodal = any([not msg.history_multimodals.empty() for msg in messages])
        if has_multimodal:
            has_multimodal = False
            input_multimodals = [msg.get_input_multimodals() for msg in messages]
            for input_mm in input_multimodals:
                for val in input_mm.values():
                    if len(val) > 0:
                        has_multimodal = True
                        break
                if has_multimodal:
                    break

        vision_embedding_inputs = None
        if has_embedding or has_multimodal or history_image_nums is not None:
            vision_embedding_inputs = VisionModelInputs(history_lengths=history_lengths,
                                                        history_image_nums=history_image_nums,
                                                        history_image_token_lengths=history_image_token_lengths,
                                                        input_embeddings=input_embeddings,
                                                        input_embedding_indexing=input_embedding_indexing,
                                                        input_embedding_ranges=input_embedding_ranges,
                                                        input_multimodals=input_multimodals)

        # cross
        cross_length = torch.tensor([msg.num_cross for msg in messages])
        history_cross_length = torch.tensor([msg.num_history_cross for msg in messages])
        if (cross_length + history_cross_length).max().item() == 0:
            cross_length = None
            history_cross_length = None

        return ModelInputs(
            input_ids=input_ids,
            seq_length=seq_length,
            history_lengths=history_lengths,
            block_offsets=block_offsets,
            is_decoding=is_decoding,
            num_ignored_history=num_ignored_history,
            local_adapter_ids=local_adapter_ids,
            vision_inputs=vision_embedding_inputs,
            cross_length=cross_length,
            history_cross_length=history_cross_length,
            model_metas=model_metas,
        )

    @logging_timer('UpdateRunning', logger)
    def update_running(self, running: SeqList, next_token_ids: torch.Tensor, stopped: torch.Tensor,
                       model_metas: List[Dict[str, Any]]):
        """update scheduler."""
        if model_metas is None:
            model_metas = [None] * len(running)
        next_token_ids = next_token_ids.numpy()
        for token, msg, stop, model_meta in zip(next_token_ids, running, stopped, model_metas):
            if msg.status != MessageStatus.LOCKED:
                continue
            update_token = token
            if stop:
                update_token = _EMPTY_TOKEN
            else:
                msg.num_new_tokens += 1
            msg.update_token_ids(update_token, model_meta=model_meta)
            if stop:
                msg.status = MessageStatus.STOPPED

    def _do_prefill(self):
        # decoding if no waiting
        if not self.scheduler.has_waiting():
            return False
        num_running = self.scheduler.num_running()
        num_waiting = self.scheduler.num_waiting()
        max_batches = self.scheduler_config.max_batches
        # prefill if too much waiting
        if num_waiting >= 4:
            return True
        # prefill if no enough running
        if num_running < max_batches * 0.5:
            return True
        # decoding
        return False

    def _make_infer_outputs(self, next_token_ids: torch.LongTensor, running: SeqList, logits: torch.Tensor,
                            stopped: torch.Tensor, model_metas: List[Dict[str, Any]]):
        """make infer output."""

        seq_length = [seq.num_token_ids for seq in running]
        is_run = [seq.status == MessageStatus.LOCKED for seq in running]
        stopped = stopped.tolist()
        self.update_running(running, next_token_ids, stopped, model_metas)

        # generate output
        outputs: Dict[int, InferOutput] = dict()
        for idx, msg in enumerate(running):
            if not is_run[idx]:
                continue
            token_ids = msg.all_ids[-msg.num_new_tokens:]
            finish = msg.status == MessageStatus.STOPPED
            if not finish and len(token_ids) == 0:
                continue
            session_id = msg.session_id
            out = InferOutput(
                session_id=session_id,
                resp=msg.resp,
                finish=finish,
                token_ids=token_ids,
            )
            outputs[session_id] = out

            if msg.return_logits:
                outputs[session_id].logits = logits.split(seq_length)[idx]
        return outputs

    def _make_forward_inputs(self, prefill: bool = None, enable_empty: bool = False):
        """make forward inputs."""
        prefill_interval = self.scheduler_config.prefill_interval

        def __gather_all_ids(seqs: SeqList, sampling_inputs: SamplingInputs):
            """gather history."""
            if sampling_inputs.repetition_penalty is None and not any(sampling_inputs.logits_processors):
                return None
            batch = len(seqs)
            max_len = max(seq.num_all_ids for seq in seqs)
            pad_id = self.model_config.bos_token_id
            pad_id = 0 if pad_id is None else pad_id
            output = torch.full((batch, max_len), pad_id, dtype=torch.int64)
            for idx, seq in enumerate(seqs):
                h_len = seq.num_all_ids
                if h_len == 0:
                    continue
                h_ids = torch.from_numpy(seq.all_ids)
                output[idx, -h_len:] = h_ids
            return output

        def __gather_guided_input_ids(seqs: SeqList, sampling_inputs: SamplingInputs):
            """gather input ids for guided decode."""
            if not any(sampling_inputs.response_formats or ()):
                return None
            batch = len(seqs)
            max_len = max(seq.num_new_tokens for seq in seqs)
            pad_id = self.model_config.bos_token_id
            pad_id = 0 if pad_id is None else pad_id
            output = torch.full((batch, max_len), pad_id, dtype=torch.int64)
            for idx, seq in enumerate(seqs):
                h_len = seq.num_new_tokens
                if h_len == 0:
                    continue
                h_ids = torch.from_numpy(seq.all_ids[-seq.num_new_tokens:])
                output[idx, -h_len:] = h_ids
            return output

        def __get_num_appendable_ids(seqs: SeqList):
            """get num appendable ids."""
            ret = [seq.sampling_param.max_new_tokens - seq.num_new_tokens for seq in seqs]
            return torch.tensor(ret)

        def __get_num_ignore_eos(seqs: SeqList):
            """get num ignore eos."""
            ret = [seq.sampling_param.min_new_tokens - seq.num_new_tokens for seq in seqs]
            return torch.tensor(ret)

        def __need_logits(seqs: SeqList):
            """need logits."""
            return any(seq.return_logits for seq in seqs)

        if prefill is None:
            prefill = self._do_prefill()
        scheduler_output = self.scheduler.schedule(is_prefill=prefill, prealloc_size=prefill_interval)

        if enable_empty and len(scheduler_output.running) == 0:
            return None

        # schedule decoding if no valid prefill reqs.
        if prefill and len(scheduler_output.running) == 0:
            prefill = False
            scheduler_output = self.scheduler.schedule(is_prefill=prefill, prealloc_size=prefill_interval)

        running = scheduler_output.running
        swap_in_map = scheduler_output.swap_in_map
        swap_out_map = scheduler_output.swap_out_map
        assert len(running) > 0

        # create inputs
        inputs = self.create_model_inputs(running, prefill)
        sampling_inputs = SamplingInputs.from_sampling_params(running)
        all_ids = __gather_all_ids(running, sampling_inputs)
        guided_input_ids = __gather_guided_input_ids(running, sampling_inputs)
        num_appendable_ids = __get_num_appendable_ids(running)
        num_ignore_eos = __get_num_ignore_eos(running)
        return_logits = __need_logits(running)

        num_loops = 1 if prefill else prefill_interval

        return dict(running=running,
                    inputs=inputs,
                    swap_in_map=swap_in_map,
                    swap_out_map=swap_out_map,
                    all_ids=all_ids,
                    guided_input_ids=guided_input_ids,
                    sampling_inputs=sampling_inputs,
                    num_appendable_ids=num_appendable_ids,
                    num_ignore_eos=num_ignore_eos,
                    loop_count=num_loops,
                    return_logits=return_logits)

    def _set_has_runable_event(self, has_runable_event: asyncio.Event):
        """set has runable event."""
        if self.scheduler.has_unfinished():
            has_runable_event.set()
        else:
            has_runable_event.clear()

    async def _await_forward_event(self, forward_event: asyncio.Event):
        """await forward event."""
        if self.scheduler.has_unfinished():
            await forward_event.wait()

    @torch.inference_mode()
    async def _async_loop_preprocess_message(self, forward_event: asyncio.Event, has_runable_event: asyncio.Event):
        """preprocess msg."""
        while True:
            await self._await_forward_event(forward_event)
            await self.req_manager.step()
            self._set_has_runable_event(has_runable_event)

    @torch.inference_mode()
    async def _async_loop_background(self, in_que: asyncio.Queue, out_que: asyncio.Queue, forward_event: asyncio.Event):
        """async loop background."""
        while True:
            forward_inputs = await in_que.get()

            forward_event.clear()
            await self.model_agent._async_step_background(
                **forward_inputs,
                output_que=out_que,
            )
            forward_event.set()

    async def _async_loop_send_responses(self, que: asyncio.Queue, forward_event: asyncio.Event):
        """send responses."""

        def __send_resp(out: InferOutput):
            """send response."""
            resp_type = (ResponseType.FINISH if out.finish else ResponseType.SUCCESS)
            self._response(out.resp, resp_type, data=dict(token_ids=out.token_ids, logits=out.logits))

        def __send_resps(step_outputs: List[InferOutput]):
            """send response callback."""
            for out in step_outputs:
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

    @torch.inference_mode()
    async def _async_loop_main(self, resp_que: asyncio.Queue, has_runable_event: asyncio.Event):
        """Main loop of the engine.

        Each engine instance would communicate with the engine by queue.
        """

        forward_inputs = None
        next_running = None

        async def _send_next_inputs(prefill: bool = None, enable_empty: bool = False):
            nonlocal forward_inputs, next_running
            forward_inputs = self._make_forward_inputs(prefill, enable_empty)
            if forward_inputs is None:
                forward_inputs = None
                next_running = None
                return
            next_running = forward_inputs.pop('running')
            await self.executor.forward_async(forward_inputs)

        async def _prefetch_next_inputs():
            enable = False
            prefill = self._do_prefill()
            if prefill:
                enable = True
            else:
                num_running = self.scheduler.num_running()
                is_decoding = forward_inputs['inputs'].is_decoding
                running_threshold = (self.scheduler_config.max_batches // 4) if is_decoding else 0

                if num_running > running_threshold:
                    enable = True

            if enable:
                # send next forward
                await _send_next_inputs(prefill, True)

        while True:
            if next_running is None:
                await has_runable_event.wait()
                await _send_next_inputs()
            num_loops = forward_inputs['loop_count']
            running = next_running
            next_running = None
            self.scheduler.lock_running(running)
            for idx in range(num_loops):
                if idx >= num_loops - 1:
                    await _prefetch_next_inputs()
                out = await self.executor.get_output_async()
                step_outputs = self._make_infer_outputs(**out, running=running)
                resp_que.put_nowait(step_outputs)
            self.scheduler.unlock_running(running)
            self._set_has_runable_event(has_runable_event)

    @staticmethod
    def _add_loop_tasks_done_callback(tasks: List[asyncio.Task]):
        """add loop tasks done callback."""

        def __task_callback(task: asyncio.Task) -> None:
            """raise exception on finish."""
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
        """finally process for dist."""
        self.executor.stop()
        self.executor.release()

    async def async_loop(self):
        try:
            event_loop = asyncio.get_event_loop()

            # forward task
            forward_event = asyncio.Event()
            forward_event.set()

            logger.debug('Starting executor.')
            self.executor.start(forward_event)

            # preprocess task
            logger.debug('Starting async task MainLoopPreprocessMessage.')
            has_runable_event = asyncio.Event()
            loop_msg_proc = event_loop.create_task(self._async_loop_preprocess_message(
                forward_event, has_runable_event),
                                                   name='MainLoopPreprocessMessage')

            # response task
            logger.debug('Starting async task MainLoopResponse.')
            resp_que = asyncio.Queue()
            loop_send_resp = event_loop.create_task(self._async_loop_send_responses(resp_que, forward_event),
                                                    name='MainLoopResponse')

            # binding done callback
            loop_main = asyncio.current_task()
            loop_tasks: List[asyncio.Task] = [loop_main, loop_msg_proc, loop_send_resp]
            self._add_loop_tasks_done_callback(loop_tasks)
            self._loop_main = loop_main

            # main loop
            logger.debug('Starting async task MainLoop.')
            await self._async_loop_main(resp_que=resp_que, has_runable_event=has_runable_event)
        finally:
            self._loop_finally()

    def close(self):
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
