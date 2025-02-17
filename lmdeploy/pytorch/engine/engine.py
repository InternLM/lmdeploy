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
from ..config import BackendConfig, CacheConfig, SchedulerConfig
from ..devices import DeviceContext, get_device_manager
from ..messages import MessageStatus, SchedulerSequence
from ..model_inputs import ModelInputs, VisionModelInputs
from ..paging import Scheduler
from .engine_checker import EngineChecker
from .logits_process import FusedLogitsProcessor, SamplingInputs
from .model_agent import build_model_agent
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


def _tensorlize_block_offsets(block_offsets):
    """tensorlize block_offsets."""
    from torch.nn.utils.rnn import pad_sequence
    block_offsets = [torch.from_numpy(off) for off in block_offsets]
    block_offsets = pad_sequence(block_offsets, batch_first=True)
    return block_offsets


def _check_finish(scheduler: Scheduler, current_iter: int):
    """dynamic prefill interval."""
    if not scheduler.has_waiting():
        return False
    scheduler_config = scheduler.scheduler_config
    max_prefill_interval = scheduler_config.prefill_interval
    max_batches = scheduler_config.max_batches
    num_batches = len(scheduler.running)
    ratio = num_batches / max_batches
    min_iter = max_prefill_interval * ratio
    if current_iter >= min_iter:
        return True
    return False


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
        if engine_config is None:
            engine_config = PytorchEngineConfig()
        else:
            engine_config = copy.deepcopy(engine_config)
        if engine_config.max_batch_size is None:
            engine_config.max_batch_size = get_max_batch_size(engine_config.device_type)

        checker = EngineChecker(model_path=model_path,
                                engine_config=engine_config,
                                trust_remote_code=trust_remote_code,
                                logger=logger)
        checker.handle()

        self.tokenizer = tokenizer
        adapters = engine_config.adapters
        self.engine_config = engine_config
        self.tp = engine_config.tp

        self.device_context = DeviceContext(device_type=engine_config.device_type)

        if not os.path.exists(model_path):
            model_path = get_model(model_path, engine_config.download_dir, engine_config.revision)
        self.model_path = model_path

        if adapters is not None and len(adapters) > 0:
            adapters = self._download_adapters(adapters, engine_config)

        scheduler_config = _build_scheduler_config(engine_config)
        cache_config = _build_cache_config(engine_config)
        backend_config = _build_backend_config(engine_config)

        with get_device_manager().context(self.device_context):
            self.model_agent = build_model_agent(model_path,
                                                 cache_config=cache_config,
                                                 backend_config=backend_config,
                                                 trust_remote_code=trust_remote_code,
                                                 adapters=adapters,
                                                 tp=self.tp,
                                                 dtype=engine_config.dtype,
                                                 custom_module_map=engine_config.custom_module_map)

        self.input_processor = self.model_agent.get_input_processor()

        cache_config = self.model_agent.cache_config
        self.adapter_manager = self._build_adapter_manager(adapters)
        self.scheduler = Scheduler(scheduler_config, cache_config)

        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.backend_config = backend_config
        self.stream = self.model_agent.stream
        self.max_session_len = self._get_max_session_len()

        self.req_manager = self._bind_request_manager()

        # create main thread
        self._start_loop()
        self._output_stream = torch.cuda.Stream()

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

    def _on_add_session(self, reqs: Request, **kwargs):
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

    def _on_stop_session(self, reqs: Request, **kwargs):
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

    def _on_end_session(self, reqs: Request, **kwargs):
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

    def _on_add_message(self, reqs: Request, **kwargs):
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

    def _add_message(self, reqs):

        def __update_bad_words(msg):
            """update bad words."""
            sampling_param = msg.sampling_param
            eos_token_id = self.model_config.eos_token_id
            if eos_token_id is None:
                return
            if sampling_param.ignore_eos:
                sampling_param.bad_words += eos_token_id
            else:
                for eid in eos_token_id:
                    if eid not in sampling_param.stop_words:
                        sampling_param.stop_words.append(eid)

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
                __update_bad_words(msg)
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
                __update_bad_words(msg)
                __update_max_new_tokens(msg)

            msg.resp = req.resp

    @property
    def model_config(self):
        """model config."""
        return self.model_agent.model_config

    @property
    def gpu_count(self):
        return self.tp

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
        block_offsets = _tensorlize_block_offsets(block_offsets)

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

    def _batch_stopping_criteria(self, token_ids: torch.Tensor, stop_words: torch.Tensor,
                                 num_appendable_ids: torch.Tensor):
        """batched stopping criteria."""
        num_appendable_ids = num_appendable_ids - 1
        # one more step to cache last token(stop word)
        stopped = num_appendable_ids < 0
        if stop_words is not None:
            sw_stopped = (token_ids[:, None] == stop_words).any(1)
            one_ids = torch.clamp_max(num_appendable_ids, 0)
            num_appendable_ids = torch.where(sw_stopped, one_ids, num_appendable_ids)
        return stopped, num_appendable_ids

    @logging_timer('SamplingLogits', logger)
    async def async_sampling_logits(self, logits: torch.Tensor, all_ids: torch.Tensor, guided_input_ids: torch.Tensor,
                                    sampling_inputs: SamplingInputs, inputs: ModelInputs, ignore_eos: torch.Tensor):
        """sampling logits."""

        def __get_last_logits():
            """get last logits."""
            seq_length = inputs.seq_length
            if len(seq_length) == logits.size(0):
                return logits

            last_idx = seq_length.cumsum(-1) - 1
            return logits[last_idx, :]

        split_logits = __get_last_logits()
        logits_processor = FusedLogitsProcessor(sampling_inputs, ignore_eos, self.tokenizer.model.model)
        logits = await logits_processor(all_ids, guided_input_ids, split_logits)
        next_token_ids = logits_processor.sampling(logits)

        return next_token_ids

    @logging_timer('UpdateRunning', logger)
    def update_running(self, running: SeqList, next_token_ids: torch.Tensor, stopped: torch.Tensor,
                       model_metas: List[Dict[str, Any]]):
        """update scheduler."""
        if model_metas is None:
            model_metas = [None] * len(running)
        next_token_ids = next_token_ids.numpy()
        for token, msg, stop, model_meta in zip(next_token_ids, running, stopped, model_metas):
            if msg.status != MessageStatus.RUNNING:
                continue
            update_token = token
            if stop:
                update_token = _EMPTY_TOKEN
            else:
                msg.num_new_tokens += 1
            msg.update_token_ids(update_token, model_meta=model_meta)
            if stop:
                msg.status = MessageStatus.STOPPED

    @logging_timer('ModelForward', logger)
    async def _async_model_forward(self, inputs: ModelInputs, swap_in_map: Dict, swap_out_map: Dict,
                                   return_logits: bool):
        """model forward."""
        max_prefill_token_num = self.cache_config.max_prefill_token_num
        swap_done = False

        class _OutputGather:
            """output gather."""

            def __init__(self, max_seq_len):
                self._max_seq_len = max_seq_len
                self._start = 0
                self._output = None

            def gather(self, output):
                """gather."""
                tmp_output = output['hidden_states']

                if not return_logits:
                    self._output = tmp_output
                    return

                out_logits = self._output
                start = self._start
                seq_len = tmp_output.size(-2)
                if out_logits is None:
                    out_logits = tmp_output.new_empty(1, self._max_seq_len, tmp_output.size(-1), device='cpu')
                out_logits[:, start:start + seq_len].copy_(tmp_output, non_blocking=True)
                self._start = start + seq_len
                self._output = out_logits

            def get_output(self):
                """get tmp_output."""
                if not return_logits:
                    return self._output[:, -1:]
                torch.cuda.synchronize()
                return self._output

        async def __forward(inputs):
            """forward."""
            nonlocal swap_done, swap_in_map, swap_out_map
            if swap_done:
                return await self.model_agent.async_forward(inputs, swap_in_map=dict(), swap_out_map=dict())
            else:
                swap_done = True
                return await self.model_agent.async_forward(inputs, swap_in_map=swap_in_map, swap_out_map=swap_out_map)

        async def __long_context_single_forward(inputs):
            """one large sequence."""
            seq_len = inputs.seq_length
            max_seq_len = inputs.seq_length[0]
            batch_size = seq_len.size(0)
            assert batch_size == 1

            new_inputs = inputs.split(max_prefill_token_num)

            model_metas = new_inputs[0].model_metas
            output_gather = _OutputGather(max_seq_len)
            for inp in new_inputs:
                inp.model_metas = model_metas
                tmp_out = await __forward(inp)
                model_metas = tmp_out.get('model_metas')
                output_gather.gather(tmp_out)
                tmp_out.pop('hidden_states', None)
            tmp_out['hidden_states'] = output_gather.get_output()
            return tmp_out

        if inputs.input_ids.numel() <= max_prefill_token_num:
            ret = await __forward(inputs)
            if not return_logits and not inputs.is_decoding:
                last_token_loc = inputs.seq_length.cumsum(0) - 1
                ret['hidden_states'] = ret['hidden_states'][:, last_token_loc]
        else:
            ret = await __long_context_single_forward(inputs)
            if not return_logits and not inputs.is_decoding:
                last_token_loc = [-1]
                ret['hidden_states'] = ret['hidden_states'][:, last_token_loc]
            else:
                ret['hidden_states'] = ret['hidden_states'].to('cuda')

        hidden_states = ret.pop('hidden_states')
        logits = self.model_agent.get_logits(hidden_states)
        ret['logits'] = logits
        return ret

    async def _make_infer_outputs(self, next_token_ids: torch.LongTensor, logits: torch.Tensor, stopped: torch.Tensor,
                                  model_metas: List[Dict[str, Any]], event: torch.cuda.Event):
        """make infer output."""

        def __get_q_start_loc():
            inputs = self._inputs
            seq_length = inputs.seq_length
            batch_size = len(seq_length)
            if inputs.is_decoding:
                return torch.arange(0, batch_size)
            else:
                return seq_length.cumsum(0) - seq_length

        while not event.query():
            await asyncio.sleep(0.001)
        with torch.cuda.stream(self._output_stream):
            next_token_ids = next_token_ids.cpu()
            stopped = stopped.cpu()

        running = self._running
        is_run = [seq.status == MessageStatus.RUNNING for seq in running]
        stopped = stopped.tolist()
        self.update_running(running, next_token_ids, stopped, model_metas)

        # generate output
        next_token_ids = next_token_ids.tolist()
        q_start_loc = __get_q_start_loc()
        outputs: Dict[int, InferOutput] = dict()
        for idx, msg in enumerate(running):
            if not is_run[idx]:
                continue
            token_ids = msg.all_ids[-msg.num_new_tokens:]
            finish = msg.status == MessageStatus.STOPPED
            if not finish and len(token_ids) == 0:
                continue
            session_id = msg.session_id
            resp = msg.resp
            out = InferOutput(
                session_id=session_id,
                resp=resp,
                finish=finish,
                token_ids=token_ids,
            )
            outputs[session_id] = out

            if msg.return_logits:
                inputs = self._inputs
                start = q_start_loc[idx]
                seqlen = inputs.seq_length[idx]
                outputs[session_id].logits = logits[start:start + seqlen]
        return outputs

    async def _async_step_background(self, inputs: ModelInputs, swap_in_map: Dict, swap_out_map: Dict,
                                     all_ids: torch.Tensor, guided_input_ids: torch.Tensor,
                                     sampling_inputs: SamplingInputs, num_appendable_ids: torch.LongTensor,
                                     num_ignore_eos: torch.LongTensor, loop_count: int, return_logits: bool,
                                     output_que: asyncio.Queue):
        """asyc forward task."""

        def __update_inputs(next_token_ids):
            """update inputs."""
            nonlocal all_ids, guided_input_ids
            inputs.update(next_token_ids)
            if all_ids is not None:
                all_ids = torch.cat([all_ids, next_token_ids[:, None].to(all_ids.device)], 1)
            if guided_input_ids is not None:
                guided_input_ids = torch.cat([guided_input_ids, next_token_ids[:, None].to(guided_input_ids.device)], 1)
            if sampling_inputs.random_offsets is not None:
                sampling_inputs.random_offsets += 1

        logger.debug('<ForwardTask>: '
                     f'batch_size={inputs.seq_length.size(0)} '
                     f'num_tokens={inputs.input_ids.size(-1)}')
        inputs = inputs.to_device('cuda')
        is_decoding = inputs.is_decoding
        if all_ids is not None:
            all_ids = all_ids.cuda()
        if guided_input_ids is not None:
            guided_input_ids = guided_input_ids.cuda()
        sampling_inputs = sampling_inputs.to_device('cuda')
        num_appendable_ids = num_appendable_ids.cuda()
        num_ignore_eos = num_ignore_eos.cuda()

        for idx in range(loop_count):
            # inference
            output = await self._async_model_forward(inputs,
                                                     swap_in_map=swap_in_map,
                                                     swap_out_map=swap_out_map,
                                                     return_logits=return_logits)
            logits = output['logits']
            logits = logits[0]  # [bs, seq, prob] -> [seq, prob]

            # sampling
            next_token_ids = await self.async_sampling_logits(logits, all_ids, guided_input_ids, sampling_inputs,
                                                              inputs, num_ignore_eos > 0)
            num_ignore_eos = num_ignore_eos - 1

            # stopping criteria
            stopped, num_appendable_ids = self._batch_stopping_criteria(next_token_ids, sampling_inputs.stop_words,
                                                                        num_appendable_ids)

            # send output
            model_metas = output.get('model_metas')
            finish = (idx == loop_count - 1)
            finish = finish or _check_finish(self.scheduler, idx)
            event = torch.cuda.Event()
            event.record()
            output = dict(next_token_ids=next_token_ids,
                          logits=logits,
                          stopped=stopped,
                          model_metas=model_metas,
                          event=event)
            output_que.put_nowait((finish, output))

            inputs.model_metas = model_metas

            if finish:
                break

            # update for next loop
            if is_decoding:
                swap_in_map = dict()
                swap_out_map = dict()
                __update_inputs(next_token_ids)

    def _set_has_runable_event(self, has_runable_event: asyncio.Event):
        """set has runable event."""
        if self.scheduler.has_unfinished():
            has_runable_event.set()
        else:
            has_runable_event.clear()

    @torch.inference_mode()
    async def _async_loop_preprocess_message(self, forward_event: asyncio.Event, has_runable_event: asyncio.Event):
        """preprocess msg."""
        while True:
            if self.scheduler.has_unfinished():
                await forward_event.wait()
            await self.req_manager.step()
            self._set_has_runable_event(has_runable_event)

    @torch.inference_mode()
    async def _async_loop_background(self, in_que: asyncio.Queue, out_que: asyncio.Queue, forward_event: asyncio.Event):
        """async loop background."""

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

        while True:
            is_prefill, scheduler_output = await in_que.get()
            running = scheduler_output.running
            swap_in_map = scheduler_output.swap_in_map
            swap_out_map = scheduler_output.swap_out_map
            prefill_interval = self.scheduler_config.prefill_interval
            loop_count = 1 if is_prefill else (prefill_interval - 1)
            assert len(running) > 0

            # create inputs
            inputs = self.create_model_inputs(running, is_prefill)
            sampling_inputs = SamplingInputs.from_sampling_params(running)
            all_ids = __gather_all_ids(running, sampling_inputs)
            guided_input_ids = __gather_guided_input_ids(running, sampling_inputs)
            num_appendable_ids = __get_num_appendable_ids(running)
            num_ignore_eos = __get_num_ignore_eos(running)
            return_logits = __need_logits(running)

            self._running = running
            self._inputs = inputs

            forward_event.clear()
            await self._async_step_background(
                inputs=inputs,
                swap_in_map=swap_in_map,
                swap_out_map=swap_out_map,
                all_ids=all_ids,
                guided_input_ids=guided_input_ids,
                sampling_inputs=sampling_inputs,
                num_appendable_ids=num_appendable_ids,
                num_ignore_eos=num_ignore_eos,
                loop_count=loop_count,
                return_logits=return_logits,
                output_que=out_que,
            )
            forward_event.set()

    async def _async_send_responses(self, que: asyncio.Queue, forward_event: asyncio.Event):
        """send responses."""

        def __send_resp(out: InferOutput):
            """send response."""
            resp_type = (ResponseType.FINISH if out.finish else ResponseType.SUCCESS)
            self._response(out.resp, resp_type, data=dict(token_ids=out.token_ids, logits=out.logits))

        def __send_resps(step_outputs: Dict[int, InferOutput]):
            """send response callback."""
            for out in step_outputs.values():
                __send_resp(out)

        while True:
            resps = await que.get()
            if self.scheduler.has_unfinished():
                await forward_event.wait()
            __send_resps(resps)

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
                for task in tasks:
                    if not task.cancelled():
                        task.cancel()

        for task in tasks:
            task.add_done_callback(__task_callback)

    @torch.inference_mode()
    async def _async_loop(self):
        """Main loop of the engine.

        Each engine instance would communicate with the engine by queue.
        """
        event_loop = asyncio.get_event_loop()
        prefill_interval = self.scheduler_config.prefill_interval

        # forward task
        in_que = asyncio.Queue()
        out_que = asyncio.Queue()
        forward_event = asyncio.Event()
        forward_event.set()
        loop_background = event_loop.create_task(self._async_loop_background(in_que, out_que, forward_event),
                                                 name='MainLoopBackground')

        # preprocess task
        has_runable_event = asyncio.Event()
        loop_msg_proc = event_loop.create_task(self._async_loop_preprocess_message(forward_event, has_runable_event),
                                               name='MainLoopPreprocessMessage')

        # response task
        resp_que = asyncio.Queue()
        loop_send_resp = event_loop.create_task(self._async_send_responses(resp_que, forward_event),
                                                name='MainLoopResponse')

        loop_main = asyncio.current_task()
        loop_tasks: List[asyncio.Task] = [loop_main, loop_background, loop_msg_proc, loop_send_resp]
        self._add_loop_tasks_done_callback(loop_tasks)

        def __do_prefill():
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

        async def __step():
            """step decoding."""
            prefill = __do_prefill()
            schedule_output = self.scheduler.schedule(is_prefill=prefill, prealloc_size=prefill_interval)
            # schedule decoding if no valid prefill reqs.
            if prefill and len(schedule_output.running) == 0:
                prefill = False
                schedule_output = self.scheduler.schedule(is_prefill=prefill, prealloc_size=prefill_interval)

            in_que.put_nowait((prefill, schedule_output))
            finish = False
            while not finish:
                finish, out = await out_que.get()
                step_outputs = await self._make_infer_outputs(**out)
                self._set_has_runable_event(has_runable_event)
                resp_que.put_nowait(step_outputs)

        while True:
            await has_runable_event.wait()
            await __step()

    async def async_loop(self):
        device_manager = get_device_manager()
        with device_manager.context(self.device_context), torch.cuda.stream(self.stream):
            await self._async_loop()

    def close(self):
        self.model_agent.close()

    def create_instance(self, cuda_stream_id=0):
        """Create a pytorch engine instance.

        Args:
            cuda_stream_id(int): identity of a cuda stream
        Returns:
            EngineInstance: an instance of pytorch engine
        """
        from .engine_instance import EngineInstance
        return EngineInstance(self)
