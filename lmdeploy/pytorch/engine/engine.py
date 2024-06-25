# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from lmdeploy.messages import (EngineGenerationConfig, PytorchEngineConfig,
                               ResponseType)
from lmdeploy.utils import get_logger, get_model, logging_timer

from ..adapter.adapter import AdapterManager, SchedulerAdapter
from ..check_env import check_adapters, check_env, check_model
from ..config import CacheConfig, SchedulerConfig
from ..devices import DeviceContext, get_device_manager
from ..messages import (InputEmbeddingRangeType, InputEmbeddingType,
                        MessageStatus, SchedulerSequence)
from ..paging import Scheduler
from .logits_process import FusedLogitsProcessor, SamplingInputs
from .model_agent import (AdapterInfo, AutoModelAgent, ModelInputs,
                          VisionModelInputs)
from .request import Request, RequestManager, RequestType, Response

logger = get_logger('lmdeploy')

SeqList = List[SchedulerSequence]
AdapterList = List[SchedulerAdapter]


def _raise_exception_on_finish(task: asyncio.Task) -> None:
    """raise exception on finish."""
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as e:
        raise e


class NoRunningSeqs(Exception):
    """NoRunningSeqs."""
    pass


@dataclass
class InferOutput:
    """The output of the model inference."""

    session_id: int
    token_ids: List[int]
    sender_id: int
    req_id: int
    meta: Any = None
    finish: bool = False
    logits: torch.Tensor = None


def _paging_adapters(adapters: dict, model_agent: AutoModelAgent,
                     scheduler: Scheduler):
    adapters = adapters or dict()
    weight_maps = []
    for name in adapters:
        weight_map = scheduler.add_adapter(name)
        weight_map.rank_offset = torch.tensor(weight_map.rank_offset)
        weight_maps.append(weight_map)
    model_agent.paging_adapters(weight_maps)


def _tensorlize_block_offsets(block_offsets):
    """tensorlize block_offsets."""
    from torch.nn.utils.rnn import pad_sequence
    block_offsets = [torch.from_numpy(off) for off in block_offsets]
    block_offsets = pad_sequence(block_offsets, batch_first=True)
    return block_offsets


def _get_adapter_ids(seqs: SeqList, adapters: AdapterList):
    """get adapter ids."""
    adapter_names_map = dict(
        (ada.name, idx) for idx, ada in enumerate(adapters))
    adapter_ids = [adapter_names_map[seq.adapter_name] for seq in seqs]
    return adapter_ids


class Engine:
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
        check_env(engine_config.device_type)
        check_model(model_path, trust_remote_code)
        if engine_config is None:
            engine_config = PytorchEngineConfig()
        if engine_config.adapters is not None:
            check_adapters(list(engine_config.adapters.values()))

        self.engine_config = engine_config
        model_name = engine_config.model_name
        tp = engine_config.tp

        self.tp = tp
        self.model_name = model_name

        self.device_context = DeviceContext(
            device_type=engine_config.device_type)

        scheduler_config = SchedulerConfig(
            max_batches=engine_config.max_batch_size,
            max_session_len=engine_config.session_len,
            eviction_type=engine_config.eviction_type,
            prefill_interval=engine_config.prefill_interval)

        # block_size = 1 to enable unified paging
        adapters = engine_config.adapters
        cache_config = CacheConfig(
            block_size=engine_config.block_size,
            num_cpu_blocks=engine_config.num_cpu_blocks,
            num_gpu_blocks=engine_config.num_gpu_blocks,
            cache_max_entry_count=engine_config.cache_max_entry_count,
            max_prefill_token_num=engine_config.max_prefill_token_num,
            enable_prefix_caching=engine_config.enable_prefix_caching,
        )

        if not os.path.exists(model_path):
            model_path = get_model(model_path, engine_config.download_dir,
                                   engine_config.revision)
        self.model_path = model_path

        with get_device_manager().context(self.device_context):
            self.model_agent = AutoModelAgent.from_pretrained(
                model_path,
                cache_config=cache_config,
                trust_remote_code=trust_remote_code,
                adapters=adapters,
                tp=tp)

        cache_config = self.model_agent.cache_config
        self.adapter_manager = self._build_adapter_manager(adapters)
        self.scheduler = Scheduler(scheduler_config, cache_config,
                                   self.adapter_manager)

        if adapters:
            _paging_adapters(adapters,
                             model_agent=self.model_agent,
                             scheduler=self.scheduler)

        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.stream = torch.cuda.Stream()

        self.req_manager = self._bind_request_manager()

        # create main thread
        self._start_loop()
        self._create_buffers()
        self.engine_instance = self.create_instance()

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
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
            engine_config (PytorchEngineConfig): Pytorch engine config.
            trust_remote_code (bool): Trust remote code
        """
        if len(kwargs) > 0:
            logger.debug(f'Get unexpected kwargs: {kwargs}')
        return cls(model_path=pretrained_model_name_or_path,
                   engine_config=engine_config,
                   trust_remote_code=trust_remote_code)

    @property
    def tokenizer(self):
        """create tokenizer."""
        from lmdeploy.tokenizer import Tokenizer
        if not hasattr(self, '_tokenizer'):
            self._tokenizer = Tokenizer(self.model_path)
        return self._tokenizer

    def _create_buffers(self):
        max_batches = self.scheduler_config.max_batches

        # buffers to create inputs
        self._seq_length_buf = torch.ones(max_batches, dtype=torch.long)

    def _build_adapter_manager(self, adapters):
        if adapters is not None and len(adapters) > 0:
            linear_info = self.model_agent.get_loralinear_info()
        else:
            linear_info = dict()
        block_numel = self.model_agent.get_block_numel()
        return AdapterManager(linear_info, block_numel)

    def _bind_request_manager(self):
        """bind request manager."""
        req_manager = RequestManager(self.engine_config.thread_safe)
        req_manager.bind_func(RequestType.ADD_SESSION, self._on_add_session)
        req_manager.bind_func(RequestType.STOP_SESSION, self._on_stop_session)
        req_manager.bind_func(RequestType.END_SESSION, self._on_end_session)
        req_manager.bind_func(RequestType.ADD_MESSAGE, self._on_add_message)
        return req_manager

    def _start_loop(self):
        """start loop."""
        return self.req_manager.start_loop(self.async_loop)

    def _response(self,
                  resp_type: ResponseType,
                  sender_id: int,
                  req_id: int,
                  data: Any = None,
                  err_msg: str = ''):
        """response."""
        self.req_manager.response(
            Response(type=resp_type,
                     sender_id=sender_id,
                     req_id=req_id,
                     data=data,
                     err_msg=err_msg))

    def _on_add_session(self, reqs: Request, **kwargs):
        """on add session callback."""
        for req in reqs:
            session_id = req.data['session_id']
            resp_type = ResponseType.SESSION_REPEAT
            if session_id not in self.scheduler.sessions:
                self.scheduler.add_session(session_id)
                resp_type = ResponseType.SUCCESS
            self._response(resp_type, req.sender_id, req.req_id)

    def _on_stop_session(self, reqs: Request, **kwargs):
        """on stop session callback."""
        for req in reqs:
            session_id = req.data['session_id']
            resp_type = ResponseType.SESSION_NOT_EXIST
            if session_id in self.scheduler.sessions:
                self.scheduler.stop_session(session_id)
                resp_type = ResponseType.SUCCESS
            self._response(resp_type, req.sender_id, req.req_id)

    def _on_end_session(self, reqs: Request, **kwargs):
        """on end session callback."""
        for req in reqs:
            session_id = req.data['session_id']
            resp_type = ResponseType.SESSION_NOT_EXIST
            if session_id in self.scheduler.sessions:
                self.scheduler.end_session(session_id)
                resp_type = ResponseType.SUCCESS
            self._response(resp_type, req.sender_id, req.req_id)

    def _on_add_message(self, reqs: Request, **kwargs):
        """on add message callback."""

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
            max_session_len = self.scheduler_config.max_session_len
            if max_session_len is not None:
                sampling_param = msg.sampling_param
                sampling_param.max_new_tokens = min(
                    sampling_param.max_new_tokens,
                    max_session_len - msg.num_all_tokens())

        for req in reqs:
            session_id = req.data['session_id']
            if session_id not in self.scheduler.sessions:
                self._response(ResponseType.SESSION_NOT_EXIST, req.sender_id,
                               req.req_id)
                continue
            session_id = req.data['session_id']
            sess = self.scheduler.sessions[session_id]
            # TODO: support 1 session n sequence
            if len(sess.sequences) == 0:
                assert len(
                    req.data['token_ids']) > 0, ('Empty input is not allowed.')
                sess.add_sequence(
                    req.data['token_ids'],
                    sampling_param=req.data['sampling_param'],
                    adapter_name=req.data['adapter_name'],
                    return_logits=req.data.get('return_logits', False),
                    input_embeddings=req.data.get('input_embeddings'),
                )
                msg = next(iter(sess.sequences.values()))
                __update_bad_words(msg)
                __update_max_new_tokens(msg)
                self.scheduler.add_sequence(msg)
            else:
                msg = next(iter(sess.sequences.values()))
                msg.update_token_ids(req.data['token_ids'],
                                     req.data.get('input_embeddings'))
                msg.num_new_tokens = 0
                msg.sampling_param = req.data['sampling_param']
                msg.return_logits = req.data.get('return_logits', False)
                msg.status = MessageStatus.WAITING
                __update_bad_words(msg)
                __update_max_new_tokens(msg)

            msg.sender_id = req.sender_id
            msg.req_id = req.req_id

    @property
    def model_config(self):
        """model config."""
        return self.model_agent.model_config

    @property
    def gpu_count(self):
        return self.tp

    @logging_timer('CreateModelInputs', logger)
    def create_model_inputs(self, messages: SeqList, adapters: AdapterList,
                            is_prefill: bool):
        """create model inputs from messages.

        Args:
            messages (SeqList): The input messages.
            adapters (AdapterList): Adapters.
        """
        history_lengths = [msg.history_len for msg in messages]
        history_lengths = torch.tensor(history_lengths)

        token_ids = [msg.token_ids for msg in messages]

        meta = messages[0].meta

        if isinstance(token_ids[0], int):
            token_ids = [token_ids]

        batch_size = len(messages)
        input_ids = torch.from_numpy(np.concatenate(token_ids))

        is_decoding = not is_prefill
        if not is_decoding:
            seq_length = [len(tokens) for tokens in token_ids]
            seq_length = torch.tensor(seq_length, dtype=torch.long)
        else:
            seq_length = self._seq_length_buf[:batch_size]
        max_q_seq_length = seq_length.max().item()
        max_history_length = history_lengths.max().item()

        # TODO: get block offsets is slow when block_size = 1
        block_offsets = self.scheduler.get_block_tables(messages)
        block_offsets = _tensorlize_block_offsets(block_offsets)

        local_adapter_ids = None
        adapter_info = None
        if self.adapter_manager.num_adapters() > 1:
            local_adapter_ids = _get_adapter_ids(messages, adapters)
            local_adapter_ids = seq_length.new_tensor(local_adapter_ids)
            adapter_info = AdapterInfo.from_adapters(adapters)

        # add batch dim [bs=1, seq_len]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        num_ignored_history = [msg.num_ignored_history for msg in messages]
        num_ignored_history = torch.tensor(num_ignored_history)

        def __get_cogvlm_image_info():
            """Get cogvlm history image info for position ids."""
            history_image_nums = torch.LongTensor(
                [msg.history_image_num for msg in messages])
            history_image_token_lengths = torch.LongTensor(
                [msg.history_image_token_len for msg in messages])
            return history_image_nums, history_image_token_lengths

        def __get_vlm_embeddings():
            """get vlm input embeddings and indexings."""
            input_embeddings = [[
                emb.embeddings if isinstance(emb.embeddings, torch.Tensor) else
                torch.from_numpy(emb.embeddings)
                for emb in msg.input_embeddings
            ] for msg in messages]
            input_embedding_ranges = [
                torch.tensor([[emb.start, emb.end]
                              for emb in msg.input_embeddings])
                for msg in messages
            ]
            input_embedding_indexing = torch.zeros(
                (batch_size, max_q_seq_length), dtype=torch.bool)
            for msg_id, msg in enumerate(messages):
                for emb in msg.input_embeddings:
                    # make slice index relative to embeddings
                    emb_start = emb.start - msg.history_len
                    emb_end = emb.end - msg.history_len
                    input_embedding_indexing[msg_id][emb_start:emb_end] = True
            return (input_embeddings, input_embedding_indexing,
                    input_embedding_ranges)

        # for inputs with embeddings
        history_image_nums = None
        history_image_token_lengths = None
        # only for cogvlm
        if self.model_config.model_arch == 'CogVLMForCausalLM':
            (history_image_nums,
             history_image_token_lengths) = __get_cogvlm_image_info()

        input_embeddings = None
        input_embedding_indexing = None
        input_embedding_ranges = None
        has_embedding = any(
            [len(msg.input_embeddings) > 0 for msg in messages])
        if has_embedding:
            (input_embeddings, input_embedding_indexing,
             input_embedding_ranges) = __get_vlm_embeddings()

        vision_embedding_inputs = None
        if has_embedding or history_image_nums is not None:
            vision_embedding_inputs = VisionModelInputs(
                history_lengths=history_lengths,
                history_image_nums=history_image_nums,
                history_image_token_lengths=history_image_token_lengths,
                input_embeddings=input_embeddings,
                input_embedding_indexing=input_embedding_indexing,
                input_embedding_ranges=input_embedding_ranges)

        return ModelInputs(input_ids=input_ids,
                           seq_length=seq_length,
                           history_lengths=history_lengths,
                           block_offsets=block_offsets,
                           max_q_seq_length=max_q_seq_length,
                           max_history_length=max_history_length,
                           is_decoding=is_decoding,
                           num_ignored_history=num_ignored_history,
                           local_adapter_ids=local_adapter_ids,
                           adapter_info=adapter_info,
                           vision_inputs=vision_embedding_inputs,
                           meta=meta)

    def _batch_stopping_criteria(self, token_ids: torch.Tensor,
                                 stop_words: torch.Tensor,
                                 num_appendable_ids: torch.Tensor):
        """batched stopping criteria."""
        num_appendable_ids = num_appendable_ids - 1
        # one more step to cache last token(stop word)
        stopped = num_appendable_ids < 0
        if stop_words is not None:
            sw_stopped = (token_ids[:, None] == stop_words).any(1)
            one_ids = torch.clamp_max(num_appendable_ids, 0)
            num_appendable_ids = torch.where(sw_stopped, one_ids,
                                             num_appendable_ids)
        return stopped, num_appendable_ids

    @logging_timer('SamplingLogits', logger)
    def async_sampling_logits(self, logits: torch.Tensor,
                              history_ids: torch.Tensor,
                              sampling_inputs: SamplingInputs,
                              inputs: ModelInputs, ignore_eos: torch.Tensor):
        """sampling logits."""

        def __get_last_logits():
            """get last logits."""
            if inputs.is_decoding:
                return logits

            seq_length = inputs.seq_length
            last_idx = seq_length.cumsum(-1) - 1
            return logits[last_idx, :]

        split_logits = __get_last_logits().cuda()
        logits_processor = FusedLogitsProcessor(sampling_inputs, ignore_eos)
        logits = logits_processor(history_ids, split_logits)
        next_token_ids = logits_processor.sampling(logits)
        next_token_ids = next_token_ids

        return next_token_ids

    @logging_timer('UpdateRunning', logger)
    def update_running(self, running: SeqList, next_token_ids: torch.Tensor,
                       stopped: torch.Tensor):
        """update scheduler."""
        next_token_ids = next_token_ids.numpy()
        for token, msg, stop in zip(next_token_ids, running, stopped):
            if msg.status != MessageStatus.RUNNING:
                continue
            msg.num_new_tokens += 1
            update_token = token
            if stop:
                update_token = np.empty((0, ), dtype=np.int64)
            msg.update_token_ids(update_token)
            if stop:
                msg.status = MessageStatus.STOPPED

    @logging_timer('ModelForward', logger)
    async def _async_model_forward(self, inputs: ModelInputs,
                                   swap_in_map: Dict, swap_out_map: Dict):
        """model forward."""
        max_prefill_token_num = self.cache_config.max_prefill_token_num
        swap_done = False

        class _LogitsGather:
            """logits gather."""

            def __init__(self, max_seq_len):
                self._max_seq_len = max_seq_len
                self._start = 0
                self._out_logits = None

            def gather(self, output):
                """gather."""
                logits = output['logits']
                out_logits = self._out_logits
                start = self._start
                seq_len = logits.size(-2)
                if out_logits is None:
                    out_logits = logits.new_empty(1,
                                                  self._max_seq_len,
                                                  logits.size(-1),
                                                  device='cpu')
                out_logits[:, start:start + seq_len].copy_(logits,
                                                           non_blocking=True)
                self._start = start + seq_len
                self._out_logits = out_logits

            def get_logits(self):
                """get logits."""
                torch.cuda.synchronize()
                return self._out_logits

        async def __forward(inputs):
            """forward."""
            nonlocal swap_done, swap_in_map, swap_out_map
            if swap_done:
                return await self.model_agent.async_forward(
                    inputs, swap_in_map=dict(), swap_out_map=dict())
            else:
                swap_done = True
                return await self.model_agent.async_forward(
                    inputs, swap_in_map=swap_in_map, swap_out_map=swap_out_map)

        async def __long_context_single_forward(inputs):
            """one large sequence."""
            seq_len = inputs.seq_length
            max_seq_len = inputs.seq_length[0]
            batch_size = seq_len.size(0)
            assert batch_size == 1

            new_inputs = inputs.split(max_prefill_token_num,
                                      self.cache_config.block_size)

            logits_gather = _LogitsGather(max_seq_len)
            for inp in new_inputs:
                tmp_out = await __forward(inp)
                logits_gather.gather(tmp_out)
                tmp_out.pop('logits', None)
            tmp_out['logits'] = logits_gather.get_logits()
            return tmp_out

        if inputs.input_ids.numel() <= max_prefill_token_num:
            return await __forward(inputs)
        else:
            return await __long_context_single_forward(inputs)

    def _make_infer_outputs(self, next_token_ids: torch.LongTensor,
                            logits: torch.Tensor, stopped: torch.Tensor):
        """make infer output."""

        def __get_out_token_ids(token: torch.Tensor, msg: SchedulerSequence,
                                stopped: bool):
            """check if output is necessary."""
            if stopped:
                return []
            if token in msg.sampling_param.stop_words:
                return []
            return [token]

        def __get_q_start_loc():
            inputs = self._inputs
            seq_length = inputs.seq_length
            batch_size = len(seq_length)
            if inputs.is_decoding:
                return torch.arange(0, batch_size)
            else:
                return seq_length.cumsum(0) - seq_length

        running = self._running
        is_run = [seq.status == MessageStatus.RUNNING for seq in running]
        stopped = stopped.tolist()
        self.update_running(running, next_token_ids, stopped)

        # generate output
        next_token_ids = next_token_ids.tolist()
        q_start_loc = __get_q_start_loc()
        outputs: Dict[int, InferOutput] = dict()
        for idx, msg in enumerate(running):
            if not is_run[idx]:
                continue
            token_ids = __get_out_token_ids(next_token_ids[idx], msg,
                                            stopped[idx])
            finish = msg.status == MessageStatus.STOPPED
            if not finish and len(token_ids) == 0:
                continue
            session_id = msg.session_id
            out = InferOutput(
                session_id=session_id,
                sender_id=msg.sender_id,
                req_id=msg.req_id,
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

    async def _async_step_background(
            self, inputs: ModelInputs, swap_in_map: Dict, swap_out_map: Dict,
            history_ids: torch.Tensor, sampling_inputs: SamplingInputs,
            num_appendable_ids: torch.LongTensor,
            num_ignore_eos: torch.LongTensor, output_que: asyncio.Queue):
        """asyc forward task."""

        def __update_inputs(next_token_ids):
            """update inputs."""
            nonlocal history_ids
            inputs.update(next_token_ids)
            if history_ids is not None:
                history_ids = torch.cat([
                    history_ids, next_token_ids[:, None].to(history_ids.device)
                ], 1)
            if sampling_inputs.random_offsets is not None:
                sampling_inputs.random_offsets += 1

        logger.debug('<ForwardTask>: '
                     f'batch_size={inputs.seq_length.size(0)} '
                     f'num_tokens={inputs.input_ids.size(-1)}')
        is_decoding = inputs.is_decoding
        if history_ids is not None:
            history_ids = history_ids.cuda()
        sampling_inputs = sampling_inputs.to_device('cuda')
        num_appendable_ids = num_appendable_ids.cuda()
        num_ignore_eos = num_ignore_eos.cuda()

        loop_count = (1 if not is_decoding else
                      self.scheduler_config.prefill_interval - 1)

        for idx in range(loop_count):
            # inference
            output = await self._async_model_forward(inputs,
                                                     swap_in_map=swap_in_map,
                                                     swap_out_map=swap_out_map)
            logits = output['logits']
            logits = logits[0]  # [bs, seq, prob] -> [seq, prob]

            # sampling
            next_token_ids = self.async_sampling_logits(
                logits, history_ids, sampling_inputs, inputs,
                num_ignore_eos > 0)
            num_ignore_eos = num_ignore_eos - 1

            # stopping criteria
            stopped, num_appendable_ids = self._batch_stopping_criteria(
                next_token_ids, sampling_inputs.stop_words, num_appendable_ids)

            # send output
            stopped = stopped.cpu()
            finish = stopped.all().item() or (idx == loop_count - 1)
            output = (next_token_ids.cpu(), logits, stopped)
            output_que.put_nowait((finish, output))

            if finish:
                break

            # update for next loop
            if is_decoding:
                swap_in_map = dict()
                swap_out_map = dict()
                __update_inputs(next_token_ids)

    @torch.inference_mode()
    async def _async_loop_background(self, in_que: asyncio.Queue,
                                     out_que: asyncio.Queue):
        """async loop background."""

        def __gather_history(seqs: SeqList, sampling_inputs: SamplingInputs):
            """gather history."""
            if sampling_inputs.repetition_penalty is None:
                return None
            batch = len(seqs)
            max_len = max(seq.history_len for seq in seqs)
            pad_id = self.model_config.bos_token_id
            pad_id = 0 if pad_id is None else pad_id
            output = torch.full((batch, max_len), pad_id, dtype=torch.int64)
            for idx, seq in enumerate(seqs):
                h_len = seq.history_len
                if h_len == 0:
                    continue
                h_ids = torch.from_numpy(seq.history_ids)
                output[idx, -h_len:] = h_ids
            return output

        def __get_num_appendable_ids(seqs: SeqList):
            """get num appendable ids."""
            ret = [
                seq.sampling_param.max_new_tokens - seq.num_new_tokens
                for seq in seqs
            ]
            return torch.tensor(ret)

        def __get_num_ignore_eos(seqs: SeqList):
            """get num ignore eos."""
            ret = [
                seq.sampling_param.min_new_tokens - seq.num_new_tokens
                for seq in seqs
            ]
            return torch.tensor(ret)

        while True:
            is_prefill = await in_que.get()
            try:
                prefill_interval = self.scheduler_config.prefill_interval
                schedule_output = self.scheduler.schedule(
                    is_prefill=is_prefill, prealloc_size=prefill_interval)
                running: SeqList = schedule_output.running
                adapters = schedule_output.adapters
                if len(running) == 0:
                    raise NoRunningSeqs()

                # create inputs
                inputs = self.create_model_inputs(running, adapters,
                                                  is_prefill)
                sampling_inputs = SamplingInputs.from_sampling_params(running)
                history_ids = __gather_history(running, sampling_inputs)
                num_appendable_ids = __get_num_appendable_ids(running)
                num_ignore_eos = __get_num_ignore_eos(running)

                self._running = running
                self._inputs = inputs

                await self._async_step_background(
                    inputs=inputs,
                    swap_in_map=schedule_output.swap_in_map,
                    swap_out_map=schedule_output.swap_out_map,
                    history_ids=history_ids,
                    sampling_inputs=sampling_inputs,
                    num_appendable_ids=num_appendable_ids,
                    num_ignore_eos=num_ignore_eos,
                    output_que=out_que,
                )
            except Exception as e:
                out_que.put_nowait((True, e))
            finally:
                in_que.task_done()

    @torch.inference_mode()
    async def _async_loop(self):
        """Main loop of the engine.

        Each engine instance would communicate with the engine by queue.
        """
        in_que = asyncio.Queue()
        out_que = asyncio.Queue()
        loop_background = asyncio.get_event_loop().create_task(
            self._async_loop_background(in_que, out_que),
            name='MainLoopBackground')
        loop_background.add_done_callback(_raise_exception_on_finish)

        def __send_resp(out: InferOutput):
            """send response."""
            resp_type = (ResponseType.FINISH
                         if out.finish else ResponseType.SUCCESS)
            self._response(resp_type,
                           sender_id=out.sender_id,
                           req_id=out.req_id,
                           data=dict(token_ids=out.token_ids,
                                     logits=out.logits))

        def __send_resps(step_outputs: Dict[int, InferOutput]):
            """send response callback."""
            for out in step_outputs.values():
                __send_resp(out)

        async def __step(prefill: bool):
            """step decoding."""
            in_que.put_nowait(prefill)
            finish = False
            while not finish:
                finish, out = await out_que.get()
                try:
                    if isinstance(out, Exception):
                        raise out
                    next_token_ids, logits, stopped = out
                    step_outputs = self._make_infer_outputs(
                        next_token_ids, logits, stopped)
                    __send_resps(step_outputs)
                except NoRunningSeqs:
                    break
                except Exception as e:
                    raise e
                finally:
                    out_que.task_done()

        while True:
            if self.req_manager.has_requests():
                self.req_manager.step()

            if not self.scheduler.has_unfinished():
                await asyncio.sleep(0.01)
                continue

            # prefill
            if self.scheduler.has_waiting():
                await __step(True)

            # decoding
            if self.scheduler.has_running():
                await __step(False)

    async def async_loop(self):
        device_manager = get_device_manager()
        with device_manager.context(self.device_context), torch.cuda.stream(
                self.stream):
            await self._async_loop()

    def create_instance(self, cuda_stream_id=0):
        """Create a pytorch engine instance.

        Args:
            cuda_stream_id(int): identity of a cuda stream
        Returns:
            EngineInstance: an instance of pytorch engine
        """
        from .engine_instance import EngineInstance
        return EngineInstance(self)

    async def async_batched_infer(
            self,
            session_ids: List[int],
            token_ids: List[List[int]] = None,
            gen_config: EngineGenerationConfig = None,
            adapter_names: List[str] = None,
            keep_cache: bool = False,
            input_embeddings: List[InputEmbeddingType] = None,
            input_embedding_ranges: List[InputEmbeddingRangeType] = None):
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
        return await self.engine_instance.async_batched_infer(
            session_ids=session_ids,
            token_ids=token_ids,
            gen_config=gen_config,
            adapter_names=adapter_names,
            input_embeddings=input_embeddings,
            input_embedding_ranges=input_embedding_ranges,
            keep_cache=keep_cache)

    def batched_infer(
            self,
            session_ids: List[int],
            token_ids: List[List[int]] = None,
            gen_config: EngineGenerationConfig = None,
            adapter_names: List[str] = None,
            keep_cache: bool = False,
            input_embeddings: List[InputEmbeddingType] = None,
            input_embedding_ranges: List[InputEmbeddingRangeType] = None):
        """batched infer."""
        return self.engine_instance.batched_infer(
            session_ids=session_ids,
            token_ids=token_ids,
            gen_config=gen_config,
            adapter_names=adapter_names,
            input_embeddings=input_embeddings,
            input_embedding_ranges=input_embedding_ranges,
            keep_cache=keep_cache)

    async def async_add_session(self, session_id: int):
        """Add new session."""
        return await self.engine_instance._async_try_add_session(session_id)

    def add_session(self, session_id: int):
        """Add new session."""
        return self.engine_instance._try_add_session(session_id)

    async def async_cancel(self, session_id: int):
        """Stop the given session."""
        return await self.engine_instance.async_cancel(session_id)

    def cancel(self, session_id: int):
        """Add new session."""
        return self.engine_instance.cancel(session_id)

    async def async_end(self, session_id: int):
        """End the given session."""
        return await self.engine_instance.async_end(session_id)

    def end(self, session_id: int):
        """Add new session."""
        return self.engine_instance.end(session_id)

    def decode(self,
               input_ids,
               input_embeddings: List[InputEmbeddingType] = None,
               input_embedding_ranges: List[InputEmbeddingRangeType] = None,
               steps: List[int] = None,
               sequence_start: bool = True,
               sequence_end: bool = True,
               adapter_names: List[str] = None):
        """Perform context decode on input tokens.

        Args:
            input_ids (List[List[int]] | List[np.ndaray]): the batch of input
                 token ids
            steps (List[int]): the offset of the k/v cache
            input_embeddings (List[List[Union[torch.Tensor, np.ndarray]]]):
                embeddings features
            input_embedding_ranges: (List[List[Tuple[int, int]]]):
                the begin/end offsets of input_embeddings to input_ids
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
            adapter_names (List[str]): The name of the adapters.
        """
        return self.engine_instance.decode(
            input_ids,
            input_embeddings=input_embeddings,
            input_embedding_ranges=input_embedding_ranges,
            steps=steps,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            adapter_names=adapter_names)
