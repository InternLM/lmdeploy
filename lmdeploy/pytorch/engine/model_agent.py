# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import functools
from contextlib import contextmanager
from typing import Any, Dict

import torch
import torch.distributed as dist

from lmdeploy.utils import get_logger, logging_timer

from ..backends import get_backend
from ..config import BackendConfig, CacheConfig, ModelConfig
from ..devices import DeviceContext, get_device_manager
from ..distributed import DistContext, get_dist_manager
from ..model_inputs import ModelInputs
from ..models.patch import add_adapters, build_patched_model, update_custom_module_map
from ..utils import get_gpu_memory
from ..weight_loader.model_weight_loader import load_model_weights
from .cache_engine import CacheEngine
from .logits_process import FusedLogitsProcessor, SamplingInputs

logger = get_logger('lmdeploy')


def msg_with_rank(rank: int, msg: str):
    """return message with rank."""
    return f'rank[{rank}] - {msg}'


def cache_swapping(cache_engine: CacheEngine, swap_in_map: dict, swap_out_map: dict):
    """perform cache swapping."""
    issued_cache_op = False
    if len(swap_in_map) > 0:
        cache_engine.swap_in(swap_in_map)
        issued_cache_op = True
    if len(swap_out_map) > 0:
        cache_engine.swap_out(swap_out_map)
        issued_cache_op = True

    if issued_cache_op:
        cache_engine.events.wait()


@torch.inference_mode()
def model_forward(
    model: torch.nn.Module,
    inputs: ModelInputs,
    cache_engine: CacheEngine,
    world_size: int = 1,
    stream: torch.cuda.Stream = None,
):
    """perform model forward."""
    stream = stream or torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        # forward
        ctx_mgr = model.ctx_mgr
        context = ctx_mgr.build_context(
            inputs=inputs,
            model_config=cache_engine.model_config,
            world_size=world_size,
            kv_caches=cache_engine.gpu_cache,
            kv_quant_policy=cache_engine.cache_config.quant_policy,
        )
        with ctx_mgr.context(context):
            model_metas = None
            model_metas = model.update_model_metas(
                past_key_values=cache_engine.gpu_cache,
                context=context,
            )
            input_dict = model.prepare_inputs_for_generation(
                past_key_values=cache_engine.gpu_cache,
                context=context,
            )
            output = model(**input_dict)
    return dict(hidden_states=output, model_metas=model_metas)


def _batch_stopping_criteria(token_ids: torch.Tensor, stop_words: torch.Tensor, num_appendable_ids: torch.Tensor):
    """batched stopping criteria."""
    num_appendable_ids = num_appendable_ids - 1
    # one more step to cache last token(stop word)
    stopped = num_appendable_ids < 0
    if stop_words is not None:
        sw_stopped = (token_ids[:, None] == stop_words).any(1)
        one_ids = torch.clamp_max(num_appendable_ids, 0)
        num_appendable_ids = torch.where(sw_stopped, one_ids, num_appendable_ids)
    return stopped, num_appendable_ids


SwapMap = Dict[int, int]


class AutoModelAgent:
    """Base model agent."""

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        tokenizer: Any,
        dist_ctx: DistContext,
        device_ctx: DeviceContext,
    ):
        self.model_config = model_config
        self.cache_config = cache_config
        self.tokenizer = tokenizer

        self._in_que = None
        self._out_que = None
        self._background_task = None

        self.stream = torch.cuda.Stream()
        self.out_stream = torch.cuda.Stream()

        self.dist_ctx = dist_ctx
        self.device_ctx = device_ctx

    @contextmanager
    def all_context(self):
        device_mgr = get_device_manager()
        dist_mgr = get_dist_manager()
        with device_mgr.context(self.device_ctx), dist_mgr.context(self.dist_ctx):
            yield

    async def async_forward(self, inputs: ModelInputs, swap_in_map: SwapMap, swap_out_map: SwapMap):
        """model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
            swap_in_map (SwapMap): Cache maps to swap in.
            swap_out_map (SwapMap): Cache maps to swap out.
        """
        raise NotImplementedError('Not implemented.')

    def get_logits(self, hidden_states: torch.Tensor):
        """get logits of model output."""
        raise NotImplementedError('Not implemented.')

    def get_input_processor(self):
        """get input processor."""
        raise NotImplementedError('Not implemented.')

    def build_model(self):
        """build model."""
        raise NotImplementedError('Not implemented.')

    def build_graph_runner(self):
        """build graph runner."""
        raise NotImplementedError('Not Implemented.')

    def build_cache_engine(self):
        """build cache engine."""
        raise NotImplementedError('Not Implemented.')

    def release(self):
        """release."""
        raise NotImplementedError('Not Implemented.')

    def set_cache_config(self, cache_config: CacheConfig):
        """set all cache config."""
        self.cache_config = cache_config

    def set_model_config(self, model_config: ModelConfig):
        """set model config."""
        self.model_config = model_config

    def get_free_mem(self):
        """gather available memory."""
        with self.all_context():
            torch.cuda.empty_cache()
            gpu_mem_physical_free, _ = get_gpu_memory()
            return gpu_mem_physical_free

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
                return await self.async_forward(inputs, swap_in_map=dict(), swap_out_map=dict())
            else:
                swap_done = True
                return await self.async_forward(inputs, swap_in_map=swap_in_map, swap_out_map=swap_out_map)

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
        logits = self.get_logits(hidden_states)
        ret['logits'] = logits
        return ret

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
        logits_processor = FusedLogitsProcessor(sampling_inputs, ignore_eos, self.tokenizer)
        logits = await logits_processor(all_ids, guided_input_ids, split_logits)
        next_token_ids = logits_processor.sampling(logits)

        return next_token_ids

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
        non_blocking = True
        inputs = inputs.to_device('cuda', non_blocking=non_blocking)
        is_decoding = inputs.is_decoding
        if all_ids is not None:
            all_ids = all_ids.cuda(non_blocking=non_blocking)
        if guided_input_ids is not None:
            guided_input_ids = guided_input_ids.cuda(non_blocking=non_blocking)
        sampling_inputs = sampling_inputs.to_device('cuda', non_blocking=non_blocking)
        num_appendable_ids = num_appendable_ids.cuda(non_blocking=non_blocking)
        num_ignore_eos = num_ignore_eos.cuda(non_blocking=non_blocking)

        self.stream.synchronize()

        # dist tools
        dist_ctx = get_dist_manager().current_context()
        rank = dist_ctx.rank
        tp = dist_ctx.tp

        for idx in range(loop_count):
            # inference
            output = await self._async_model_forward(inputs,
                                                     swap_in_map=swap_in_map,
                                                     swap_out_map=swap_out_map,
                                                     return_logits=return_logits)
            logits = output['logits']
            logits = logits[0]  # [bs, seq, prob] -> [seq, prob]

            if rank % tp == 0:
                # sampling
                next_token_ids = await self.async_sampling_logits(logits, all_ids, guided_input_ids, sampling_inputs,
                                                                  inputs, num_ignore_eos > 0)
                num_ignore_eos = num_ignore_eos - 1

                # stopping criteria
                stopped, num_appendable_ids = _batch_stopping_criteria(next_token_ids, sampling_inputs.stop_words,
                                                                       num_appendable_ids)
            else:
                next_token_ids = torch.empty_like(num_ignore_eos)
                stopped = None

            if tp > 1 and idx < loop_count - 1:
                tp_gpu_group = dist_ctx.tp_gpu_group
                dist.broadcast(next_token_ids, src=rank // tp * tp, group=tp_gpu_group)

            # send output
            model_metas = output.get('model_metas')
            if rank % tp == 0:
                event = torch.cuda.Event()
                event.record()
                output = dict(next_token_ids=next_token_ids,
                              logits=logits if return_logits else None,
                              stopped=stopped,
                              model_metas=model_metas,
                              event=event)
                output_que.put_nowait(output)

            # update for next loop
            if is_decoding and idx < loop_count - 1:
                swap_in_map = dict()
                swap_out_map = dict()
                inputs.model_metas = model_metas
                __update_inputs(next_token_ids)

    @torch.inference_mode()
    async def _async_loop_background(self, forward_event: asyncio.Event = None):
        """async loop background."""
        with self.all_context(), torch.cuda.stream(self.stream):
            while True:
                forward_inputs = await self._in_que.get()

                if forward_event is not None:
                    forward_event.clear()
                await self._async_step_background(
                    **forward_inputs,
                    output_que=self._out_que,
                )
                if forward_event is not None:
                    forward_event.set()

    @staticmethod
    def _on_finish_callback(task: asyncio.Task, ptask: asyncio.Task) -> None:
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
            if not task.done():
                task.cancel()
            if not ptask.done():
                ptask.cancel()

    def start(self, forward_event: asyncio.Event = None):
        """start event loop."""
        event_loop = asyncio.get_event_loop()
        self._in_que = asyncio.Queue()
        self._out_que = asyncio.Queue()
        self._background_task = event_loop.create_task(self._async_loop_background(forward_event),
                                                       name='ModelAgentLoop')
        done_callback = functools.partial(self._on_finish_callback, ptask=asyncio.current_task())
        self._background_task.add_done_callback(done_callback)

    def stop(self):
        """stop task."""
        if self._background_task is not None:
            if not self._background_task.done():
                self._background_task.cancel()

    def set_forward_inputs(self, inputs):
        """set forward inputs."""
        assert self._in_que is not None, ('Please start backendground task before forward.')
        self._in_que.put_nowait(inputs)

    async def get_output_async(self):
        """async get output."""
        assert self._out_que is not None, ('Please start backendground task before forward.')
        out = await self._out_que.get()
        event = out.pop('event')
        while not event.query():
            await asyncio.sleep(0.001)
        with torch.cuda.stream(self.out_stream):
            out['next_token_ids'] = out['next_token_ids'].cpu()
            out['stopped'] = out['stopped'].cpu()
            if out['logits'] is not None:
                out['logits'] = out['logits'].cpu()
        return out


class BaseModelAgent(AutoModelAgent):
    """Base model agent.

    load model on local gpu

    Args:
        model_path (str): The hugging face model path.
        model_config (ModelConfig): The config of the model.
        cache_config (CacheConfig): The config of the cache info.
        trust_remote_code (bool): Trust remote code
    """

    def __init__(self,
                 model_path: str,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 backend_config: BackendConfig,
                 tokenizer: Any,
                 dist_ctx: DistContext,
                 device_ctx: DeviceContext,
                 adapters: Dict[str, str] = None):
        super().__init__(
            model_config=model_config,
            cache_config=cache_config,
            tokenizer=tokenizer,
            dist_ctx=dist_ctx,
            device_ctx=device_ctx,
        )
        device = 'cuda'
        self.backend_config = backend_config
        rank = dist_ctx.rank

        self.model_path = model_path
        self.adapters = adapters
        self.device = device
        self.rank = rank

        tp_rank = dist_ctx.tp_rank
        tp = dist_ctx.tp
        world_size = dist_ctx.world_size
        self.tp = tp
        self.world_size = world_size
        self.tp_rank = tp_rank

        self.patched_model = None
        self.cache_engine = None

    def _build_model(self):
        """build patched model."""
        model_path = self.model_path
        adapters = self.adapters
        device = self.device
        rank = self.rank
        custom_module_map = self.model_config.custom_module_map
        if custom_module_map is not None:
            update_custom_module_map(custom_module_map)
        logger.debug(msg_with_rank(rank, 'build model.'))
        patched_model = build_patched_model(self.model_config, device=device)
        logger.debug(msg_with_rank(rank, 'loading weights.'))
        load_model_weights(patched_model, model_path, device=device)
        if adapters is not None:
            logger.debug(msg_with_rank(rank, 'loading adapters.'))
            add_adapters(patched_model, adapters, dtype=self.model_config.dtype, device=device)
        self.patched_model = patched_model

    def build_model(self):
        """build model api."""
        with self.all_context():
            self._build_model()

    def build_graph_runner(self):
        """build graph runner."""
        with self.all_context():
            backend = get_backend()
            self.patched_model = backend.build_graph_runner(self.patched_model,
                                                            model_config=self.model_config,
                                                            cache_config=self.cache_config,
                                                            backend_config=self.backend_config,
                                                            device=self.device)

    def build_cache_engine(self):
        """build cache engine."""
        with self.all_context():
            self.cache_engine = CacheEngine(self.cache_config, self.model_config, self.tp_rank, world_size=self.tp)

    def _forward_impl(self, inputs: ModelInputs, swap_in_map: SwapMap, swap_out_map: SwapMap):
        cache_swapping(self.cache_engine, swap_in_map=swap_in_map, swap_out_map=swap_out_map)
        output = model_forward(
            self.patched_model,
            inputs,
            self.cache_engine,
            world_size=1,
            stream=self.stream,
        )
        return output

    async def async_forward(self, inputs: ModelInputs, swap_in_map: SwapMap, swap_out_map: SwapMap):
        """model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
            swap_in_map (SwapMap): Cache maps to swap in.
            swap_out_map (SwapMap): Cache maps to swap out.
        """
        output = self._forward_impl(inputs, swap_in_map=swap_in_map, swap_out_map=swap_out_map)
        await asyncio.sleep(0)
        return output

    def get_logits(self, hidden_states: torch.Tensor):
        """get logits of model output."""
        return self.patched_model.get_logits(hidden_states)

    def get_input_processor(self):
        """get input processor.."""
        return self.patched_model.get_input_processor()

    def reset_graph_runner(self):
        """reset graph runner to prevent tp hanging."""
        if hasattr(self.patched_model, 'reset'):
            self.patched_model.reset()

    def release(self):
        """release."""
        self.reset_graph_runner()
        self.patched_model = None
        self.cache_engine = None
        torch.cuda.empty_cache()


def build_model_agent(model_path: str,
                      model_config: ModelConfig,
                      cache_config: CacheConfig,
                      backend_config: BackendConfig,
                      tokenizer: Any,
                      dist_ctx: DistContext = None,
                      device_ctx: DeviceContext = None,
                      adapters: Dict[str, str] = None):
    """create model agent.

    Args:
        model_path (str): the path of the input model
        cache_config (CacheConfig): config of kv cache
        backend_config (BackendConfig): config of backend devices
        trust_remote_code (bool): To use the remote modeling code or not
        adapters (Dict): lora adapters
        tp (int): the number of devices to be used in tensor parallelism
        dtype (str): the data type of model weights and activations
        custom_module_map (str): customized nn module map
    """
    if device_ctx is None:
        device_mgr = get_device_manager()
        device_ctx = device_mgr.current_context()
    if dist_ctx is None:
        dist_mgr = get_dist_manager()
        dist_ctx = dist_mgr.current_context()

    model_agent = BaseModelAgent(
        model_path,
        model_config=model_config,
        cache_config=cache_config,
        backend_config=backend_config,
        tokenizer=tokenizer,
        adapters=adapters,
        dist_ctx=dist_ctx,
        device_ctx=device_ctx,
    )
    return model_agent
