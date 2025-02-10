# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Any, Dict

import torch
import torch.distributed as dist

from lmdeploy.utils import get_logger, logging_timer

from ..backends import get_backend
from ..config import BackendConfig, CacheConfig, ModelConfig
from ..distributed import get_dist_manager, get_world_rank
from ..model_inputs import ModelInputs
from ..models.patch import add_adapters, build_patched_model, update_custom_module_map
from ..utils import get_gpu_memory
from ..weight_loader.model_weight_loader import load_model_weights
from .cache_engine import CacheEngine
from .dist_proc_manager import broadcast_inputs
from .logits_process import FusedLogitsProcessor, SamplingInputs

logger = get_logger('lmdeploy')


def msg_with_rank(rank: int, msg: str):
    """return message with rank."""
    return f'rank[{rank}] - {msg}'


def _update_cache_config(model_config: ModelConfig,
                         cache_config: CacheConfig,
                         gpu_id: int = 0,
                         host_mem_size: int = 1 * (1 << 30),
                         world_size: int = 1):
    """Update the gpu mem and cpu mem according to model info.

    Args:
        model_config (ModelConfig): The config of the model.
        cache_config (CacheConfig): The config of the cache info.
        gpu_id (int): The GPU id to use.
    """

    def __get_runtime_size(num_free_gpu_mem: int, cache_block_size: int, vocal_size: int):
        """find best prefill num."""
        cache_max_entry_count = cache_config.cache_max_entry_count
        max_prefill_token_num = cache_config.max_prefill_token_num
        runtime_cache_size = 0
        while max_prefill_token_num > 0:
            # lm_head output(2) + to float(4) + estimated misc(1) = 7
            runtime_cache_size = int(max_prefill_token_num * vocal_size * 7)
            num_available = (num_free_gpu_mem - runtime_cache_size) * cache_max_entry_count
            if int(num_available) // cache_block_size >= 16:
                break
            max_prefill_token_num = max_prefill_token_num // 2
        return runtime_cache_size, max_prefill_token_num

    def __get_free_gpu_mem_size(cache_block_size: int):
        """get free gpu memory size."""
        torch.cuda.empty_cache()
        gpu_mem_physical_free, _ = get_gpu_memory(gpu_id)
        logger.debug(f'device<{gpu_id}> free gpu memory:'
                     f' {gpu_mem_physical_free>>20} mb')
        vocal_size = model_config.vocab_size

        runtime_cache_size, max_prefill_token_num = __get_runtime_size(gpu_mem_physical_free, cache_block_size,
                                                                       vocal_size)
        if cache_config.max_prefill_token_num != max_prefill_token_num:
            if max_prefill_token_num <= 0:
                raise RuntimeError('No enough gpu memory for runtime.')
            cache_config.max_prefill_token_num = max_prefill_token_num
            logger.warning(f'device<{gpu_id}> No enough memory. '
                           'update max_prefill_token_num='
                           f'{max_prefill_token_num}')
        gpu_mem_physical_free -= runtime_cache_size
        logger.debug('estimated max runtime memory:'
                     f' {runtime_cache_size>>20} mb')
        return gpu_mem_physical_free * cache_config.cache_max_entry_count

    def __adjust_block_size():
        """adjust block_size."""
        # TODO: support kernel with both large head dim and large block size.
        if model_config.k_head_dim >= 512 and cache_config.block_size > 32:
            cache_config.block_size = 32
            _, rank = get_world_rank()
            if rank == 0:
                logger.warning(f'Update `block_size={cache_config.block_size}`'
                               f' for large `head_dim={model_config.k_head_dim}`.')

    __adjust_block_size()

    cache_block_size = CacheEngine.get_cache_block_size(cache_config.block_size, model_config, world_size,
                                                        cache_config.quant_policy)
    gpu_mem = __get_free_gpu_mem_size(cache_block_size)
    cpu_mem = host_mem_size
    if cache_config.num_cpu_blocks == 0:
        cache_config.num_cpu_blocks = int(cpu_mem / cache_block_size)
        if cache_config.num_cpu_blocks <= 0:
            raise RuntimeError('No enough host memory for kv cache.')
    if cache_config.num_gpu_blocks == 0:
        cache_config.num_gpu_blocks = int(gpu_mem / cache_block_size)
        if cache_config.num_gpu_blocks <= 0:
            raise RuntimeError('No enough gpu memory for kv cache.')
    cache_config.window_size = model_config.sliding_window

    logger.debug('block num: {}'.format(cache_config.num_gpu_blocks))


def _broadcast_config(cache_config: CacheConfig):
    """broadcast cache config, use minimum cache."""
    # get dist info
    dist_ctx = get_dist_manager().current_context()
    world_group = dist_ctx.world_cpu_group
    world_size = dist_ctx.world_size
    rank = dist_ctx.rank

    if rank == 0:
        gathered_configs = [None] * world_size
        dist.gather_object(cache_config, gathered_configs, group=world_group)

        # set minimal blocks
        num_gpu_blocks_list = [config.num_gpu_blocks for config in gathered_configs]
        num_cpu_blocks_list = [config.num_cpu_blocks for config in gathered_configs]
        min_num_gpu_blocks = min(num_gpu_blocks_list)
        min_num_cpu_blocks = min(num_cpu_blocks_list)
        cache_config.num_cpu_blocks = min_num_cpu_blocks
        cache_config.num_gpu_blocks = min_num_gpu_blocks
        config_list = [cache_config]
    else:
        gathered_configs = None
        dist.gather_object(cache_config, gathered_configs, group=world_group)
        config_list = [None]

    # broadcast
    dist.broadcast_object_list(config_list, group=world_group)
    return config_list[0]


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

    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig, tokenizer: Any):
        self.model_config = model_config
        self.cache_config = cache_config
        self.tokenizer = tokenizer

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
        inputs = inputs.to_device('cuda')
        is_decoding = inputs.is_decoding
        if all_ids is not None:
            all_ids = all_ids.cuda()
        if guided_input_ids is not None:
            guided_input_ids = guided_input_ids.cuda()
        sampling_inputs = sampling_inputs.to_device('cuda')
        num_appendable_ids = num_appendable_ids.cuda()
        num_ignore_eos = num_ignore_eos.cuda()

        # dist tools
        dist_ctx = get_dist_manager().current_context()

        for idx in range(loop_count):

            # broadcast inputs
            if dist_ctx.tp > 1:
                tp_cpu_group = dist_ctx.tp_cpu_group
                (inputs, swap_in_map, swap_out_map) = broadcast_inputs(0, [inputs, swap_in_map, swap_out_map],
                                                                       tp_cpu_group)

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
            stopped, num_appendable_ids = _batch_stopping_criteria(next_token_ids, sampling_inputs.stop_words,
                                                                   num_appendable_ids)

            # send output
            model_metas = output.get('model_metas')
            event = torch.cuda.Event()
            event.record()
            output = dict(next_token_ids=next_token_ids,
                          logits=logits,
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
                 adapters: Dict[str, str] = None):
        super().__init__(model_config=model_config, cache_config=cache_config, tokenizer=tokenizer)
        device = 'cuda'
        self.backend_config = backend_config
        dist_ctx = get_dist_manager().current_context()
        rank = dist_ctx.rank

        self.patched_model = self._build_model(model_path, adapters, device=device, rank=rank)

        local_rank = dist_ctx.local_rank
        tp = dist_ctx.tp
        world_size = dist_ctx.world_size
        _update_cache_config(model_config, cache_config, gpu_id=local_rank, world_size=tp)
        if world_size > 1:
            # broadcast cache config
            _broadcast_config(cache_config)

        backend = get_backend()
        self.patched_model = backend.build_graph_runner(self.patched_model,
                                                        model_config=model_config,
                                                        cache_config=cache_config,
                                                        backend_config=backend_config,
                                                        device=device)

        self.cache_engine = CacheEngine(cache_config, model_config, local_rank, world_size=tp)

        self.stream = torch.cuda.Stream()

    def _build_model(
        self,
        model_path: str,
        adapters: Dict[str, str] = None,
        device: torch.device = 'cuda',
        rank: int = 0,
    ):
        """build patched model."""
        custom_module_map = self.model_config.custom_module_map
        if custom_module_map is not None:
            update_custom_module_map(custom_module_map)
        logger.info(msg_with_rank(rank, 'build model.'))
        patched_model = build_patched_model(self.model_config, device=device)
        logger.info(msg_with_rank(rank, 'loading weights.'))
        load_model_weights(patched_model, model_path, device=device)
        if adapters is not None:
            logger.info(msg_with_rank(rank, 'loading adapters.'))
            add_adapters(patched_model, adapters, dtype=self.model_config.dtype, device=device)
        return patched_model

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
        self.patched_model.reset()


def build_model_agent(model_path: str,
                      cache_config: CacheConfig,
                      backend_config: BackendConfig,
                      tokenizer: Any,
                      trust_remote_code: bool,
                      adapters: Dict[str, str] = None,
                      tp: int = 1,
                      dtype: str = 'auto',
                      custom_module_map: str = None):
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
    model_config = ModelConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code, dtype=dtype, tp=tp)
    model_config.custom_module_map = custom_module_map

    model_agent = BaseModelAgent(model_path,
                                 model_config=model_config,
                                 cache_config=cache_config,
                                 backend_config=backend_config,
                                 tokenizer=tokenizer,
                                 adapters=adapters)
    return model_agent
