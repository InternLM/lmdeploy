# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
from torch import multiprocessing as mp
from transformers import AutoModelForCausalLM

from lmdeploy.pytorch.accel import LoadNoInit
from lmdeploy.utils import get_logger

from ..adapter.adapter import (AdapterWeightMap, get_indexed_lora_linears,
                               get_max_lora_weight_size, update_lora_linears)
from ..config import CacheConfig, ModelConfig
from ..models import patch
from ..utils import get_gpu_memory
from .cache_engine import CacheEngine

logger = get_logger('lmdeploy')

_PATCH_ARG_NAMES = ['context', 'use_origin']


def _infer_block_size(model: torch.nn.Module,
                      model_config: ModelConfig,
                      cache_config: CacheConfig,
                      world_size: int = 1):
    """infer block size."""
    max_weight_dim = get_max_lora_weight_size(model)
    if max_weight_dim == 0:
        return cache_config.block_size

    per_token_size = model_config.get_head_size(
    ) * model_config.num_key_value_heads // world_size
    block_size = 1
    while block_size * per_token_size < max_weight_dim:
        block_size *= 2
    return block_size * world_size


def _update_cache_config(model_config: ModelConfig,
                         cache_config: CacheConfig,
                         gpu_id: int = 0,
                         host_mem_size: int = 4 * (1 << 30),
                         world_size: int = 1):
    """Update the gpu mem and cpu mem according to model info.

    Args:
        model_config (ModelConfig): The config of the model.
        cache_config (CacheConfig): The config of the cache info.
        gpu_id (int): The GPU id to use.
    """

    def __get_runtime_size(num_free_gpu_mem: int, cache_block_size: int,
                           vocal_size: int):
        """find best prefill num."""
        cache_max_entry_count = cache_config.cache_max_entry_count
        max_prefill_token_num = cache_config.max_prefill_token_num
        runtime_cache_size = 0
        while max_prefill_token_num > 0:
            # lm_head output(2) + to float(4) + estimated misc(1) = 7
            runtime_cache_size = int(max_prefill_token_num * vocal_size * 7)
            num_available = (num_free_gpu_mem -
                             runtime_cache_size) * cache_max_entry_count
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

        runtime_cache_size, max_prefill_token_num = __get_runtime_size(
            gpu_mem_physical_free, cache_block_size, vocal_size)
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

    cache_block_size = CacheEngine.get_cache_block_size(
        cache_config.block_size, model_config, world_size)
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


@dataclass
class ModelInputs:
    """Input of the model."""
    input_ids: torch.LongTensor
    seq_length: torch.LongTensor
    history_lengths: torch.LongTensor
    block_offsets: torch.LongTensor
    max_q_seq_length: int
    max_history_length: int
    is_decoding: bool
    num_ignored_history: torch.LongTensor
    local_adapter_ids: torch.LongTensor = None
    global_adapter_ids: torch.LongTensor = None
    adapter_offsets: torch.LongTensor = None
    max_rank: int = 0
    meta: Any = None
    input_embeddings: List[torch.Tensor] = None
    input_embedding_ranges: torch.LongTensor = None
    token_type_ids: torch.Tensor = None
    position_ids: torch.LongTensor = None
    history_position_lengths: torch.LongTensor = None

    def update(self, input_ids: torch.LongTensor):
        """update input ids."""
        assert self.is_decoding
        self.history_lengths = self.history_lengths + 1
        self.max_history_length = self.max_history_length + 1
        if self.history_position_lengths is not None:
            self.history_position_lengths = self.history_position_lengths + 1

        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        self.input_ids = input_ids
        return self

    def split(self, split_size: int, block_size: int):
        """split inputs."""
        assert len(
            self.seq_length) == 1, ('Can not perform split on batched input.')
        assert split_size % block_size == 0, (
            'split_size should be multi of block_size.')

        input_ids = self.input_ids
        if input_ids.numel() < split_size:
            return self

        num_blocks = split_size // block_size
        overlap = (self.history_lengths[0] % block_size != 0)
        max_seq_len = self.seq_length[0].item()
        ret = []
        block_start = 0
        for i in range(0, max_seq_len, split_size):
            start = i
            end = min(max_seq_len, i + split_size)
            block_end = block_start + num_blocks
            if overlap:
                block_end += 1

            local_adapter_ids = self.local_adapter_ids
            if local_adapter_ids is not None:
                local_adapter_ids = local_adapter_ids[:, start:end]

            block_offsets = self.block_offsets[:, :block_end]
            inp = ModelInputs(
                input_ids=self.input_ids[:, start:end],
                seq_length=input_ids.new_tensor([end - start]),
                block_offsets=block_offsets,
                history_lengths=self.history_lengths + start,
                max_q_seq_length=input_ids.new_tensor(end - start),
                max_history_length=self.max_history_length + start,
                is_decoding=self.is_decoding,
                num_ignored_history=self.num_ignored_history,
                local_adapter_ids=local_adapter_ids,
                global_adapter_ids=self.global_adapter_ids,
                adapter_offsets=self.adapter_offsets,
                max_rank=self.max_rank,
                meta=self.meta,
            )
            ret.append(inp)
            block_start += num_blocks

        return ret

    def to_device(self, device: str):
        """to device."""
        input_dict = asdict(self)
        out_dict = dict()
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            elif isinstance(v, List) and len(v) > 0 and isinstance(
                    v[0], torch.Tensor):
                v = [_.to(device) for _ in v]
            out_dict[k] = v

        return ModelInputs(**out_dict)


@dataclass
class StepContext:
    """context of Model.

    patched model might need extra information to perform inference. This
    dataclass provide these infos and tools.
    """
    inputs: ModelInputs
    block_offsets: torch.LongTensor
    position_ids: torch.LongTensor
    position_ids_1d: torch.LongTensor
    q_start_loc: torch.LongTensor
    attention_mask: torch.LongTensor
    history_lengths: torch.LongTensor
    q_seq_length: torch.LongTensor
    kv_seq_length: torch.LongTensor
    max_q_seq_length: int
    max_kv_seq_length: int
    kv_caches: List
    is_decoding: bool
    world_size: int = 1
    json_config: Dict = None
    local_adapter_ids: torch.LongTensor = None
    global_adapter_ids: torch.LongTensor = None
    adapter_offsets: torch.LongTensor = None
    max_rank: int = 0
    history_position_lengths: torch.LongTensor = None

    _outputs: Dict = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        inputs: ModelInputs,
        world_size: int = 1,
        device: str = 'cuda',
        json_config: dict = None,
        kv_caches: List = None,
        cache_config: CacheConfig = None,
    ):
        """build step context.

        Args:
            inputs (ModelInputs): packaged model inputs.
            world_size (int): The distribution world size.
            device (str): The device of the tensors.
        """
        q_seq_length = inputs.seq_length
        max_q_seq_length = inputs.max_q_seq_length
        history_position_lengths = inputs.history_lengths
        if inputs.history_position_lengths is not None:
            history_position_lengths = inputs.history_position_lengths

        batch_size = len(q_seq_length)
        device = q_seq_length.device

        # q_start_loc and kv_seq_length
        if inputs.is_decoding:
            q_start_loc = torch.arange(0, batch_size, device=device)
            attention_mask = torch.ones_like(q_seq_length)[:, None]
            position_ids = history_position_lengths.unsqueeze(-1)
        else:
            q_start_loc = q_seq_length.cumsum(0) - q_seq_length
            mask_range = torch.arange(max_q_seq_length, device=device)[None, :]
            attention_mask = (mask_range < q_seq_length[:, None]).long()
            if inputs.position_ids is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids += history_position_lengths.unsqueeze(-1)
            else:
                position_ids = inputs.position_ids

        # position ids 1d
        position_ids_1d = cls.get_position_ids_1d(position_ids, q_seq_length,
                                                  device)

        # seq_len + history_length
        kv_seq_length = q_seq_length + inputs.history_lengths
        max_kv_seq_length = max_q_seq_length + inputs.max_history_length

        window_size = getattr(cache_config, 'window_size', 0)
        if window_size > 0:
            kv_seq_length -= inputs.num_ignored_history

        ret = StepContext(
            inputs=inputs,
            block_offsets=inputs.block_offsets,
            position_ids=position_ids,
            position_ids_1d=position_ids_1d,
            attention_mask=attention_mask,
            q_start_loc=q_start_loc,
            history_lengths=inputs.history_lengths,
            history_position_lengths=inputs.history_position_lengths,
            q_seq_length=inputs.seq_length,
            kv_seq_length=kv_seq_length,
            max_q_seq_length=max_q_seq_length,
            max_kv_seq_length=max_kv_seq_length,
            kv_caches=kv_caches,
            is_decoding=inputs.is_decoding,
            world_size=world_size,
            json_config=json_config,
            local_adapter_ids=inputs.local_adapter_ids,
            global_adapter_ids=inputs.global_adapter_ids,
            adapter_offsets=inputs.adapter_offsets,
            max_rank=inputs.max_rank)
        return ret

    @classmethod
    def get_position_ids_1d(cls,
                            position_ids: torch.LongTensor,
                            seq_length: torch.LongTensor,
                            device: str = 'cuda'):
        """get 1d position_ids."""
        if position_ids.size(0) == 1 or position_ids.size(1) == 1:
            position_ids_1d = position_ids.flatten()
        else:
            position_ids_1d = [
                ids[:l] for ids, l in zip(position_ids.cpu(), seq_length.cpu())
            ]
            position_ids_1d = torch.cat(position_ids_1d).to(device)
        return position_ids_1d

    def set_output(self, key, value):
        """set output."""
        self._outputs[key] = value

    def get_output(self, key):
        """get output."""
        if key in self._outputs:
            return self._outputs[key]
        return None


def cache_swapping(cache_engine: CacheEngine, swap_in_map: dict,
                   swap_out_map: dict):
    """perform cache swapping."""
    issued_cache_op = False
    if len(swap_in_map) > 0:
        cache_engine.swap_in(swap_in_map)
        issued_cache_op = True
    if len(swap_out_map) > 0:
        cache_engine.swap_out(swap_out_map)
        issued_cache_op = True

    if issued_cache_op:
        cache_events = cache_engine.events
        for event in cache_events:
            event.wait()


def model_forward(patched_model: torch.nn.Module,
                  inputs: ModelInputs,
                  cache_engine: CacheEngine,
                  json_config: dict = None,
                  world_size: int = 1,
                  stream: torch.cuda.Stream = None):
    """perform model forward."""

    stream = stream or torch.cuda.current_stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        # forward

        inputs = inputs.to_device('cuda')

        context = StepContext.new(
            inputs=inputs,
            world_size=world_size,
            json_config=json_config,
            kv_caches=cache_engine.gpu_cache,
            cache_config=cache_engine.cache_config,
        )
        output = patched_model.patched_forward(
            input_ids=inputs.input_ids,
            position_ids=context.position_ids,
            attention_mask=context.attention_mask,
            past_key_values=cache_engine.gpu_cache,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            use_origin=False,
            context=context)
    return dict(logits=output['logits'], custom_outputs=context._outputs)


def _load_adapters(hf_model: torch.nn.Module,
                   adapters: Dict[str, str],
                   device_map: str = 'cpu'):
    """load adapters."""
    if not adapters:
        return
    for name, path in adapters.items():
        logger.info(f'load adapter <{name}> from "{path}".')
        hf_model.load_adapter(path, name, device_map=device_map)


def _add_adapters(hf_model: torch.nn.Module, adapters: Dict[str, str]):
    """add adapters."""
    if not adapters:
        return
    from peft import PeftConfig, inject_adapter_in_model
    for name, path in adapters.items():
        config = PeftConfig.from_pretrained(path)
        inject_adapter_in_model(config, model=hf_model, adapter_name=name)


def _unparam_lora_weight(model: torch.nn.Module):
    """unparam lora weight.

    We don't want to move weight of lora to gpu.
    """
    from peft.tuners.lora import Linear as LoRALinear

    def _tensorize_weight(linear):
        """tensorize weight."""
        w = linear.weight
        del linear.weight
        linear.weight = w.data

    for _, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            lora_A = mod.lora_A
            lora_B = mod.lora_B
            for linear in lora_A.values():
                _tensorize_weight(linear)
            for linear in lora_B.values():
                _tensorize_weight(linear)


SwapMap = Dict[int, int]


class AutoModelAgent:
    """Base model agent."""

    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig):
        self.model_config = model_config
        self.cache_config = cache_config

    def paging_adapters(self, weight_maps: List[AdapterWeightMap]):
        """paging adapter."""
        raise NotImplementedError('Not implemented.')

    async def async_forward(self, inputs: ModelInputs, swap_in_map: SwapMap,
                            swap_out_map: SwapMap):
        """model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
            swap_in_map (SwapMap): Cache maps to swap in.
            swap_out_map (SwapMap): Cache maps to swap out.
        """
        raise NotImplementedError('Not implemented.')

    def forward(self, inputs: ModelInputs, swap_in_map: SwapMap,
                swap_out_map: SwapMap):
        """model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
            swap_in_map (SwapMap): Cache maps to swap in.
            swap_out_map (SwapMap): Cache maps to swap out.
        """
        raise NotImplementedError('Not implemented.')

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        cache_config: CacheConfig,
                        trust_remote_code: bool,
                        adapters: Dict[str, str] = None,
                        tp: int = 1):
        """from pretrained."""
        return build_model_agent(pretrained_model_name_or_path,
                                 cache_config=cache_config,
                                 trust_remote_code=trust_remote_code,
                                 adapters=adapters,
                                 tp=tp)


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
                 adapters: Dict[str, str] = None,
                 trust_remote_code: bool = True):
        super().__init__(model_config=model_config, cache_config=cache_config)
        torch_dtype = model_config.dtype

        self.patched_model = self._build_model(
            model_path,
            torch_dtype=torch_dtype,
            adapters=adapters,
            trust_remote_code=trust_remote_code)

        block_size = _infer_block_size(self.patched_model, model_config,
                                       cache_config)
        if block_size != cache_config.block_size:
            cache_config.block_size = block_size
            logger.warning(f'infered block size: {block_size}')
        _update_cache_config(model_config, cache_config)

        self.cache_engine = CacheEngine(cache_config, model_config)
        self.stream = torch.cuda.Stream()

    def _build_model(self,
                     model_path: str,
                     torch_dtype: torch.dtype,
                     adapters: Dict[str, str] = None,
                     trust_remote_code: bool = True):
        """build patched model."""
        with LoadNoInit():
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                **self.model_config.init_kwargs)
            hf_model.eval()
            hf_model.config.use_cache = True
            # build for vlm model, TODO
            if hasattr(hf_model, 'model') and hasattr(hf_model.model,
                                                      'vision'):
                del hf_model.model.vision

        if adapters:
            _load_adapters(hf_model, adapters)

        patched_model = patch(hf_model, _PATCH_ARG_NAMES)

        if adapters:
            _unparam_lora_weight(patched_model)

        patched_model = patched_model.cuda()
        return patched_model

    def paging_adapters(self, weight_maps: List[AdapterWeightMap]):
        """paging adapter."""
        logger.info('paging adapters.')
        lora_linears = get_indexed_lora_linears(self.patched_model)
        cpu_caches = self.cache_engine.cpu_cache
        num_blocks = self.cache_engine.num_cpu_blocks
        cpu_caches = [(kcache.view(num_blocks,
                                   -1), vcache.view(num_blocks, -1))
                      for kcache, vcache in cpu_caches]
        for weight_map in weight_maps:
            weight_map.cache_adapter(lora_linears, cpu_caches)
        update_lora_linears(lora_linears, weight_maps, device='cuda')

    def _forward_impl(self, inputs: ModelInputs, swap_in_map: SwapMap,
                      swap_out_map: SwapMap):
        cache_swapping(self.cache_engine,
                       swap_in_map=swap_in_map,
                       swap_out_map=swap_out_map)
        output = model_forward(self.patched_model,
                               inputs,
                               self.cache_engine,
                               self.model_config.json_config,
                               world_size=1,
                               stream=self.stream)
        return output

    def forward(self, inputs: ModelInputs, swap_in_map: SwapMap,
                swap_out_map: SwapMap):
        """model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
            swap_in_map (SwapMap): Cache maps to swap in.
            swap_out_map (SwapMap): Cache maps to swap out.
        """
        output = self._forward_impl(inputs,
                                    swap_in_map=swap_in_map,
                                    swap_out_map=swap_out_map)
        self.stream.synchronize()
        return output

    async def async_forward(self, inputs: ModelInputs, swap_in_map: SwapMap,
                            swap_out_map: SwapMap):
        """model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
            swap_in_map (SwapMap): Cache maps to swap in.
            swap_out_map (SwapMap): Cache maps to swap out.
        """
        output = self._forward_impl(inputs,
                                    swap_in_map=swap_in_map,
                                    swap_out_map=swap_out_map)
        await asyncio.get_event_loop().run_in_executor(None,
                                                       self.stream.synchronize)
        return output


def _get_model_memory_usage(model: torch.nn.Module) -> int:
    """get model memory usage."""
    size = 0
    for _, param in model.named_parameters():
        size += param.element_size() * param.numel()
    for _, buf in model.named_buffers():
        size += buf.element_size() * param.numel()
    return size


def _create_device_map(model: torch.nn.Module,
                       world_size: int,
                       device_map: dict = None):
    """Distribute params to each devices."""
    free_mems = [get_gpu_memory(gpu_id)[0] for gpu_id in range(world_size)]
    free_mems = torch.tensor(free_mems)
    if device_map is None:
        device_map = dict()
    for name, param in model.named_parameters():
        device_id = free_mems.argmax().item()
        device_map[name] = device_id
        free_mems[device_id] -= param.numel() * param.element_size()
    for name, param in model.named_buffers():
        device_id = free_mems.argmax().item()
        device_map[name] = device_id
        free_mems[device_id] -= param.numel() * param.element_size()
    return device_map


@torch.inference_mode()
def _tp_build_model(
    rank: int,
    model_path: str,
    model_config: ModelConfig,
    cache_config: CacheConfig,
    adapters: Dict[str, str],
    world_size: int,
    trust_remote_code=True,
):
    """build tensor parallel model."""
    from accelerate import init_empty_weights

    patched_model = None
    cache_engine = None

    def __get_device_map(model, device_map=None):
        """get device map of model."""
        import psutil
        model_size = _get_model_memory_usage(model)
        if psutil.virtual_memory().available < model_size:
            logger.debug('Preload model on GPU.')
            return device_map
        else:
            logger.debug('Preload model on CPU.')
            return 'cpu'

    def __load_params_and_buffers(param_mod, mod):
        """load param and buffer."""
        for name, param in param_mod.named_parameters(recurse=False):
            mod.register_parameter(name, param)
        for name, buffer in param_mod.named_buffers(recurse=False):
            mod.register_buffer(name, buffer)

    def __load_state_dict_assign(param_model, model):
        """load state dict assign."""
        try:
            model.load_state_dict(param_model.state_dict(), assign=True)
        except Exception:
            __load_params_and_buffers(param_model, model)
            mods = dict(model.named_modules())
            for mod_name, param_mod in param_model.named_modules():
                mod = mods[mod_name]
                __load_params_and_buffers(param_mod, mod)

    def _broadcast_config(cache_config):
        """broadcast cache config, use minimum cache."""
        if rank == 0:
            gathered_configs = [None] * world_size
            dist.gather_object(cache_config, gathered_configs)
            num_gpu_blocks_list = [
                config.num_gpu_blocks for config in gathered_configs
            ]
            num_cpu_blocks_list = [
                config.num_cpu_blocks for config in gathered_configs
            ]
            min_num_gpu_blocks = min(num_gpu_blocks_list)
            min_num_cpu_blocks = min(num_cpu_blocks_list)
            cache_config.num_cpu_blocks = min_num_cpu_blocks
            cache_config.num_gpu_blocks = min_num_gpu_blocks
            config_list = [cache_config]
        else:
            gathered_configs = None
            dist.gather_object(cache_config, gathered_configs)
            config_list = [None]
        dist.broadcast_object_list(config_list)
        return config_list[0]

    try:
        config = model_config.hf_config
        torch_dtype = model_config.dtype
        device_map = None
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                **model_config.init_kwargs)
            if rank == 0:
                device_map = _create_device_map(model, world_size)
            _add_adapters(model, adapters)
            if rank == 0:
                # adapter would remove weight of linear.
                device_map = _create_device_map(model, world_size, device_map)
        model.eval()
        model.config.use_cache = True

        if rank == 0:
            with LoadNoInit():
                device_map = __get_device_map(model, device_map)
                param_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                    **model_config.init_kwargs)
                _load_adapters(param_model, adapters, device_map=device_map)
                __load_state_dict_assign(param_model, model)
                param_model = param_model.to('meta')
                del param_model

        patched_model = patch(
            model,
            extra_args=_PATCH_ARG_NAMES,
            rank=rank,
            world_size=world_size,
        )

        block_size = _infer_block_size(patched_model, model_config,
                                       cache_config, world_size)
        if block_size != cache_config.block_size:
            cache_config.block_size = block_size
            if rank == 0:
                logger.warning(f'infered block size: {block_size}')
        _update_cache_config(model_config,
                             cache_config,
                             gpu_id=rank,
                             world_size=world_size)
        cache_config = _broadcast_config(cache_config)
        cache_engine = CacheEngine(cache_config,
                                   model_config,
                                   rank=rank,
                                   world_size=world_size)
    except Exception as e:
        raise e

    return patched_model, cache_engine, cache_config


def _broadcast_inputs(rank: int, inputs: Any, stream: torch.cuda.Stream):
    """get input tensor parallel."""
    # broadcast meta info
    if rank != 0:
        inputs = [None, None, None]

    with torch.cuda.stream(stream):
        dist.broadcast_object_list(inputs)
    return inputs


@torch.inference_mode()
def _tp_paging_adapters(
    rank: int,
    patched_model: torch.nn.Module,
    cache_engine: CacheEngine,
    weight_map: AdapterWeightMap = None,
):
    """tp paging adapters."""

    def __get_weight_map():
        """get weight map."""
        if rank == 0:
            assert weight_map is not None
            dist_obj = [weight_map]
        else:
            dist_obj = [None]
        dist.broadcast_object_list(dist_obj)
        return dist_obj[0]

    def __paging(weight_maps):
        """paging."""
        lora_linears = get_indexed_lora_linears(patched_model)
        cpu_caches = cache_engine.cpu_cache
        num_blocks = cache_engine.num_cpu_blocks
        cpu_caches = [(kcache.view(num_blocks,
                                   -1), vcache.view(num_blocks, -1))
                      for kcache, vcache in cpu_caches]
        for weight_map in weight_maps:
            weight_map.cache_adapter(lora_linears, cpu_caches)
        update_lora_linears(lora_linears, weight_maps, device='cuda')

    weight_maps = __get_weight_map()

    if rank == 0:
        logger.info('tp paging adapters.')
    if len(weight_maps) > 0:
        __paging(weight_maps)


def _tp_model_loop(
    rank: int,
    model_path: str,
    model_config: ModelConfig,
    cache_config: CacheConfig,
    adapters: Dict[str, str],
    world_size: int,
    trust_remote_code=True,
):
    """Start model loops for tensor parallel model inference.

    Args:
        rank (int): Distribution rank.
        model_path (int): Path of the hugging face model. Could be
            local or online.
        model_config (ModelConfig): The config of the model.
        cache_config (CacheConfig): The config of the cache.
        in_que (mp.Queue): Input queue. Used to receive model input.
        out_que (mp.Queue): Output queue. Used to send the model output.
        world_size (int): The distribution world size.
    """
    stream = torch.cuda.Stream()
    patched_model, cache_engine, _ = _tp_build_model(
        rank,
        model_path,
        model_config,
        cache_config,
        adapters,
        world_size=world_size,
        trust_remote_code=trust_remote_code)

    if adapters:
        _tp_paging_adapters(rank,
                            patched_model,
                            cache_engine=cache_engine,
                            weight_map=None)

    while True:
        inputs, swap_in_map, swap_out_map = _broadcast_inputs(
            rank, None, stream)

        cache_swapping(cache_engine,
                       swap_in_map=swap_in_map,
                       swap_out_map=swap_out_map)

        model_forward(patched_model,
                      inputs,
                      cache_engine,
                      model_config.json_config,
                      world_size=world_size,
                      stream=stream)


def _start_tp_process(proc_id: int,
                      world_size: int,
                      func: Callable,
                      args: List = None,
                      kwargs: Dict = None):
    """Start the tensor parallel process.

    Args:
        rank (int): The distribution rank.
        world_size (int): The distribution world size.
        func (Callable): The function to be called in the process.
        args (List): The arguments of the func.
        kwargs (Dict): The keyword arguments of the func.
    """
    rank = proc_id + 1
    try:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        with torch.cuda.device(rank), torch.inference_mode():
            args = args or tuple()
            kwargs = kwargs or dict()
            func(rank, *args, **kwargs)
    except Exception as e:
        from traceback import print_exc
        logger.error(f'Rank[{rank}] failed.')
        print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e


def _check_context_alive(mp_context: mp.ProcessContext):
    """check context alive."""
    procs: List[mp.Process] = mp_context.processes
    failed_ranks = list(idx for idx, p in enumerate(procs) if not p.is_alive())
    if len(failed_ranks) > 0:
        for p in procs:
            if p.is_alive():
                p.terminate()
            else:
                p.close()
        logger.error(f'TP process Rank{failed_ranks} failed.')
        exit(1)


def _find_available_port() -> bool:
    """find available port."""
    import socket
    port = 29500
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
            port += 1


class TPModelAgent(AutoModelAgent):
    """Tensor Parallelism model agent.

    load model on multiple GPUs

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
                 world_size: int,
                 adapters: Dict[str, str] = None,
                 trust_remote_code: bool = True) -> None:
        import signal

        def __signal_term_handler(sig, frame):
            """sigterm handler."""
            if hasattr(self, 'mp_context'):
                procs = self.mp_context.processes
                for p in procs:
                    if p.is_alive():
                        p.kill()
            logger.error(f'Get signal[{sig}], kill all processes.')
            signal.signal(sig, signal.SIG_DFL)
            signal.raise_signal(sig)

        super().__init__(model_config=model_config, cache_config=cache_config)

        signal.signal(signal.SIGTERM, __signal_term_handler)

        self.mp_ctx = mp.get_context('spawn')
        self.world_size = world_size

        self._start_sub_process(model_path,
                                model_config=model_config,
                                cache_config=cache_config,
                                adapters=adapters,
                                world_size=world_size,
                                trust_remote_code=trust_remote_code)

        model, cache_engine, cache_config = self._build_model(
            model_path=model_path,
            model_config=model_config,
            cache_config=cache_config,
            adapters=adapters,
            world_size=world_size,
            trust_remote_code=trust_remote_code,
        )
        self.patched_model = model
        self.cache_config = cache_config
        self.cache_engine = cache_engine
        self.stream = torch.cuda.Stream()

    def _start_sub_process(self, model_path: str, model_config: ModelConfig,
                           cache_config: CacheConfig, adapters: Dict[str, str],
                           world_size: int, trust_remote_code: bool):
        """Start tensor parallel sub process."""
        port = _find_available_port()
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', str(port))
        addr = os.environ['MASTER_ADDR']
        port = os.environ['MASTER_PORT']
        logger.info(f'MASTER_ADDR={addr}, MASTER_PORT={port}')

        self.mp_context = mp.spawn(
            _start_tp_process,
            args=(
                world_size,
                _tp_model_loop,
                (model_path, ),
                dict(model_config=model_config,
                     cache_config=cache_config,
                     adapters=adapters,
                     world_size=world_size,
                     trust_remote_code=trust_remote_code),
            ),
            nprocs=world_size - 1,
            join=False,
            daemon=True,
        )
        _check_context_alive(self.mp_context)

        rank = 0
        try:
            dist.init_process_group('nccl', rank=rank, world_size=world_size)
        except Exception as e:
            from traceback import print_exc
            logger.error(f'Rank[{rank}] failed.')
            print_exc()
            if dist.is_initialized():
                dist.destroy_process_group()
            raise e

    @torch.inference_mode()
    def _build_model(
        self,
        model_path: str,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        adapters: Dict[str, str],
        world_size: int,
        trust_remote_code=True,
    ):
        """build model."""
        _check_context_alive(self.mp_context)
        rank = 0
        model, cache_engine, cache_config = _tp_build_model(
            rank,
            model_path=model_path,
            model_config=model_config,
            cache_config=cache_config,
            adapters=adapters,
            world_size=world_size,
            trust_remote_code=trust_remote_code,
        )

        return model, cache_engine, cache_config

    def paging_adapters(self, weight_maps: List[AdapterWeightMap]):
        """load adapter."""
        if not weight_maps:
            return
        _check_context_alive(self.mp_context)
        rank = 0
        _tp_paging_adapters(rank, self.patched_model, self.cache_engine,
                            weight_maps)

    def _forward_impl(self, inputs: ModelInputs, swap_in_map: SwapMap,
                      swap_out_map: SwapMap):
        """forward impl."""
        _check_context_alive(self.mp_context)
        rank = 0
        _broadcast_inputs(rank, [inputs, swap_in_map, swap_out_map],
                          self.stream)
        cache_swapping(self.cache_engine,
                       swap_in_map=swap_in_map,
                       swap_out_map=swap_out_map)
        output = model_forward(self.patched_model,
                               inputs,
                               self.cache_engine,
                               self.model_config.json_config,
                               world_size=1,
                               stream=self.stream)
        return output

    def forward(self, inputs: ModelInputs, swap_in_map: SwapMap,
                swap_out_map: SwapMap):
        """model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
            swap_in_map (SwapMap): Cache maps to swap in.
            swap_out_map (SwapMap): Cache maps to swap out.
        """
        output = self._forward_impl(inputs,
                                    swap_in_map=swap_in_map,
                                    swap_out_map=swap_out_map)
        self.stream.synchronize()
        return output

    async def async_forward(self, inputs: ModelInputs, swap_in_map: SwapMap,
                            swap_out_map: SwapMap):
        """model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
            swap_in_map (SwapMap): Cache maps to swap in.
            swap_out_map (SwapMap): Cache maps to swap out.
        """
        output = self._forward_impl(inputs,
                                    swap_in_map=swap_in_map,
                                    swap_out_map=swap_out_map)
        await asyncio.get_event_loop().run_in_executor(None,
                                                       self.stream.synchronize)
        return output


def build_model_agent(model_path: str,
                      cache_config: CacheConfig,
                      trust_remote_code: bool,
                      adapters: Dict[str, str] = None,
                      tp: int = 1):
    """create model agent."""
    model_config = ModelConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code)
    if tp == 1:
        model_agent = BaseModelAgent(model_path,
                                     model_config=model_config,
                                     cache_config=cache_config,
                                     adapters=adapters,
                                     trust_remote_code=trust_remote_code)
    else:
        model_agent = TPModelAgent(model_path,
                                   model_config=model_config,
                                   cache_config=cache_config,
                                   world_size=tp,
                                   adapters=adapters,
                                   trust_remote_code=trust_remote_code)
    return model_agent
