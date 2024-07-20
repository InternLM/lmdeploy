# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
import warnings
from datetime import timedelta
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
from torch import multiprocessing as mp

from lmdeploy.pytorch.accel import LoadNoInit
from lmdeploy.utils import get_logger

from ..adapter.adapter import (AdapterWeightMap, get_indexed_lora_linears,
                               get_loralinear_info, update_lora_linears)
from ..config import CacheConfig, ModelConfig
from ..devices import DeviceContext, get_device_manager
from ..model_inputs import ModelInputs, StepContext
from ..models.patch import patch, update_model
from ..utils import get_gpu_memory
from ..weight_loader.model_weight_loader import load_model_weights
from .cache_engine import CacheEngine

logger = get_logger('lmdeploy')

_PATCH_ARG_NAMES = ['context', 'use_origin']


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

    def __adjust_block_size():
        """adjust block_size."""
        # TODO: support kernel with both large head dim and large block size.
        if model_config.k_head_dim >= 512 and cache_config.block_size > 32:
            cache_config.block_size = 32
            rank = 0
            if dist.is_initialized():
                rank = dist.get_rank()
            if rank == 0:
                logger.warning(
                    f'Update `block_size={cache_config.block_size}`'
                    f' for large `head_dim={model_config.k_head_dim}`.')

    __adjust_block_size()

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
        inputs = inputs.to_device('cuda')
        context = StepContext.new(
            inputs=inputs,
            world_size=world_size,
            kv_caches=cache_engine.gpu_cache,
            cache_config=cache_engine.cache_config,
        )
        ctx_mgr = model.ctx_mgr
        with ctx_mgr.context(context):
            output = model(
                input_ids=inputs.input_ids,
                position_ids=context.position_ids,
                past_key_values=cache_engine.gpu_cache,
                context=context,
            )
    return dict(logits=output)


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


def _remove_unused_modules(hf_model: torch.nn.Module, model_cfg: ModelConfig):
    """remove unused modules."""
    if model_cfg.unused_modules is not None and len(
            model_cfg.unused_modules) > 0:
        for mod in model_cfg.unused_modules:
            has_mod = True
            parts = mod.split('.')
            mod_path = 'hf_model'
            for p in parts:
                if eval(f'hasattr({mod_path}, "{p}")'):
                    mod_path = f'{mod_path}.{p}'
                else:
                    has_mod = False
                    break
            if has_mod:
                exec(f'del {mod_path}')
    return hf_model


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

    def get_loralinear_info(self):
        """get lora linear info."""
        raise NotImplementedError('Not implemented')

    def get_block_numel(self):
        """get block nelement."""
        raise NotImplementedError('Not implemented')

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

        _update_cache_config(model_config, cache_config)

        self.cache_engine = CacheEngine(cache_config, model_config)
        self.stream = torch.cuda.Stream()

    def _build_model(self,
                     model_path: str,
                     torch_dtype: torch.dtype,
                     adapters: Dict[str, str] = None,
                     trust_remote_code: bool = True):
        """build patched model."""
        device = 'cuda'
        with LoadNoInit(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            hf_model = self.model_config.auto_model_cls.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=trust_remote_code,
                **self.model_config.init_kwargs)
            hf_model.eval()
            hf_model.config.use_cache = True
            # build for vlm model
            _remove_unused_modules(hf_model, self.model_config)

        if adapters:
            _load_adapters(hf_model, adapters)

        patched_model = update_model(hf_model)

        if adapters:
            _unparam_lora_weight(patched_model)

        return patched_model

    def get_loralinear_info(self):
        """get lora linear info."""
        return get_loralinear_info(self.patched_model)

    def get_block_numel(self):
        """get block nelement."""
        k_cache = self.cache_engine.local_gpu_cache[0][0]
        return k_cache[0].numel()

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
        output = model_forward(
            self.patched_model,
            inputs,
            self.cache_engine,
            world_size=1,
            stream=self.stream,
        )
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
        with init_empty_weights(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = model_config.auto_model_cls.from_config(
                config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                **model_config.init_kwargs)
            # build for vlm model
            _remove_unused_modules(model, model_config)
            if rank == 0:
                device_map = _create_device_map(model, world_size)
            _add_adapters(model, adapters)
            if rank == 0:
                # adapter would remove weight of linear.
                device_map = _create_device_map(model, world_size, device_map)

        model.eval()
        model.config.use_cache = True

        patched_model = patch(model)
        load_model_weights(patched_model,
                           model_path,
                           adapters,
                           rank=rank,
                           world_size=world_size,
                           device='cuda')
        if rank == 0:
            logger.debug('Updating model.')
        patched_model = update_model(patched_model)

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

        model_forward(
            patched_model,
            inputs,
            cache_engine,
            world_size=world_size,
            stream=stream,
        )


def _start_tp_process(proc_id: int,
                      world_size: int,
                      func: Callable,
                      device_context: DeviceContext,
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
        dist.init_process_group('nccl',
                                rank=rank,
                                world_size=world_size,
                                timeout=timedelta(days=35600))
        torch.cuda.set_device(rank)
        with get_device_manager().context(
                device_context), torch.inference_mode():
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

        device_context = get_device_manager().current_context()
        self.mp_context = mp.spawn(
            _start_tp_process,
            args=(
                world_size,
                _tp_model_loop,
                device_context,
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
            dist.init_process_group('nccl',
                                    rank=rank,
                                    world_size=world_size,
                                    timeout=timedelta(days=35600))
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

    def get_loralinear_info(self):
        """get lora linear info."""
        return get_loralinear_info(self.patched_model)

    def get_block_numel(self):
        """get block nelement."""
        k_cache = self.cache_engine.local_gpu_cache[0][0]
        return k_cache[0].numel()

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
        output = model_forward(
            self.patched_model,
            inputs,
            self.cache_engine,
            world_size=1,
            stream=self.stream,
        )
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
