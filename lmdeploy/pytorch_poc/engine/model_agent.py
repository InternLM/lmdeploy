# Copyright (c) OpenMMLab. All rights reserved.

import json
import os
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
from torch import multiprocessing as mp
from torch.distributed._tensor import DeviceMesh, Replicate, distribute_tensor
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, cached_file

from lmdeploy.pytorch.accel import LoadNoInit
from lmdeploy.pytorch_poc.config import CacheConfig, ModelConfig
from lmdeploy.pytorch_poc.models import patch
from lmdeploy.pytorch_poc.utils import get_gpu_memory
from lmdeploy.utils import get_logger

from .cache_engine import CacheEngine

logger = get_logger('lmdeploy')


def _update_cache_config(model_config: ModelConfig,
                         cache_config: CacheConfig,
                         gpu_id: int = 0):
    """Update the gpu mem and cpu mem according to model info.

    Args:
        model_config (ModelConfig): The config of the model.
        cache_config (CacheConfig): The config of the cache info.
        gpu_id (int): The GPU id to use.
    """
    GPU_MEM_PERCENT = 0.7
    SWAP_SPACE = 4 * (1 << 30)
    gpu_mem_physical_free, _ = get_gpu_memory(gpu_id)
    gpu_mem = gpu_mem_physical_free * GPU_MEM_PERCENT
    cpu_mem = SWAP_SPACE
    cache_block_size = CacheEngine.get_cache_block_size(
        cache_config.block_size, model_config)
    if cache_config.num_cpu_blocks == 0:
        cache_config.num_cpu_blocks = int(cpu_mem / cache_block_size)
    if cache_config.num_gpu_blocks == 0:
        cache_config.num_gpu_blocks = int(gpu_mem / cache_block_size)


def _get_torch_dtype(config: Any, default: str = 'float16'):
    """Get the torch dtype from the model config.

    Args:
        config: Config of the hf model.
        default (str): default device type.
    """
    torch_dtype = getattr(config, 'torch_dtype', default)
    return eval(f'torch.{torch_dtype}')


class ModelContext:
    """context of Model.

    patched model might need extra information to perform inference.
    This dataclass provide these infos and tools.

    Args:
        block_offsets (List[List[int]]): The block offsets of each layers.
        history_lengths (List[int]): The history length of the caches.
        position_ids (Tensor): The position ids of the input tokens.
        world_size (int): The distribution world size.
        device (str): The device of the tensors.
    """

    def __init__(
        self,
        block_offsets: List[List[int]],
        history_lengths: List[int],
        position_ids: torch.Tensor,
        q_start_loc: torch.Tensor,
        seq_length: torch.Tensor,
        world_size: int = 1,
        device='cuda',
    ):
        self.block_offsets_list = block_offsets
        self.history_lengths = history_lengths
        self.position_ids = position_ids
        self.q_start_loc = q_start_loc
        self.seq_length = seq_length
        self.world_size = world_size

        # padding zero
        # torch.nn.utils.rnn.pad_sequence is slower than manually concate
        offset_len = [len(offset) for offset in block_offsets]
        max_offsets_len = max(offset_len)
        pad_block_offsets = [
            offset + [0] * (max_offsets_len - off_len)
            for offset, off_len in zip(block_offsets, offset_len)
        ]
        self.block_offsets = torch.tensor(pad_block_offsets).to(device)

        # update position_ids_1d
        if position_ids.size(1) == 1:
            position_ids_1d = position_ids.flatten()
        else:
            position_ids_1d = [
                ids[:l] for ids, l in zip(position_ids.cpu(), seq_length.cpu())
            ]
            position_ids_1d = torch.cat(position_ids_1d).to(device)
        self.position_ids_1d = position_ids_1d

    def get_block_offsets(self):
        """return block offsets."""
        return self.block_offsets

    def fill_cache(
        self,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
        start_loc: torch.Tensor,
        seq_length: torch.Tensor,
        k_caches: torch.Tensor,
        v_caches: torch.Tensor,
    ):
        """Fill current key/value to cache.

        Args:
            k_states (torch.Tensor): [packed_seq_len, head, dim]
            v_states (torch.Tensor): [packed_seq_len, head, dim]
            start_loc (torch.Tensor): [bs]
            seq_length (torch.Tensor): [bs]
            k_caches (torch.Tensor): [num_blocks, block_size, head, dim]
            v_caches (torch.Tensor): [num_blocks, block_size, head, dim]
        """

        block_size = k_caches.size(1)
        block_offsets = self.block_offsets_list

        history_lengths = torch.tensor(self.history_lengths)
        first_free_block_offsets = history_lengths // block_size
        first_token_offsets = history_lengths % block_size

        for bid in range(len(history_lengths)):
            loc = start_loc[bid]
            seq_len = seq_length[bid]
            b_offsets = block_offsets[bid]
            free_offset = first_free_block_offsets[bid]
            token_offset = first_token_offsets[bid]

            assert 0 <= loc <= k_states.size(0)
            assert 0 <= loc + seq_len <= k_states.size(0)

            k_state = k_states[loc:loc + seq_len]
            v_state = v_states[loc:loc + seq_len]

            # fill remain(last non-full block)
            block_id = b_offsets[free_offset]
            fill_token_num = min(block_size - token_offset, seq_len)

            assert 0 <= fill_token_num <= block_size

            k_caches[block_id][token_offset:token_offset +
                               fill_token_num] = k_state[:fill_token_num]
            v_caches[block_id][token_offset:token_offset +
                               fill_token_num] = v_state[:fill_token_num]

            # update offset
            seq_len = seq_len - fill_token_num
            free_offset += 1
            k_state = k_state[fill_token_num:]
            v_state = v_state[fill_token_num:]

            for seq_offset in range(0, seq_len, block_size):
                token_num = min(seq_len - seq_offset, block_size)
                block_id = b_offsets[free_offset]
                k_caches[block_id][:token_num] = k_state[:token_num]
                v_caches[block_id][:token_num] = v_state[:token_num]

                free_offset += 1
                k_state = k_state[token_num:]
                v_state = v_state[token_num:]


class BaseModelAgent:
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
                 trust_remote_code: bool = True):
        self.model_config = model_config
        self.cache_config = cache_config
        torch_dtype = model_config.dtype

        with LoadNoInit():
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code)
            hf_model.eval()

        self.patched_model = patch(
            hf_model, ['context', 'use_origin', 'q_seq_info']).cuda()
        _update_cache_config(model_config, cache_config)

        self.cache_engine = CacheEngine(cache_config, model_config)
        self.stream = torch.cuda.Stream()
        logger.debug(
            f'Initialize cache engine with {cache_config.num_gpu_blocks}'
            f' gpu blocks and {cache_config.num_cpu_blocks} cpu blocks.')

    def forward(self, inputs: Dict, swap_in_map: Dict[int, int],
                swap_out_map: Dict[int, int]):
        """model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
            swap_in_map (Dict[int, int]): Cache maps to swap in.
            swap_out_map (Dict[int, int]): Cache maps to swap out.
        """

        # swap in/out
        issued_cache_op = False
        if len(swap_in_map) > 0:
            self.cache_engine.swap_in(swap_in_map)
            issued_cache_op = True
        if len(swap_out_map) > 0:
            self.cache_engine.swap_out(swap_out_map)
            issued_cache_op = True

        if issued_cache_op:
            cache_events = self.cache_engine.events
        else:
            cache_events = None

        if cache_events is not None:
            for event in cache_events:
                event.wait()

        with torch.no_grad(), torch.cuda.stream(self.stream):
            # forward
            output = self.patched_model(
                input_ids=inputs['input_ids'],
                position_ids=inputs['position_ids'],
                attention_mask=inputs['attention_mask'],
                past_key_values=self.cache_engine.gpu_cache,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                use_origin=False,
                context=ModelContext(
                    block_offsets=inputs['block_offsets'],
                    history_lengths=inputs['history_lengths'],
                    position_ids=inputs['position_ids'],
                    q_start_loc=inputs['q_start_loc'],
                    seq_length=inputs['seq_length'],
                ),
                q_seq_info=(inputs['q_start_loc'], inputs['seq_length']),
            )
        self.stream.synchronize()
        return output['logits']


def _tp_model_loop(
    rank: int,
    model_path: str,
    extra_args: List[str],
    model_config: ModelConfig,
    cache_config: CacheConfig,
    in_que: mp.Queue,
    out_que: mp.Queue,
    world_size: int,
    trust_remote_code=True,
):
    """Start model loops for tensor parallel model inference.

    Args:
        rank (int): Distribution rank.
        model_path (int): Path of the hugging face model. Could be
            local or online.
        extra_args (List[str]): The extra arguments to add to the
            patched model.
        model_config (ModelConfig): The config of the model.
        cache_config (CacheConfig): The config of the cache.
        in_que (mp.Queue): Input queue. Used to receive model input.
        out_que (mp.Queue): Output queue. Used to send the model output.
        world_size (int): The distribution world size.
    """
    from accelerate import init_empty_weights

    device_mesh = DeviceMesh('cuda', list(range(world_size)))

    error_code = 0
    error_type = None

    try:
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code)
        torch_dtype = _get_torch_dtype(config)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code)

        try:
            torch_model_json_path = cached_file(model_path, WEIGHTS_INDEX_NAME)
            with open(torch_model_json_path, mode='r') as f:
                torch_model_json = json.load(f)

            weight_map = torch_model_json['weight_map']

            checkpoints = list(set(weight_map.values()))
            checkpoints = [
                cached_file(model_path, ckpt) for ckpt in checkpoints
            ]
        except Exception:
            logger.warning(f'load failed, try load from {WEIGHTS_NAME}.')
            checkpoints = [cached_file(model_path, WEIGHTS_NAME)]
        patched_model = patch(
            model,
            extra_args=extra_args,
            rank=rank,
            world_size=world_size,
            checkpoints=checkpoints,
        )

        _update_cache_config(model_config, cache_config)
        cache_engine = CacheEngine(cache_config,
                                   model_config,
                                   rank=rank,
                                   world_size=world_size)
        stream = torch.cuda.Stream()
    except Exception as e:
        error_code = 1
        error_type = e

    error_code = torch.tensor(error_code).cuda(rank)

    dist.all_reduce(error_code)

    error_code = error_code.item()
    if error_code > 0:
        all_errors = [None] * world_size
        dist.all_gather_object(all_errors, error_type)
        if rank == 0:
            out_que.put((1, all_errors, cache_config))
        return
    else:
        if rank == 0:
            out_que.put((0, None, cache_config))

    while True:
        input_tensor_names = [
            'input_ids',
            'seq_length',
            'attention_mask',
            'position_ids',
            'q_start_loc',
        ]

        if rank == 0:
            inputs, swap_in_map, swap_out_map = in_que.get()
            input_tensors = dict((name, t) for name, t in inputs.items()
                                 if name in input_tensor_names)
            input_metas = dict((name, (t.shape, t.dtype))
                               for name, t in input_tensors.items())
            input_metas['block_offsets'] = inputs['block_offsets']
            input_metas['history_lengths'] = inputs['history_lengths']

            objs = [input_metas, swap_in_map, swap_out_map]
        else:
            objs = [None, None, None]

        # broadcast input shapes and dtypes
        dist.broadcast_object_list(objs)

        if rank != 0:
            input_metas = objs[0]
            input_tensors = dict((name, torch.empty(meta[0], dtype=meta[1]))
                                 for name, meta in input_metas.items()
                                 if name in input_tensor_names)

        updated_inputs = dict()
        for name, t in input_tensors.items():
            updated_inputs[name] = distribute_tensor(t,
                                                     device_mesh=device_mesh,
                                                     placements=[Replicate()
                                                                 ]).to_local()

        inputs = updated_inputs
        inputs['block_offsets'] = input_metas['block_offsets']
        inputs['history_lengths'] = input_metas['history_lengths']

        swap_in_map = objs[1]
        swap_out_map = objs[2]

        # swap in/out
        issued_cache_op = False
        if len(swap_in_map) > 0:
            cache_engine.swap_in(swap_in_map)
            issued_cache_op = True
        if len(swap_out_map) > 0:
            cache_engine.swap_out(swap_out_map)
            issued_cache_op = True

        if issued_cache_op:
            cache_events = cache_engine.events
        else:
            cache_events = None

        if cache_events is not None:
            for event in cache_events:
                event.wait()

        with torch.no_grad(), torch.cuda.stream(stream):
            output = patched_model(
                input_ids=inputs['input_ids'],
                position_ids=inputs['position_ids'],
                attention_mask=inputs['attention_mask'],
                past_key_values=cache_engine.gpu_cache,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                use_origin=False,
                context=ModelContext(
                    block_offsets=inputs['block_offsets'],
                    history_lengths=inputs['history_lengths'],
                    position_ids=inputs['position_ids'],
                    q_start_loc=inputs['q_start_loc'],
                    seq_length=inputs['seq_length'],
                    world_size=world_size,
                ),
                q_seq_info=(inputs['q_start_loc'], inputs['seq_length']),
            )

            if rank == 0:
                out_que.put((0, output['logits'].cpu()))


def _start_tp_process(rank: int,
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
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        with torch.cuda.device(rank), torch.no_grad():
            if args is None:
                args = tuple()
            if kwargs is None:
                kwargs = dict()
            func(rank, *args, **kwargs)
    except Exception as e:
        from traceback import print_exc

        logger.error(f'rank[{rank}]: {e}')
        print_exc()
        raise e


class TPModelAgent:
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
                 trust_remote_code: bool = True) -> None:
        mp.set_start_method('spawn')
        self.world_size = world_size
        self.model_config = model_config
        self.cache_config = cache_config
        self.tp_model_in_que = mp.Queue(10)
        self.tp_model_out_que = mp.Queue(10)

        self.patch_model_tp(model_path,
                            ['context', 'use_origin', 'q_seq_info'],
                            model_config=model_config,
                            cache_config=cache_config,
                            in_que=self.tp_model_in_que,
                            out_que=self.tp_model_out_que,
                            world_size=world_size,
                            trust_remote_code=trust_remote_code)
        ret_code, e, cache_config = self.tp_model_out_que.get()
        if ret_code != 0:
            logger.error(f'Init tp model failed with error: {e}')
            for err in e:
                if err is not None:
                    raise err
        self.cache_config = cache_config

    def patch_model_tp(self, model_path: str, extra_args: List[str],
                       model_config: ModelConfig, cache_config: CacheConfig,
                       in_que: mp.Queue, out_que: mp.Queue, world_size: int,
                       trust_remote_code: bool):
        """Start tensor parallel sub process.

        Args:
            model_path (int): Path of the hugging face model.
                Could be local or online.
            extra_args (List[str]): The extra arguments to add to the
                patched model.
            model_config (ModelConfig): The config of the model.
            cache_config (CacheConfig): The config of the cache.
            in_que (mp.Queue): Input queue. Used to receive model input.
            out_que (mp.Queue): Output queue. Used to send the model output.
            world_size (int): The distribution world size.
        """
        self.mp_context = mp.spawn(
            _start_tp_process,
            args=(
                world_size,
                _tp_model_loop,
                (model_path, extra_args),
                dict(model_config=model_config,
                     cache_config=cache_config,
                     in_que=in_que,
                     out_que=out_que,
                     world_size=world_size,
                     trust_remote_code=trust_remote_code),
            ),
            nprocs=world_size,
            join=False,
            daemon=True,
        )

    def forward(self, inputs: Dict, swap_in_map: Dict[int, int],
                swap_out_map: Dict[int, int]):
        """model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
            swap_in_map (Dict[int, int]): Cache maps to swap in.
            swap_out_map (Dict[int, int]): Cache maps to swap out.
        """
        with torch.no_grad():
            self.tp_model_in_que.put((inputs, swap_in_map, swap_out_map))

        ret_code, output = self.tp_model_out_que.get()
        if ret_code != 0:
            logger.error('tp forward failed.')
            exit()

        return output
