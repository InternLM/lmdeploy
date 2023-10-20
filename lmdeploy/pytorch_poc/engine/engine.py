# Copyright (c) OpenMMLab. All rights reserved.
import enum
import itertools
import json
import os
import time
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
from torch import multiprocessing as mp
from torch.distributed._tensor import DeviceMesh, Replicate, distribute_tensor
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.generation.logits_process import (LogitsProcessorList,
                                                    TemperatureLogitsWarper,
                                                    TopKLogitsWarper,
                                                    TopPLogitsWarper)
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, cached_file

from lmdeploy.pytorch.accel import LoadNoInit
from lmdeploy.pytorch_poc.config import (CacheConfig, ModelConfig,
                                         SchedulerConfig)
from lmdeploy.pytorch_poc.messages import (MessageStatus, SamplingParam,
                                           SchedulerSequence, SchedulerSession)
from lmdeploy.pytorch_poc.models import patch
from lmdeploy.pytorch_poc.paging import Scheduler
from lmdeploy.pytorch_poc.utils import get_gpu_memory
from lmdeploy.utils import get_logger

from .cache_engine import CacheEngine

logger = get_logger('lmdeploy')


class RequestType(enum.Enum):
    """Request type."""

    ADD_SESSION = enum.auto()
    ADD_MESSAGE = enum.auto()
    STOP_SESSION = enum.auto()
    END_SESSION = enum.auto()
    STOP_ENGINE = enum.auto()
    RESUME_ENGINE = enum.auto()


class ResponseType(enum.Enum):
    """Response type."""

    SUCCESS = enum.auto()
    FINISH = enum.auto()
    ENGINE_STOP_ERROR = enum.auto()
    REPEAT_ERROR = enum.auto()
    NOT_EXIST_ERROR = enum.auto()


@dataclass
class Request:
    """Request."""

    type: RequestType
    resp: Queue
    req_id: int
    data: Any = None


@dataclass
class Response:
    """Response."""

    type: ResponseType
    req_id: int
    data: Any = None
    err_msg: str = ''


@dataclass
class InferOutput:
    """The output of the model inference."""

    session_id: int
    token_ids: List[int]
    req_id: int = 0
    finish: bool = False
    logits: torch.Tensor = None


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
        pad_sequence = torch.nn.utils.rnn.pad_sequence
        block_offsets = [
            torch.tensor(offset, device=device) for offset in block_offsets
        ]
        block_offsets = pad_sequence(block_offsets, True)
        self.block_offsets = block_offsets

        # update position_ids_1d
        position_ids_1d = [ids[:l] for ids, l in zip(position_ids, seq_length)]
        position_ids_1d = torch.cat(position_ids_1d)
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
    reserved_mem = torch.cuda.memory_reserved(gpu_id)
    gpu_mem = (get_gpu_memory(gpu_id) - reserved_mem) * GPU_MEM_PERCENT
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

        with torch.no_grad():
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


class Engine:
    """The inference engine of lmdeploy pytorch.

    Args:
        model_path (str): The hugging face model path.
        scheduler_config (SchedulerConfig): The config of the scheduler.
        cache_config (CacheConfig): The config of the cache info.
        tp (int): Number of tensor parallel.
    """

    def __init__(
        self,
        model_path: str,
        scheduler_config: SchedulerConfig = None,
        cache_config: CacheConfig = None,
        tp: int = 1,
        trust_remote_code=True,
    ) -> None:

        self.tp = tp
        hf_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code)
        torch_dtype = _get_torch_dtype(hf_config)
        self.torch_dtype = torch_dtype

        if scheduler_config is None:
            scheduler_config = SchedulerConfig(max_batches=64,
                                               max_session_len=2048,
                                               max_request_output_len=512)
        if cache_config is None:
            cache_config = CacheConfig(block_size=64,
                                       num_cpu_blocks=0,
                                       num_gpu_blocks=0)
        if 'falcon' in model_path:
            if hf_config.multi_query:
                kv_dim = hf_config.hidden_size // hf_config.num_attention_heads
                kv_head = 1
            else:
                kv_dim = hf_config.hidden_size
                kv_head = hf_config.num_attention_heads
            model_config = ModelConfig(
                kv_dim,
                hf_config.num_hidden_layers,
                kv_head,
                bos_token_id=hf_config.bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                dtype=torch_dtype,
            )
        elif 'chatglm' in model_path:
            model_config = ModelConfig(
                hf_config.hidden_size // hf_config.num_attention_heads *
                hf_config.multi_query_group_num,
                hf_config.num_layers,
                hf_config.multi_query_group_num,
                bos_token_id=hf_config.bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                dtype=torch_dtype,
            )
        else:
            model_config = ModelConfig(
                hf_config.hidden_size,
                hf_config.num_hidden_layers,
                hf_config.num_attention_heads,
                bos_token_id=hf_config.bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                dtype=torch_dtype,
            )

        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.model_config = model_config
        self.session_len = scheduler_config.max_session_len

        if tp == 1:
            with LoadNoInit():
                hf_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code)
                hf_model.eval()
                hf_model.config.use_cache = True

            self.patched_model = patch(
                hf_model, ['context', 'use_origin', 'q_seq_info']).cuda()
            _update_cache_config(model_config, cache_config)

            self.cache_engine = CacheEngine(cache_config, model_config)
            logger.debug(
                f'Initialize cache engine with {cache_config.num_gpu_blocks}'
                f' gpu blocks and {cache_config.num_cpu_blocks} cpu blocks.')
        else:
            mp.set_start_method('spawn')
            self.tp_model_in_que = mp.Queue(5)
            self.tp_model_out_que = mp.Queue(5)

            self.patch_model_tp(
                model_path,
                ['context', 'use_origin', 'q_seq_info'],
                model_config=model_config,
                cache_config=cache_config,
                in_que=self.tp_model_in_que,
                out_que=self.tp_model_out_que,
                world_size=tp,
            )

            ret_code, e, cache_config = self.tp_model_out_que.get()
            if ret_code != 0:
                logger.error(f'Init tp model failed with error: {e}')
                for err in e:
                    if err is not None:
                        raise err

            # # have to update cache on host to support scheduler
            # _update_cache_config(model_config, cache_config)

        self.scheduler = Scheduler(scheduler_config, cache_config)

        self.requests = Queue(scheduler_config.max_batches)

        # create main thread
        loop_threads = Thread(target=self.loop, daemon=True)
        loop_threads.start()
        self.loop_threads = loop_threads

    def patch_model_tp(
        self,
        model_path: str,
        extra_args: List[str],
        model_config: ModelConfig,
        cache_config: CacheConfig,
        in_que: mp.Queue,
        out_que: mp.Queue,
        world_size: int,
    ):
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
                dict(
                    model_config=model_config,
                    cache_config=cache_config,
                    in_que=in_que,
                    out_que=out_que,
                    world_size=world_size,
                ),
            ),
            nprocs=world_size,
            join=False,
            daemon=True,
        )

    def create_instance(self, cuda_stream_id=0):
        """Create a turbomind instance.

        Args:
            cuda_stream_id(int): identity of a cuda stream
        Returns:
            EngineInstance: an instance of turbomind
        """
        return EngineInstance(self)

    def add_session(self, session_id: int):
        """Add new session."""
        self.scheduler.add_session(session_id)

    def add_message(self, message: SchedulerSequence):
        """Add new message."""
        self.scheduler.add_sequence(message)

    def _make_inputs(self,
                     messages: List[SchedulerSequence],
                     device: str = 'cuda'):
        """create model inputs from messages.

        Args:
            messages (List[SchedulerSequence]): The input messages.
            device (str): Device name.
        """
        history_lengths = [msg.history_len for msg in messages]

        token_ids = [msg.token_ids for msg in messages]

        if isinstance(token_ids[0], int):
            token_ids = [token_ids]

        seq_length = [len(tokens) for tokens in token_ids]
        q_start_loc = torch.tensor([0] + seq_length).cumsum(0)[:-1].to(device)
        max_seq_len = max(seq_length)

        input_ids = list(itertools.chain(*token_ids))
        input_ids = torch.tensor(input_ids).to(device)

        attention_mask = torch.tensor([
            seq_len * [1] + (max_seq_len - seq_len) * [0]
            for seq_len in seq_length
        ]).to(device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids += position_ids.new_tensor(history_lengths).unsqueeze(-1)
        seq_length = torch.tensor(seq_length).to(device)

        block_tables = self.scheduler.get_block_tables(messages)
        block_offsets = [[block.block_id for block in block_table]
                         for block_table in block_tables]

        # add batch dim [bs=1, seq_len]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        return dict(
            input_ids=input_ids,
            seq_length=seq_length,
            attention_mask=attention_mask,
            block_offsets=block_offsets,
            position_ids=position_ids,
            q_start_loc=q_start_loc,
        )

    def stop_session(self, session_id: int):
        """stop session."""
        self.scheduler.stop_session(session_id)
        self.scheduler.update()

    def end_session(self, session_id: int):
        """end session."""
        self.scheduler.end_session(session_id)
        self.scheduler.update()

    def _stoping_criteria(self, msg: SchedulerSequence, next_token_id: int):
        """Check if the message should stop.

        Args:
            msg (SchedulerSequence): The input message.
            next_token_id (int): The next token id from inference result.

        Returns:
            bool: Weither the message should be stopped.
        """
        # check eof
        sampling_param = msg.sampling_param
        if not sampling_param.ignore_eos:
            if next_token_id == self.model_config.eos_token_id:
                return True

        # check stop words
        if (sampling_param.stop_words is not None
                and next_token_id in sampling_param.stop_words):
            return True

        # check request_len
        if msg.remain_output_len <= 0:
            return True

        # check session len
        session_len = sum(block.num_tokens for block in msg.logical_blocks)
        if session_len >= self.scheduler_config.max_session_len:
            return True

        return False

    def _model_forward(self, inputs: Dict, swap_in_map: Dict[int, int],
                       swap_out_map: Dict[int, int]):
        """model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
            swap_in_map (Dict[int, int]): Cache maps to swap in.
            swap_out_map (Dict[int, int]): Cache maps to swap out.
        """
        if self.tp == 1:
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

            with torch.no_grad():
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
                return output['logits']

        else:
            with torch.no_grad():
                self.tp_model_in_que.put((inputs, swap_in_map, swap_out_map))

            ret_code, output = self.tp_model_out_que.get()
            if ret_code != 0:
                logger.error('tp forward failed.')
                exit()

            return output

    def step(self, return_logits=False):
        """one step inference. Used to perform streaming chat.

        Args:
            return_logits (bool): Weither to return the output logits.

        Returns:
            Dict[int, InferOutput]: The output of each session.
        """

        # schedule
        schedule_output = self.scheduler.schedule()

        running: List[SchedulerSequence] = schedule_output.running
        swap_in_map = schedule_output.swap_in_map
        swap_out_map = schedule_output.swap_out_map
        if len(running) == 0:
            return dict()

        session_ids = [msg.session_id for msg in running]
        req_ids = [msg.meta['req_id'] for msg in running]
        history_lengths = [msg.history_len for msg in running]

        # make batch
        inputs = self._make_inputs(running)
        inputs['history_lengths'] = history_lengths

        # inference
        logits = self._model_forward(inputs, swap_in_map, swap_out_map)

        logits = logits[0]  # [bs, seq, prob] -> [seq, prob]

        # gather output
        sampling_params: List[SamplingParam] = [
            msg.sampling_param for msg in running
        ]
        seq_length = inputs['seq_length']
        accum_seq_length = inputs['seq_length'].cumsum(0)
        logits = logits.cuda()
        split_logits = [
            logits[x - y:x] for x, y in zip(accum_seq_length, seq_length)
        ]

        next_token_ids = []
        for msg, logit, param in zip(running, split_logits, sampling_params):
            input_ids = torch.tensor(msg.token_ids)
            logits_processor = LogitsProcessorList([
                TopKLogitsWarper(param.top_k),
                TopPLogitsWarper(param.top_p),
                TemperatureLogitsWarper(param.temperature),
            ])
            logit = logits_processor(input_ids, logit)
            logit = logit.reshape([-1, logit.shape[-1]])
            next_token_ids.append(logit[-1].argmax())

        # update scheduler
        for token, msg in zip(next_token_ids, running):
            msg.update_token_ids(token)
            msg.remain_output_len -= 1
            if self._stoping_criteria(msg, token):
                msg.status = MessageStatus.STOPPED
        self.scheduler.update()

        outputs: Dict[int, InferOutput] = dict()
        for idx in range(len(session_ids)):
            session_id = session_ids[idx]
            msg = running[idx]
            out = InferOutput(
                session_id=session_id,
                req_id=req_ids[idx],
                finish=(msg.status == MessageStatus.STOPPED),
                token_ids=[next_token_ids[idx]],
            )
            outputs[session_id] = out

        if return_logits:
            for idx in range(len(session_ids)):
                outputs[session_ids[idx]].logits = split_logits[idx]
        return outputs

    def infer(self, return_logits: bool = False):
        """Perform inference until all message stopped.

        Args:
            return_logits (bool): Weither to return the output logits.

        Returns:
            Dict[int, InferOutput]: The output of each session.
        """
        ret_tokens: Dict[int, InferOutput] = dict()
        while self.scheduler.has_unfinished():
            outputs = self.step(return_logits)
            for session_id, out in outputs.items():
                if session_id not in ret_tokens:
                    # TODO: check if req_id is different
                    ret_tokens[session_id] = InferOutput(session_id=session_id,
                                                         req_id=out.req_id,
                                                         token_ids=[])

                ret_tokens[session_id].token_ids += out.token_ids

                if return_logits:
                    if ret_tokens[session_id].logits is None:
                        ret_tokens[session_id].logits = []
                    ret_tokens[session_id].logits.append(out.logits)

        if return_logits:
            for out in ret_tokens.values():
                out.logits = torch.cat(out.logits, dim=0)

        return ret_tokens

    def decode(self, prompt_token_ids: List[List[int]]):
        """Perform one step inference and get logits.

        Args:
            prompt_token_ids (List[List[int]]): Input prompts.

        Returns:
            List[Tensor]: The logits.
        """
        assert not self.scheduler.has_unfinished()

        if len(self.scheduler.sessions) > 0:
            logger.warning(
                'Unreleased session might leads to low performance.')

        session_id = 1
        sessions: List[SchedulerSession] = []
        while len(sessions) < len(prompt_token_ids):
            while session_id in self.scheduler.sessions:
                session_id += 1
            sess = SchedulerSession(session_id)
            sessions.append(sess)
            self.add_session(sess)

        msgs: List[SchedulerSequence] = []
        for token_ids, sess in zip(prompt_token_ids, sessions):
            msg = sess.add_sequence(token_ids=token_ids)
            msgs.append(msg)
            self.scheduler.add_sequence(msg)

        outputs = self.step(True)

        logits = dict((k, out.logits) for k, out in outputs.items())

        for sess in sessions:
            self.end_session(sess.session_id)

        split_logits = [logits[sess.session_id] for sess in sessions]
        pad_sequence = torch.nn.utils.rnn.pad_sequence
        logits = pad_sequence(split_logits, True)

        return logits

    def loop(self):
        """Main loop of the engine.

        Each engine instance would communicate with the engine by queue.
        """
        out_ques: Dict[int, Queue] = dict()

        def _session_exist(
            session_id: int,
            req: Request,
            resp_if_exist: bool = False,
            resp_if_not_exist: bool = False,
        ):
            if session_id in self.scheduler.sessions:
                if resp_if_exist:
                    req.resp.put(
                        Response(ResponseType.REPEAT_ERROR, req_id=req.req_id))
                return True
            else:
                if resp_if_not_exist:
                    req.resp.put(
                        Response(ResponseType.NOT_EXIST_ERROR,
                                 req_id=req.req_id))
                return False

        while True:
            # get all requests
            num_requests = self.requests.qsize()

            if num_requests == 0 and not self.scheduler.has_unfinished():
                time.sleep(1)
                continue

            reqs: List[Request] = []
            tmp = num_requests
            while tmp:
                tmp -= 1
                reqs.append(self.requests.get())

            # gather requests
            reqs_by_type: Dict[RequestType, Request] = dict(
                (t, []) for t in RequestType)
            for req in reqs:
                reqs_by_type[req.type].append(req)

            # stop engine
            if len(reqs_by_type[RequestType.STOP_ENGINE]) > 0:
                for v in self.out_ques.values():
                    v.put(Response(ResponseType.ENGINE_STOP_ERROR))
                return

            # stop session
            stop_session_reqs: List[Request] = reqs_by_type[
                RequestType.STOP_SESSION]
            for req in stop_session_reqs:
                session_id = req.data['session_id']
                if _session_exist(session_id, req, resp_if_not_exist=True):
                    self.stop_session(session_id)

            # end session
            end_session_reqs: List[Request] = reqs_by_type[
                RequestType.END_SESSION]
            for req in end_session_reqs:
                session_id = req.data['session_id']
                if _session_exist(session_id, req, resp_if_not_exist=True):
                    self.end_session(session_id)
                    out_ques.pop(session_id)

            # add session
            add_session_reqs: List[Request] = reqs_by_type[
                RequestType.ADD_SESSION]
            for req in add_session_reqs:
                session_id = req.data['session_id']
                if not _session_exist(session_id, req, resp_if_exist=True):
                    self.add_session(session_id)
                    out_ques[session_id] = req.resp

            # add message
            add_msg_reqs: List[Request] = reqs_by_type[RequestType.ADD_MESSAGE]
            for req in add_msg_reqs:
                if not _session_exist(session_id, req, resp_if_not_exist=True):
                    continue
                session_id = req.data['session_id']
                sess = self.scheduler.sessions[session_id]
                # TODO: support 1 session n sequence
                if len(sess.sequences) == 0:
                    sess.add_sequence(
                        req.data['token_ids'],
                        max_output_len=req.data['max_request_output_len'],
                        sampling_param=req.data['sampling_param'])
                    msg = next(iter(sess.sequences.values()))
                    self.add_message(msg)
                else:
                    msg = next(iter(sess.sequences.values()))
                    msg.update_token_ids(req.data['token_ids'])
                    msg.status = MessageStatus.WAITING
                    self.scheduler.update()

                msg.meta = dict(req_id=req.data['req_id'])

            # forward
            step_tokens: Dict[int, InferOutput] = self.step()

            for session_id, out in step_tokens.items():
                if out.finish:
                    resp_type = ResponseType.FINISH
                else:
                    resp_type = ResponseType.SUCCESS
                out_ques[session_id].put(
                    Response(
                        type=resp_type,
                        req_id=out.req_id,
                        data=dict(token_ids=out.token_ids),
                    ))


class EngineInstance:
    """Instance of TurboMind.

    Args:
        engine (Engine): engine
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self.response = Queue()
        self.req_count = 0
        self.owned_sessions: List[int] = list()

    def __del__(self):
        """Destructor."""
        for session_id in self.owned_sessions:
            self.end(session_id)

    def _send_req(self, req_type: RequestType, data: Any):
        """Send request to engine.

        Args:
            req_type (RequestType): The request type to send.
            data (Any): The data of the request.
        """
        self.engine.requests.put(
            Request(type=req_type,
                    resp=self.response,
                    req_id=self.req_count,
                    data=data))
        self.req_count += 1

    def _try_add_session(self, session_id: int):
        """Add new session.

        Args:
            session_id (int): The session id to add.
        """
        if session_id not in self.owned_sessions:
            self._send_req(RequestType.ADD_SESSION,
                           dict(session_id=session_id))
            self.owned_sessions.append(session_id)

    def stream_infer(
            self,
            session_id: int,
            prompt_token_ids: List[int] = None,
            request_output_len: int = None,
            step: int = 0,
            sampling_param: SamplingParam = SamplingParam(),
    ):
        """Send stream inference request.

        Args:
            session_id (int): The session id.
            prompt_token_ids (List[int]): The input token ids.
            request_output_len (int): The max output length of this request.
            step (int): No use for now.
            sampling_param (SamplingParam): The sampling param of the output.

        Yields:
            int: Error flags. 0 if success.
            List[int]: The streaming output tokens.
            int: The number of the output tokens.
        """
        self._try_add_session(session_id)
        req_id = self.req_count
        msg = dict(
            token_ids=prompt_token_ids,
            session_id=session_id,
            max_request_output_len=request_output_len,
            req_id=req_id,
            sampling_param=sampling_param,
        )
        self._send_req(RequestType.ADD_MESSAGE, msg)

        token_ids = []
        while True:
            if not self.engine.loop_threads.is_alive():
                yield (1, [], 0)
                break

            resp: Response = self.response.get()
            if resp.req_id != req_id:
                continue
            if resp.type == ResponseType.SUCCESS:
                token_ids += resp.data['token_ids']
                yield (0, token_ids, len(token_ids))
            elif resp.type == ResponseType.FINISH:
                token_ids += resp.data['token_ids']
                yield (0, token_ids, len(token_ids))
                break
            else:
                yield (1, [], 0)
                break

    def infer(
            self,
            session_id: int,
            prompt_token_ids: List[int] = None,
            request_output_len: int = None,
            step: int = 0,
            sampling_param: SamplingParam = SamplingParam(),
    ):
        """Send inference request.

        Args:
            session_id (int): The session id.
            prompt_token_ids (List[int]): The input token ids.
            request_output_len (int): The max output length of this request.
            step (int): No use for now.
            sampling_param (SamplingParam): The sampling param of the output.

        Returns:
            int: Error flags. 0 if success.
            List[int]: The streaming output tokens.
            int: The number of the output tokens.
        """
        self._try_add_session(session_id)
        req_id = self.req_count
        msg = dict(
            token_ids=prompt_token_ids,
            session_id=session_id,
            max_request_output_len=request_output_len,
            req_id=req_id,
            sampling_param=sampling_param,
        )
        self._send_req(RequestType.ADD_MESSAGE, msg)

        token_ids = []
        status = 0
        while True:
            if not self.engine.loop_threads.is_alive():
                status = 1
                break

            resp: Response = self.response.get()
            if resp.req_id != req_id:
                continue
            if resp.type == ResponseType.SUCCESS:
                token_ids += resp.data['token_ids']
            elif resp.type == ResponseType.FINISH:
                token_ids += resp.data['token_ids']
                break
            else:
                status = 1
                break

        return (status, token_ids, len(token_ids))

    def end(self, session_id: int):
        """End the given session."""
        self._send_req(RequestType.END_SESSION, dict(session_id=session_id))
        self.owned_sessions.remove(session_id)

    def cancel(self, session_id: int):
        """Stop current streaming inference."""
        self._send_req(RequestType.STOP_SESSION, dict(session_id=session_id))

    def decode(self, prompt_token_ids: List[List[int]]):
        """Return logits of context decoding.

        Args:
            prompt_token_ids: token ids of a batch prompts.

        Returns:
            logits (numpy.ndarray) with shape
                [batch, n_max_token_of_the_batch, vocab_size]
        """
        return self.engine.decode(prompt_token_ids)
