# Copyright (c) OpenMMLab. All rights reserved.
import enum
import itertools
import json
import os
import os.path as osp
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

from lmdeploy.pytorch.accel import LoadNoInit
from lmdeploy.pytorch_poc.config import (CacheConfig, ModelConfig,
                                         SchedulerConfig)
from lmdeploy.pytorch_poc.dist_utils import try_to_local
from lmdeploy.pytorch_poc.messages import (MessageStatus, SamplingParam,
                                           SchedulerMessage, SchedulerSession)
from lmdeploy.pytorch_poc.paging import BlockTable, Scheduler
from lmdeploy.pytorch_poc.patch import patch
from lmdeploy.pytorch_poc.utils import get_gpu_memory
from lmdeploy.utils import get_logger

from .cache_engine import CacheEngine

logger = get_logger('lmdeploy')


class RequestType(enum.Enum):
    ADD_SESSION = enum.auto()
    ADD_MESSAGE = enum.auto()
    STOP_SESSION = enum.auto()
    END_SESSION = enum.auto()
    STOP_ENGINE = enum.auto()
    RESUME_ENGINE = enum.auto()


class ResponseType(enum.Enum):
    SUCCESS = enum.auto()
    FINISH = enum.auto()
    ENGINE_STOP_ERROR = enum.auto()
    REPEAT_ERROR = enum.auto()
    NOT_EXIST_ERROR = enum.auto()


@dataclass
class Request:
    type: RequestType
    resp: Queue
    req_id: int
    data: Any = None


@dataclass
class Response:
    type: ResponseType
    req_id: int
    data: Any = None
    err_msg: str = ''


@dataclass
class InferOutput:
    session_id: int
    token_ids: List[int]
    req_id: int = 0
    finish: bool = False
    logits: torch.Tensor = None


class ModelContext:

    def __init__(self,
                 block_tables: List[BlockTable],
                 history_lengths: List[int],
                 world_size: int = 1,
                 device='cuda'):
        self.block_tables = block_tables
        self.history_lengths = history_lengths
        self.world_size = world_size

        # make block offsets
        block_offsets = [[block.block_id for block in block_table]
                         for block_table in self.block_tables]

        # padding zero
        pad_sequence = torch.nn.utils.rnn.pad_sequence
        block_offsets = [
            torch.tensor(offset, device=device) for offset in block_offsets
        ]
        block_offsets = pad_sequence(block_offsets, True)
        self.block_offsets = block_offsets

    def get_block_offsets(self):
        return [[block.block_id for block in block_table]
                for block_table in self.block_tables]

    def fill_cache(
        self,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
        start_loc: torch.Tensor,
        seq_length: torch.Tensor,
        k_caches: torch.Tensor,
        v_caches: torch.Tensor,
    ):
        k_states = try_to_local(k_states)
        v_states = try_to_local(v_states)
        start_loc = try_to_local(start_loc)
        seq_length = try_to_local(seq_length)
        k_caches = try_to_local(k_caches)
        v_caches = try_to_local(v_caches)

        block_size = k_caches.size(1)
        block_offsets = self.get_block_offsets()

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


def _update_cache_config(model_config: ModelConfig, cache_config: CacheConfig):
    GPU_MEM_PERCENT = 0.7
    SWAP_SPACE = 4 * (1 << 30)
    gpu_mem = get_gpu_memory() * GPU_MEM_PERCENT
    cpu_mem = SWAP_SPACE
    cache_block_size = CacheEngine.get_cache_block_size(
        cache_config.block_size, model_config)
    if cache_config.num_cpu_blocks == 0:
        cache_config.num_cpu_blocks = int(cpu_mem / cache_block_size)
    if cache_config.num_gpu_blocks == 0:
        cache_config.num_gpu_blocks = int(gpu_mem / cache_block_size)


def _tp_model_loop(rank: int, model_path: str, extra_args: List[str],
                   model_config: ModelConfig, cache_config: CacheConfig,
                   in_que: mp.Queue, out_que: mp.Queue, world_size: int):
    from accelerate import init_empty_weights
    device_mesh = DeviceMesh('cuda', list(range(world_size)))

    error_code = 0
    error_type = None

    try:
        config = AutoConfig.from_pretrained(model_path)
        torch_dtype = getattr(config, 'torch_dtype', 'float16')
        torch_dtype = eval(f'torch.{torch_dtype}')
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config,
                                                     torch_dtype=torch_dtype)

        torch_model_json_path = osp.join(model_path,
                                         'pytorch_model.bin.index.json')
        with open(torch_model_json_path, mode='r') as f:
            torch_model_json = json.load(f)

        weight_map = torch_model_json['weight_map']

        checkpoints = list(set(weight_map.values()))
        checkpoints = [osp.join(model_path, ckpt) for ckpt in checkpoints]
        patched_model = patch(model,
                              extra_args=extra_args,
                              rank=rank,
                              world_size=world_size,
                              checkpoints=checkpoints)

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
        ]

        if rank == 0:
            inputs, swap_in_map, swap_out_map = in_que.get()
            input_tensors = dict((name, t) for name, t in inputs.items()
                                 if name in input_tensor_names)
            input_metas = dict((name, (t.shape, t.dtype))
                               for name, t in input_tensors.items())
            input_metas['block_tables'] = inputs['block_tables']
            input_metas['history_lengths'] = inputs['history_lengths']

            objs = [input_metas, swap_in_map, swap_out_map]

        else:
            objs = [None, None, None]

        # broadcast input shapes and dtypes
        dist.broadcast_object_list(objs)

        if rank != 0:
            input_metas = objs[0]
            input_tensors = dict(
                (name, torch.empty(meta[0], dtype=meta[1]).cuda(rank))
                for name, meta in input_metas.items()
                if name in input_tensor_names)

        updated_inputs = dict()
        for name, t in input_tensors.items():
            updated_inputs[name] = distribute_tensor(t,
                                                     device_mesh=device_mesh,
                                                     placements=[Replicate()
                                                                 ]).to_local()

        inputs = updated_inputs
        inputs['block_tables'] = input_metas['block_tables']
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
                context=ModelContext(block_tables=inputs['block_tables'],
                                     history_lengths=inputs['history_lengths'],
                                     world_size=world_size))

            if rank == 0:
                out_que.put((0, output['logits']))


def _error_process(rank: int,
                   world_size: int,
                   func: Callable,
                   args: List = None,
                   kwargs: Dict = None):
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        with torch.cuda.device(rank):
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

    def __init__(self,
                 model_path: str,
                 scheduler_config: SchedulerConfig = None,
                 cache_config: CacheConfig = None,
                 tp: int = 1) -> None:

        self.tp = tp
        hf_config = AutoConfig.from_pretrained(model_path,
                                               trust_remote_code=True)
        torch_dtype = getattr(hf_config, 'torch_dtype', 'float16')
        torch_dtype = eval(f'torch.{torch_dtype}')
        self.torch_dtype = torch_dtype

        if scheduler_config is None:
            scheduler_config = SchedulerConfig(max_batches=64,
                                               max_session_len=2048,
                                               max_request_output_len=512)
        if cache_config is None:
            cache_config = CacheConfig(block_size=64,
                                       num_cpu_blocks=0,
                                       num_gpu_blocks=0)
        model_config = ModelConfig(hf_config.hidden_size,
                                   hf_config.num_hidden_layers,
                                   hf_config.num_attention_heads,
                                   bos_token_id=hf_config.bos_token_id,
                                   eos_token_id=hf_config.eos_token_id,
                                   dtype=torch_dtype)

        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.model_config = model_config
        self.session_len = scheduler_config.max_session_len

        if tp == 1:
            with LoadNoInit():
                hf_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True)
                hf_model.eval()

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

            self.patch_model_tp(model_path,
                                ['context', 'use_origin', 'q_seq_info'],
                                model_config=model_config,
                                cache_config=cache_config,
                                in_que=self.tp_model_in_que,
                                out_que=self.tp_model_out_que,
                                world_size=tp)

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

    def patch_model_tp(self, model_path: str, extra_args: List[str],
                       model_config: ModelConfig, cache_config: CacheConfig,
                       in_que: mp.Queue, out_que: mp.Queue, world_size: int):
        self.mp_context = mp.spawn(_error_process,
                                   args=(world_size, _tp_model_loop,
                                         (model_path, extra_args),
                                         dict(model_config=model_config,
                                              cache_config=cache_config,
                                              in_que=in_que,
                                              out_que=out_que,
                                              world_size=world_size)),
                                   nprocs=world_size,
                                   join=False,
                                   daemon=True)

    def create_instance(self, cuda_stream_id=0):
        """Create a turbomind instance.

        Args:
            cuda_stream_id(int): identity of a cuda stream
        Returns:
            EngineInstance: an instance of turbomind
        """
        return EngineInstance(self)

    def add_session(self, session: SchedulerSession):
        self.scheduler.add_session(session)

    def add_message(self, message: SchedulerMessage):
        self.scheduler.add_message(message)

    def _make_inputs(self, messages: List[SchedulerMessage], device='cuda'):

        sessions = self.scheduler.get_sessions(messages)
        history_lengths = [sess.history_length for sess in sessions]

        token_ids = [msg.token_ids for msg in messages]

        if isinstance(token_ids[0], int):
            token_ids = [token_ids]

        seq_length = [len(tokens) for tokens in token_ids]
        max_seq_len = max(seq_length)

        input_ids = list(itertools.chain(*token_ids))
        input_ids = torch.tensor(input_ids).to(device)

        q_start_loc = torch.tensor([0] + seq_length[:-1]).to(device)
        q_seq_length = torch.tensor(seq_length).to(device)

        attention_mask = torch.tensor([
            seq_len * [1] + (max_seq_len - seq_len) * [0]
            for seq_len in seq_length
        ]).to(device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids += position_ids.new_tensor(history_lengths).unsqueeze(-1)
        seq_length = torch.tensor(seq_length).to(device)

        block_tables = self.scheduler.get_block_tables(messages)

        # add batch dim [bs=1, seq_len]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        return dict(input_ids=input_ids,
                    seq_length=seq_length,
                    attention_mask=attention_mask,
                    block_tables=block_tables,
                    position_ids=position_ids,
                    q_seq_info=(q_start_loc, q_seq_length))

    def stop_session(self, session_id: int):
        self.scheduler.stop_session(session_id)
        self.scheduler.update()

    def end_session(self, session_id: int):
        self.scheduler.end_session(session_id)
        self.scheduler.update()

    def _stoping_criteria(self, msg: SchedulerMessage, next_token_id: int):
        # check eof
        sampling_param = msg.sampling_param
        if not sampling_param.ignore_eos:
            if next_token_id == self.model_config.eos_token_id:
                return True

        if (sampling_param.stop_words is not None
                and next_token_id in sampling_param.stop_words):
            return True

        # check request_len
        if (msg.request_output_len >= msg.max_request_output_len):
            return True

        # check session len
        session = self.scheduler.sessions[msg.session_id]
        session_len = sum(block.num_tokens for block in session.logical_blocks)
        if session_len >= self.scheduler_config.max_session_len:
            return True

        return False

    def _model_forward(self, inputs: Dict, swap_in_map: Dict[int, int],
                       swap_out_map: Dict[int, int]):
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
                        block_tables=inputs['block_tables'],
                        history_lengths=inputs['history_lengths']),
                    q_seq_info=inputs['q_seq_info'])
                return output['logits']
        else:
            self.tp_model_in_que.put((inputs, swap_in_map, swap_out_map))

            ret_code, output = self.tp_model_out_que.get()
            if ret_code != 0:
                logger.error('tp forward failed.')
                exit()

            return output

    def step(self, return_logits=False):
        # TODO: cache manage

        # schedule
        schedule_output = self.scheduler.schedule()

        running: List[SchedulerMessage] = schedule_output.running
        swap_in_map = schedule_output.swap_in_map
        swap_out_map = schedule_output.swap_out_map
        if len(running) == 0:
            return dict()

        sessions = self.scheduler.get_sessions(running)
        session_ids = [msg.session_id for msg in running]
        req_ids = [msg.req_id for msg in running]
        history_lengths = [sess.history_length for sess in sessions]

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
            msg.token_ids = [token]
            msg.request_output_len += 1
            if self._stoping_criteria(msg, token):
                self.stop_session(msg.session_id)
        self.scheduler.update()

        outputs: Dict[int, InferOutput] = dict()
        for idx in range(len(session_ids)):
            session_id = session_ids[idx]
            msg = running[idx]
            out = InferOutput(session_id=session_id,
                              req_id=req_ids[idx],
                              finish=(msg.status == MessageStatus.STOPPED),
                              token_ids=[next_token_ids[idx]])
            outputs[session_id] = out

        if return_logits:
            for idx in range(len(session_ids)):
                outputs[session_ids[idx]].logits = split_logits[idx]

        return outputs

    def infer(self, return_logits: bool = False):
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

        msgs: List[SchedulerMessage] = []
        for token_ids, sess in zip(prompt_token_ids, sessions):
            msg = SchedulerMessage(token_ids=token_ids,
                                   session_id=sess.session_id)
            msgs.append(msg)
            self.scheduler.add_message(msg)

        outputs = self.step(True)

        logits = dict((k, out.logits) for k, out in outputs.items())

        for sess in sessions:
            self.end_session(sess.session_id)

        split_logits = [logits[sess.session_id] for sess in sessions]
        pad_sequence = torch.nn.utils.rnn.pad_sequence
        logits = pad_sequence(split_logits, True)

        return logits

    def loop(self):

        out_ques: Dict[int, Queue] = dict()

        def _session_exist(session_id: int,
                           req: Request,
                           resp_if_exist: bool = False,
                           resp_if_not_exist: bool = False):
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
                session = SchedulerSession(session_id=session_id,
                                           arrive_time=time.time())
                if not _session_exist(session_id, req, resp_if_exist=True):
                    self.add_session(session)
                    out_ques[session_id] = req.resp

            # add message
            add_msg_reqs: List[Request] = reqs_by_type[RequestType.ADD_MESSAGE]
            for req in add_msg_reqs:
                msg: SchedulerMessage = SchedulerMessage(**req.data)
                session_id = msg.session_id
                if not _session_exist(session_id, req, resp_if_not_exist=True):
                    continue
                else:
                    self.add_message(msg)

            # forward
            step_tokens: Dict[int, InferOutput] = self.step()
            for session_id, out in step_tokens.items():
                resp_type = (ResponseType.FINISH
                             if out.finish else ResponseType.SUCCESS)
                out_ques[session_id].put(
                    Response(type=resp_type,
                             req_id=out.req_id,
                             data=dict(token_ids=out.token_ids)))


class EngineInstance:
    """Instance of TurboMind.

    Args:
        engine (Engine): engine
        cuda_stream_id(int): identity of a cuda stream
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self.response = Queue()
        self.req_count = 0
        self.owned_sessions: List[int] = list()

    def _send_req(self, req_type: RequestType, data: Any):
        self.engine.requests.put(
            Request(type=req_type,
                    resp=self.response,
                    req_id=self.req_count,
                    data=data))
        self.req_count += 1

    def _try_add_session(self, session_id: int):
        if session_id not in self.owned_sessions:
            self._send_req(RequestType.ADD_SESSION,
                           dict(session_id=session_id))
            self.owned_sessions.append(session_id)

    def stream_infer(self,
                     session_id: int,
                     prompt_token_ids: List[int] = None,
                     request_output_len: int = None,
                     step: int = 0,
                     sampling_param: SamplingParam = SamplingParam()):
        self._try_add_session(session_id)
        req_id = self.req_count
        msg = dict(token_ids=prompt_token_ids,
                   session_id=session_id,
                   max_request_output_len=request_output_len,
                   req_id=req_id,
                   sampling_param=sampling_param)
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

    def infer(self,
              session_id: int,
              prompt_token_ids: List[int] = None,
              request_output_len: int = None,
              step: int = 0,
              sampling_param: SamplingParam = SamplingParam()):
        self._try_add_session(session_id)
        req_id = self.req_count
        msg = dict(token_ids=prompt_token_ids,
                   session_id=session_id,
                   max_request_output_len=request_output_len,
                   req_id=req_id,
                   sampling_param=sampling_param)
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
        self._send_req(RequestType.END_SESSION, dict(session_id=session_id))
        self.owned_sessions.pop(session_id)

    def cancel(self, session_id: int):
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
