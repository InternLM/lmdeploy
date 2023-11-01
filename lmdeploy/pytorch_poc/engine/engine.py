# Copyright (c) OpenMMLab. All rights reserved.
import enum
import time
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any, Dict, List

import torch
from transformers import AutoConfig

from lmdeploy.pytorch_poc.config import (CacheConfig, ModelConfig,
                                         SchedulerConfig)
from lmdeploy.pytorch_poc.engine.model_agent import (BaseModelAgent,
                                                     TPModelAgent)
from lmdeploy.pytorch_poc.messages import (MessageStatus, SamplingParam,
                                           SchedulerSequence, SchedulerSession)
from lmdeploy.pytorch_poc.paging import Scheduler
from lmdeploy.utils import get_logger

from .logits_process import FusedLogitsProcessor

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


def _get_torch_dtype(config: Any, default: str = 'float16'):
    """Get the torch dtype from the model config.

    Args:
        config: Config of the hf model.
        default (str): default device type.
    """
    torch_dtype = getattr(config, 'torch_dtype', default)
    return eval(f'torch.{torch_dtype}')


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
        self.gpu_count = tp
        hf_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code)
        torch_dtype = _get_torch_dtype(hf_config)
        self.torch_dtype = torch_dtype

        if scheduler_config is None:
            scheduler_config = SchedulerConfig(max_batches=64,
                                               max_session_len=4096,
                                               max_request_output_len=512)
        if cache_config is None:
            cache_config = CacheConfig(block_size=128,
                                       num_cpu_blocks=0,
                                       num_gpu_blocks=0)
        if 'falcon' in model_path:
            if hf_config.new_decoder_architecture:
                # 40b-instruct, GQA
                kv_dim = hf_config.hidden_size // hf_config.num_attention_heads
                kv_dim *= hf_config.num_kv_heads
                kv_head = hf_config.num_kv_heads
            if hf_config.multi_query:
                # 7b-instruct, MQA
                kv_dim = hf_config.hidden_size // hf_config.num_attention_heads
                kv_head = 1
            else:
                # rw-1b, MHA
                kv_dim = hf_config.hidden_size
                kv_head = hf_config.num_attention_heads
            model_config = ModelConfig(
                kv_dim,
                hf_config.num_hidden_layers,
                kv_head,
                bos_token_id=hf_config.bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                dtype=torch_dtype,
                multi_query_attention=hf_config.multi_query)
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

        if tp == 1:
            self.model_agent = BaseModelAgent(
                model_path,
                model_config=model_config,
                cache_config=cache_config,
                trust_remote_code=trust_remote_code)
        else:
            self.model_agent = TPModelAgent(
                model_path,
                model_config=model_config,
                cache_config=cache_config,
                world_size=tp,
                trust_remote_code=trust_remote_code)

        cache_config = self.model_agent.cache_config
        self.scheduler = Scheduler(scheduler_config, cache_config)

        self.requests = Queue(scheduler_config.max_batches)
        self.response = Queue(scheduler_config.max_batches)
        self.req_count = 0
        self.owned_sessions = []

        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.model_config = model_config
        self.session_len = scheduler_config.max_session_len
        self.stream = torch.cuda.Stream()

        # create main thread
        loop_threads = Thread(target=self.loop, daemon=True)
        loop_threads.start()
        self.loop_threads = loop_threads

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

        batch_size = len(messages)
        input_ids = [ids for ids in token_ids]
        input_ids = torch.cat(input_ids).to(device)

        is_decoding = input_ids.size(0) == batch_size
        if not is_decoding:
            seq_length = [tokens.size(0) for tokens in token_ids]
            max_seq_len = max(seq_length)
            q_start_loc = torch.tensor([0] +
                                       seq_length).cumsum(0)[:-1].to(device)

            attention_mask = torch.tensor([
                seq_len * [1] + (max_seq_len - seq_len) * [0]
                for seq_len in seq_length
            ]).to(device)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids += position_ids.new_tensor(history_lengths).unsqueeze(
                -1)
            seq_length = torch.tensor(seq_length).to(device)
        else:
            q_start_loc = torch.arange(batch_size, device=device)
            attention_mask = torch.ones(batch_size,
                                        1,
                                        dtype=torch.long,
                                        device=device)
            position_ids = q_start_loc.new_tensor(history_lengths).unsqueeze(
                -1)
            seq_length = torch.ones(batch_size,
                                    dtype=torch.long,
                                    device=device)

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
        return self.model_agent.forward(inputs,
                                        swap_in_map=swap_in_map,
                                        swap_out_map=swap_out_map)

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
        logits = logits.cuda()

        # gather output
        # most step share the same sampling parameters
        sampling_params: List[SamplingParam] = [
            msg.sampling_param for msg in running
        ]
        grouped_params = dict()
        next_token_ids = [None] * len(sampling_params)
        for i, p in enumerate(sampling_params):
            key = (p.top_k, p.top_p, p.temperature, p.repetition_penalty)
            grouped_params.setdefault(key, list())
            grouped_params[key].append(i)

        is_decoding = inputs['input_ids'].numel(
        ) == inputs['seq_length'].numel()
        # TODO: support repetition_penalty
        if not is_decoding:
            seq_length = inputs['seq_length']
            last_idx = seq_length.cumsum(-1) - 1
            split_logits = logits[last_idx, :]
            for param, idx in grouped_params.items():
                top_k, top_p, temperature, _ = param
                logits_processor = FusedLogitsProcessor(
                    SamplingParam(
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                    ))
                input_ids = None
                new_logits = split_logits[idx]
                new_logits = logits_processor(input_ids, new_logits)
                argmax_ids = new_logits.argmax(-1).cpu()
                for i, next_ids in zip(idx, argmax_ids):
                    next_token_ids[i] = next_ids
        else:
            # most step share the same sampling parameters
            split_logits = logits

            for param, idx in grouped_params.items():
                top_k, top_p, temperature, _ = param
                logits_processor = FusedLogitsProcessor(
                    SamplingParam(
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                    ))
                input_ids = inputs['input_ids'].reshape(-1, 1)
                new_logits = logits[idx]
                new_logits = logits_processor(input_ids, new_logits)
                argmax_ids = new_logits.argmax(-1).cpu()
                for i, next_ids in zip(idx, argmax_ids):
                    next_token_ids[i] = next_ids

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

    def _send_reqs(self, req_types: List[RequestType], data: List[Any]):
        """Send multiple request to engine.

        Args:
            req_types (List[RequestType]): The request type to send.
            data (List[Any]): The data of the request.
        """
        num_reqs = len(req_types)
        assert num_reqs == len(data)
        reqs = [
            Request(type=req_type,
                    resp=self.response,
                    req_id=self.req_count + idx,
                    data=d)
            for idx, req_type, d in zip(range(num_reqs), req_types, data)
        ]
        self.requests.put(reqs)
        self.req_count += num_reqs

    def _send_req(self, req_type: RequestType, data: Any):
        """Send request to engine.

        Args:
            req_type (RequestType): The request type to send.
            data (Any): The data of the request.
        """
        self._send_reqs([req_type], [data])

    def _try_add_session(self, session_id: int):
        """Add new session.

        Args:
            session_id (int): The session id to add.
        """
        if session_id not in self.owned_sessions:
            self._send_req(RequestType.ADD_SESSION,
                           dict(session_id=session_id))
            self.owned_sessions.append(session_id)

    def batched_infer(
            self,
            session_ids: List[int],
            token_ids: List[List[int]] = None,
            request_output_len: int = 512,
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

        batch_size = len(token_ids)
        assert len(session_ids) == batch_size

        req_ids = []
        add_msgs = []
        for session_id, token_id in zip(session_ids, token_ids):
            self._try_add_session(session_id)
            req_id = self.req_count
            req_ids.append(req_id)
            msg = dict(
                token_ids=token_id,
                session_id=session_id,
                max_request_output_len=request_output_len,
                req_id=req_id,
                sampling_param=sampling_param,
            )
            add_msgs.append(msg)
        self._send_reqs([RequestType.ADD_MESSAGE] * batch_size, add_msgs)

        req_idx_map = dict(zip(req_ids, range(len(req_ids))))
        output_token_ids = [list() for _ in req_ids]
        status = 0
        finish_count = batch_size
        while True:
            if not self.loop_threads.is_alive():
                status = 1
                break

            resp: Response = self.response.get()
            if resp.req_id not in req_ids:
                continue
            idx = req_idx_map[resp.req_id]
            token_ids = output_token_ids[idx]
            if resp.type == ResponseType.SUCCESS:
                token_ids += resp.data['token_ids']
            elif resp.type == ResponseType.FINISH:
                token_ids += resp.data['token_ids']
                finish_count -= 1
            else:
                status = 1
                break

            if finish_count == 0:
                break

        output_token_len = [len(token_ids) for token_ids in output_token_ids]
        return (status, output_token_ids, output_token_len)

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
                elem = self.requests.get()
                if isinstance(elem, Request):
                    elem = [elem]
                reqs += elem

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
                    msg.remain_output_len = req.data['max_request_output_len']
                    msg.sampling_param = req.data['sampling_param']
                    msg.status = MessageStatus.WAITING
                    self.scheduler.update()

                msg.meta = dict(req_id=req.data['req_id'])

            # forward
            with torch.no_grad():
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

    def stream_infer(self,
                     session_id: int,
                     input_ids: List[int] = None,
                     request_output_len: int = None,
                     step: int = 0,
                     sampling_param: SamplingParam = SamplingParam(),
                     **kwargs):
        """Send stream inference request.

        Args:
            session_id (int): The session id.
            input_ids (List[int]): The input token ids.
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
            token_ids=input_ids,
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
