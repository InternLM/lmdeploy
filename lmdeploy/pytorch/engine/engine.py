# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import time
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any, Dict, List

import torch
from transformers import AutoConfig

from lmdeploy.messages import EngineGenerationConfig
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.utils import ResponseType, get_logger

from ..config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from ..messages import (MessageStatus, SamplingParam, SchedulerSequence,
                        SchedulerSession)
from ..paging import Scheduler
from .logits_process import FusedLogitsProcessor
from .model_agent import BaseModelAgent, ModelInputs, TPModelAgent
from .request import Request, RequestManager, RequestType, Response

logger = get_logger('lmdeploy')


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


def _check_resp(resp: Response, state: ResponseType, warning_msg: str = None):
    """check if response has state."""
    ret = resp.type == state
    if not ret and warning_msg is not None:
        logger.warning(warning_msg)
    return ret


def _check_resp_success(resp: Response, warning_msg: str = None):
    """check if response success."""
    return _check_resp(resp, ResponseType.SUCCESS, warning_msg)


def _get_torch_dtype(config: Any, default: str = 'float16'):
    """Get the torch dtype from the model config.

    Args:
        config: Config of the hf model.
        default (str): default device type.
    """
    torch_dtype = getattr(config, 'torch_dtype', default)
    return eval(f'torch.{torch_dtype}')


def _build_model_config(model_path: str, hf_config: Any):
    """build model config."""
    torch_dtype = _get_torch_dtype(hf_config)
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
        model_config = ModelConfig(kv_dim,
                                   hf_config.num_hidden_layers,
                                   kv_head,
                                   bos_token_id=hf_config.bos_token_id,
                                   eos_token_id=hf_config.eos_token_id,
                                   dtype=torch_dtype,
                                   multi_query_attention=hf_config.multi_query,
                                   json_config=hf_config.to_dict())
    elif 'chatglm' in model_path:
        model_config = ModelConfig(hf_config.hidden_size //
                                   hf_config.num_attention_heads *
                                   hf_config.multi_query_group_num,
                                   hf_config.num_layers,
                                   hf_config.multi_query_group_num,
                                   bos_token_id=hf_config.bos_token_id,
                                   eos_token_id=hf_config.eos_token_id,
                                   dtype=torch_dtype,
                                   json_config=hf_config.to_dict())
    else:
        model_config = ModelConfig(hf_config.hidden_size,
                                   hf_config.num_hidden_layers,
                                   hf_config.num_attention_heads,
                                   bos_token_id=hf_config.bos_token_id,
                                   eos_token_id=hf_config.eos_token_id,
                                   dtype=torch_dtype,
                                   json_config=hf_config.to_dict())

    return model_config


def _build_model_agent(model_path: str,
                       model_config: ModelConfig,
                       cache_config: CacheConfig,
                       trust_remote_code: bool,
                       tp: int = 1):
    """create model agent."""
    if tp == 1:
        model_agent = BaseModelAgent(model_path,
                                     model_config=model_config,
                                     cache_config=cache_config,
                                     trust_remote_code=trust_remote_code)
    else:
        model_agent = TPModelAgent(model_path,
                                   model_config=model_config,
                                   cache_config=cache_config,
                                   world_size=tp,
                                   trust_remote_code=trust_remote_code)
    return model_agent


def _tensorlize_block_offsets(block_offsets):
    """tensorlize block_offsets."""
    from torch.nn.utils.rnn import pad_sequence
    block_offsets = [torch.from_numpy(off) for off in block_offsets]
    block_offsets = pad_sequence(block_offsets, batch_first=True)
    return block_offsets


class Engine:
    """The inference engine of lmdeploy pytorch.

    Args:
        model_path (str): The hugging face model path.
        engine_config (EngineConfig): The config of the Engine.
    """

    def __init__(self,
                 model_path: str,
                 engine_config: EngineConfig,
                 trust_remote_code=True) -> None:
        self.engine_config = engine_config
        model_name = engine_config.model_name
        tp = engine_config.tp

        self.tp = tp
        self.gpu_count = tp
        self.model_name = model_name

        scheduler_config = SchedulerConfig(
            max_batches=engine_config.max_batch_size,
            max_session_len=engine_config.session_len,
            eviction_type='recompute')

        # block_size = 1 to enable unified paging
        cache_config = CacheConfig(block_size=engine_config.block_size,
                                   num_cpu_blocks=engine_config.num_cpu_blocks,
                                   num_gpu_blocks=engine_config.num_gpu_blocks)

        hf_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code)

        torch_dtype = _get_torch_dtype(hf_config)
        self.torch_dtype = torch_dtype

        model_config = _build_model_config(model_path, hf_config)

        self.model_agent = _build_model_agent(
            model_path,
            model_config=model_config,
            cache_config=cache_config,
            trust_remote_code=trust_remote_code,
            tp=tp)

        cache_config = self.model_agent.cache_config
        self.scheduler = Scheduler(scheduler_config, cache_config)

        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.model_config = model_config
        self.session_len = scheduler_config.max_session_len
        self.stream = torch.cuda.Stream()

        self._bind_request_manager()
        self.owned_sessions = []

        # create main thread
        self._start_loop()

        self._create_buffers()
        self.tokenizer = Tokenizer(model_path)

    def _create_buffers(self):
        scheduler_config = self.scheduler_config
        max_batches = scheduler_config.max_batches

        # buffers to create inputs
        self._q_start_loc_buf = torch.arange(max_batches)
        self._attention_mask_buf = torch.ones(max_batches, 1, dtype=torch.long)
        self._seq_length_buf = torch.ones(max_batches, dtype=torch.long)

    def _bind_request_manager(self):
        """bind request manager."""
        req_manager = RequestManager()
        req_manager.bind_func(RequestType.ADD_SESSION, self._on_add_session)
        req_manager.bind_func(RequestType.STOP_SESSION, self._on_stop_session)
        req_manager.bind_func(RequestType.END_SESSION, self._on_end_session)
        req_manager.bind_func(RequestType.ADD_MESSAGE, self._on_add_message)
        self.req_manager = req_manager
        self.req_sender = req_manager.build_sender()

    def _start_loop(self):
        """start loop."""
        loop_threads = Thread(target=self.loop, daemon=True)
        loop_threads.start()
        self.loop_threads = loop_threads

    def _on_add_session(self, reqs: Request, **kwargs):
        """on add session callback."""
        for req in reqs:
            session_id = req.data['session_id']
            resp_type = ResponseType.SESSION_REPEAT
            if session_id not in self.scheduler.sessions:
                self.scheduler.add_session(session_id)
                resp_type = ResponseType.SUCCESS
            self.req_manager.response(
                Response(type=resp_type,
                         sender_id=req.sender_id,
                         req_id=req.req_id))

    def _on_stop_session(self, reqs: Request, **kwargs):
        """on stop session callback."""
        for req in reqs:
            session_id = req.data['session_id']
            resp_type = ResponseType.SESSION_NOT_EXIST
            if session_id in self.scheduler.sessions:
                self.scheduler.stop_session(session_id)
                resp_type = ResponseType.SUCCESS
            self.req_manager.response(
                Response(type=resp_type,
                         sender_id=req.sender_id,
                         req_id=req.req_id))
        self.scheduler.update()

    def _on_end_session(self, reqs: Request, **kwargs):
        """on end session callback."""
        for req in reqs:
            session_id = req.data['session_id']
            resp_type = ResponseType.SESSION_NOT_EXIST
            if session_id in self.scheduler.sessions:
                self.scheduler.end_session(session_id)
                resp_type = ResponseType.SUCCESS
            self.req_manager.response(
                Response(type=resp_type,
                         sender_id=req.sender_id,
                         req_id=req.req_id))
        self.scheduler.update()

    def _on_add_message(self, reqs: Request, **kwargs):
        """on add message callback."""
        for req in reqs:
            session_id = req.data['session_id']
            if session_id not in self.scheduler.sessions:
                self.req_manager.response(
                    Response(type=ResponseType.SESSION_NOT_EXIST,
                             sender_id=req.sender_id,
                             req_id=req.req_id))
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
                self.scheduler.add_sequence(msg)
            else:
                msg = next(iter(sess.sequences.values()))
                msg.update_token_ids(req.data['token_ids'])
                msg.remain_output_len = req.data['max_request_output_len']
                msg.sampling_param = req.data['sampling_param']
                msg.status = MessageStatus.WAITING

            msg.sender_id = req.sender_id
            msg.req_id = req.req_id
        self.scheduler.update()

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
        if session_id not in self.owned_sessions:
            resp = self.req_sender.send(RequestType.ADD_SESSION,
                                        dict(session_id=session_id))
            if _check_resp_success(resp, (f'Can not add session {session_id} '
                                          f'with error: {resp.type}')):
                self.owned_sessions.append(session_id)

    def stop_session(self, session_id: int):
        """Stop the given session."""
        if session_id not in self.owned_sessions:
            logger.warning(f'session {session_id} is not owned '
                           'by this instance')
        resp = self.req_sender.send(RequestType.STOP_SESSION,
                                    dict(session_id=session_id))
        _check_resp_success(resp, (f'Failed to cancel session: {session_id}. '
                                   f'Error: {resp.type}.'))

    def end_session(self, session_id: int):
        """End the given session."""
        if session_id not in self.owned_sessions:
            logger.warning(f'session {session_id} is not owned '
                           'by this instance')
        resp = self.req_sender.send(RequestType.END_SESSION,
                                    dict(session_id=session_id))
        if _check_resp_success(resp, (f'Failed to end session: {session_id}. '
                                      f'Error: {resp.type}.')):
            self.owned_sessions.remove(session_id)

    @torch.inference_mode()
    def create_model_inputs(self, messages: List[SchedulerSequence]):
        """create model inputs from messages.

        Args:
            messages (List[SchedulerSequence]): The input messages.
        """
        history_lengths = [msg.history_len for msg in messages]

        token_ids = [msg.token_ids for msg in messages]

        meta = messages[0].meta

        if isinstance(token_ids[0], int):
            token_ids = [token_ids]

        batch_size = len(messages)
        input_ids = torch.cat(token_ids)

        is_decoding = input_ids.size(0) == batch_size
        if not is_decoding:
            seq_length = [tokens.size(0) for tokens in token_ids]
            seq_length = torch.tensor(seq_length, dtype=torch.long)
            max_seq_len = max(seq_length)
            q_start_loc = seq_length.cumsum(0) - seq_length
            mask_range = torch.arange(max_seq_len)[None, :]
            attention_mask = (mask_range < seq_length[:, None]).long()
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids += position_ids.new_tensor(history_lengths).unsqueeze(
                -1)
        else:
            q_start_loc = self._q_start_loc_buf[:batch_size]
            attention_mask = self._attention_mask_buf[:batch_size]
            seq_length = self._seq_length_buf[:batch_size]
            position_ids = q_start_loc.new_tensor(history_lengths).unsqueeze(
                -1)

        # TODO: get block offsets is slow when block_size = 1
        block_offsets = self.scheduler.get_block_tables(messages)
        block_offsets = _tensorlize_block_offsets(block_offsets)

        # add batch dim [bs=1, seq_len]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        return ModelInputs(input_ids=input_ids,
                           seq_length=seq_length,
                           attention_mask=attention_mask,
                           block_offsets=block_offsets,
                           position_ids=position_ids,
                           q_start_loc=q_start_loc,
                           history_lengths=history_lengths,
                           is_decoding=is_decoding,
                           meta=meta)

    def _stoping_criteria(self, msg: SchedulerSequence, next_token_id: int):
        """Check if the message should stop.

        Args:
            msg (SchedulerSequence): The input message.
            next_token_id (int): The next token id from inference result.

        Returns:
            bool: Weither the message should be stopped.
        """

        # check eof
        def _check_eof(sampling_param, next_token_id, eos_token_id):
            return (not sampling_param.ignore_eos
                    ) and next_token_id == eos_token_id

        def _check_stop_word(sampling_param, next_token_id):
            return (sampling_param.stop_words is not None
                    and next_token_id in sampling_param.stop_words)

        def _check_request_len(msg):
            return msg.remain_output_len <= 0

        def _check_session_len(msg, max_session_len):
            if max_session_len is None:
                return False
            session_len = msg.logical_blocks.num_tokens()
            return session_len >= max_session_len

        sampling_param = msg.sampling_param
        if _check_eof(sampling_param, next_token_id,
                      self.model_config.eos_token_id):
            return True
        if _check_stop_word(sampling_param, next_token_id):
            return True
        if _check_request_len(msg):
            return True
        if _check_session_len(msg, self.scheduler_config.max_session_len):
            return True
        return False

    def sampling_logits(self, logits: torch.Tensor,
                        running: List[SchedulerSequence], inputs: ModelInputs):
        """sampling logits."""

        def _group_params(running):
            sampling_params: List[SamplingParam] = [
                msg.sampling_param for msg in running
            ]
            grouped_params = dict()
            for i, p in enumerate(sampling_params):
                key = (p.top_k, p.top_p, p.temperature, p.repetition_penalty)
                grouped_params.setdefault(key, list())
                grouped_params[key].append(i)
            return grouped_params

        def _sampling(grouped_params, split_logits, inputs):
            next_token_ids = torch.empty((len(running), ), dtype=torch.long)
            for param, idx in grouped_params.items():
                top_k, top_p, temperature, _ = param
                logits_processor = FusedLogitsProcessor(
                    SamplingParam(
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                    ))
                input_ids = inputs.input_ids.reshape(-1, 1)
                new_logits = split_logits[idx]
                new_logits = logits_processor(input_ids, new_logits)
                argmax_ids = new_logits.argmax(-1).cpu()
                next_token_ids[idx] = argmax_ids
            return next_token_ids

        logits = logits.cuda()
        grouped_params = _group_params(running)

        is_decoding = inputs.is_decoding
        # TODO: support repetition_penalty
        if not is_decoding:
            seq_length = inputs.seq_length
            last_idx = seq_length.cumsum(-1) - 1
            split_logits = logits[last_idx, :]
        else:
            # most step share the same sampling parameters
            split_logits = logits

        next_token_ids = _sampling(grouped_params, split_logits, inputs)

        return next_token_ids, split_logits

    def update_running(self, running: List[SchedulerSequence],
                       next_token_ids: torch.Tensor, meta: Any):
        """update scheduler."""
        for token, msg in zip(next_token_ids, running):
            msg.meta = meta
            msg.update_token_ids(token)
            msg.remain_output_len -= 1
            if self._stoping_criteria(msg, token):
                msg.status = MessageStatus.STOPPED

    def step(self, is_prefill: bool, return_logits: bool = False):
        """one step inference. Used to perform streaming chat.

        Args:
            return_logits (bool): Weither to return the output logits.

        Returns:
            Dict[int, InferOutput]: The output of each session.
        """

        # schedule
        schedule_output = self.scheduler.schedule(is_prefill=is_prefill)

        running: List[SchedulerSequence] = schedule_output.running
        swap_in_map = schedule_output.swap_in_map
        swap_out_map = schedule_output.swap_out_map
        if len(running) == 0:
            return dict()

        inputs = self.create_model_inputs(running)

        # inference
        output = self.model_agent.forward(inputs,
                                          swap_in_map=swap_in_map,
                                          swap_out_map=swap_out_map)
        custom_outputs = output['custom_outputs']
        logits = output['logits']
        logits = logits[0]  # [bs, seq, prob] -> [seq, prob]

        next_token_ids, split_logits = self.sampling_logits(
            logits, running, inputs)

        self.update_running(running, next_token_ids, custom_outputs)
        self.scheduler.update()

        # generate output
        outputs: Dict[int, InferOutput] = dict()
        for msg, next_id in zip(running, next_token_ids):
            session_id = msg.session_id
            out = InferOutput(
                session_id=session_id,
                sender_id=msg.sender_id,
                req_id=msg.req_id,
                finish=(msg.status == MessageStatus.STOPPED),
                token_ids=[next_id.item()],
            )
            outputs[session_id] = out

        if return_logits:
            for msg, msg_logit in zip(running, split_logits):
                outputs[msg.session_id].logits = msg_logit
        return outputs

    def batched_infer(self,
                      session_ids: List[int],
                      token_ids: List[List[int]] = None,
                      request_output_len: int = 512,
                      sampling_param: SamplingParam = SamplingParam(),
                      keep_cache: bool = False):
        """Send inference request.

        Args:
            session_id (int): The session id.
            prompt_token_ids (List[int]): The input token ids.
            request_output_len (int): The max output length of this request.
            step (int): No use for now.
            sampling_param (SamplingParam): The sampling param of the output.
            keep_cache (bool): Keep kv cache after infer.

        Returns:
            int: Error flags. 0 if success.
            List[int]: The streaming output tokens.
            int: The number of the output tokens.
        """
        batch_size = len(token_ids)
        assert len(session_ids) == batch_size

        def _add_sessions(session_ids, owned_sessions):
            for session_id in session_ids:
                if session_id not in owned_sessions:
                    self.add_session(session_id)

        def _add_messages(session_ids, token_ids):
            add_msgs = []
            for session_id, token_id in zip(session_ids, token_ids):
                msg = dict(
                    token_ids=token_id,
                    session_id=session_id,
                    max_request_output_len=request_output_len,
                    sampling_param=sampling_param,
                )
                add_msgs.append(msg)
            req_types = [RequestType.ADD_MESSAGE] * batch_size
            req_ids = self.req_sender.batched_send_async(req_types,
                                                         data=add_msgs)
            return req_ids

        _add_sessions(session_ids, self.owned_sessions)
        req_ids = _add_messages(session_ids, token_ids)

        # receive messages
        req_idx_map = dict(zip(req_ids, range(len(req_ids))))
        output_token_ids = [list() for _ in req_ids]
        status = 0
        finish_count = batch_size
        while finish_count:
            if not self.loop_threads.is_alive():
                logger.error('Engine loop is not alive.')
                status = 1
                break

            resp = self.req_sender.recv_any()
            if resp.req_id not in req_ids:
                continue
            idx = req_idx_map[resp.req_id]
            token_ids = output_token_ids[idx]
            if resp.type == ResponseType.SUCCESS:
                token_ids += resp.data['token_ids']
            elif resp.type == ResponseType.FINISH:
                token_ids += resp.data['token_ids']
                if not keep_cache:
                    session_id = session_ids[idx]
                    self.end_session(session_id=session_id)
                finish_count -= 1
            else:
                logger.error(f'Unexpected response: {resp.type}')
                status = 1
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

        outputs = self.step(return_logits=True)

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
        send_resp_que = Queue()

        def _send_resp():
            """send response callback."""
            while True:
                step_tokens = send_resp_que.get()
                for _, out in step_tokens.items():
                    if out.finish:
                        resp_type = ResponseType.FINISH
                    else:
                        resp_type = ResponseType.SUCCESS
                    self.req_manager.response(
                        Response(
                            type=resp_type,
                            sender_id=out.sender_id,
                            req_id=out.req_id,
                            data=dict(token_ids=out.token_ids),
                        ))

        send_thread = Thread(target=_send_resp, daemon=True)
        send_thread.start()
        prefill_interval = self.scheduler_config.prefill_interval
        prefill_counter = prefill_interval

        while True:
            if not self.req_manager.has_requests(
            ) and not self.scheduler.has_unfinished():
                time.sleep(0.01)
                continue

            self.req_manager.step()

            # forward
            if self.scheduler.has_unfinished():
                has_running = self.scheduler.has_running()
                is_prefill = not prefill_counter or not has_running
                if is_prefill:
                    prefill_counter = prefill_interval
                with torch.inference_mode():
                    step_tokens: Dict[int, InferOutput] = self.step(
                        is_prefill=is_prefill)
                prefill_counter -= 1

                # send response
                send_resp_que.put(step_tokens)


class EngineInstance:
    """Instance of TurboMind.

    Args:
        engine (Engine): engine
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self.req_sender = engine.req_manager.build_sender()
        self.owned_sessions: List[int] = list()

    def __del__(self):
        """Destructor."""
        for session_id in self.owned_sessions:
            self.end(session_id)

    def _try_add_session(self, session_id: int):
        """Add new session.

        Args:
            session_id (int): The session id to add.
        """
        if session_id not in self.owned_sessions:
            resp = self.req_sender.send(RequestType.ADD_SESSION,
                                        dict(session_id=session_id))
            if _check_resp_success(resp, (f'Can not add session {session_id} '
                                          f'with error: {resp.type}')):
                self.owned_sessions.append(session_id)

    async def async_stream_infer(self,
                                 session_id: int,
                                 input_ids: List[int],
                                 gen_config: EngineGenerationConfig = None,
                                 **kwargs):
        """Send stream inference request.

        Args:
            session_id (int): The session id.
            input_ids (List[int]): The input token ids.
            request_output_len (int): The max output length of this request.
            gen_config (EngineGenerationConfig): The sampling parameters.

        Yields:
            int: Error flags. 0 if success.
            List[int]: The streaming output tokens.
            int: The number of the output tokens.
        """
        for item in self.stream_infer(session_id=session_id,
                                      input_ids=input_ids,
                                      gen_config=gen_config,
                                      **kwargs):
            await asyncio.sleep(0)
            yield item

    def stream_infer(self,
                     session_id: int,
                     input_ids: List[int],
                     gen_config: EngineGenerationConfig = None,
                     **kwargs):
        """Send stream inference request.

        Args:
            session_id (int): The session id.
            input_ids (List[int]): The input token ids.
            request_output_len (int): The max output length of this request.
            gen_config (EngineGenerationConfig): The sampling parameters.

        Yields:
            int: Error flags. 0 if success.
            List[int]: The streaming output tokens.
            int: The number of the output tokens.
        """

        # TODO: support input embedding, step
        gen_config = gen_config or EngineGenerationConfig()
        request_output_len = gen_config.max_new_tokens
        sampling_param = SamplingParam(
            top_p=gen_config.top_p,
            top_k=gen_config.top_k,
            temperature=gen_config.temperature,
            repetition_penalty=gen_config.repetition_penalty,
            ignore_eos=gen_config.ignore_eos,
            random_seed=gen_config.random_seed,
            stop_words=gen_config.stop_words,
            bad_words=gen_config.bad_words)
        self._try_add_session(session_id)
        msg = dict(
            token_ids=input_ids,
            session_id=session_id,
            max_request_output_len=request_output_len,
            sampling_param=sampling_param,
        )
        req_id = self.req_sender.send_async(RequestType.ADD_MESSAGE, msg)

        token_ids = []
        while True:
            if not self.engine.loop_threads.is_alive():
                yield (ResponseType.ENGINE_STOP_ERROR, [], 0)
                break

            resp = self.req_sender.recv(req_id)
            if resp.req_id != req_id:
                continue
            if resp.type == ResponseType.SUCCESS:
                token_ids += resp.data['token_ids']
                yield (resp.type, token_ids, len(token_ids))
            elif resp.type == ResponseType.FINISH:
                token_ids += resp.data['token_ids']
                yield (resp.type, token_ids, len(token_ids))
                break
            else:
                yield (resp.type, [], 0)
                break

    def infer(self,
              session_id: int,
              prompt_token_ids: List[int] = None,
              gen_config: EngineGenerationConfig = None,
              **kwargs):
        """Send inference request.

        Args:
            session_id (int): The session id.
            prompt_token_ids (List[int]): The input token ids.
            request_output_len (int): The max output length of this request.
            step (int): No use for now.
            gen_config (EngineGenerationConfig): The sampling parameters.

        Returns:
            int: Error flags. 0 if success.
            List[int]: The streaming output tokens.
            int: The number of the output tokens.
        """
        token_ids = []
        for outputs in self.stream_infer(session_id,
                                         prompt_token_ids,
                                         gen_config=gen_config,
                                         **kwargs):
            status, tmp_ids, _ = outputs
            if status != 0:
                return (status, token_ids, len(token_ids))
            token_ids += tmp_ids

        return (0, token_ids, len(token_ids))

    def end(self, session_id: int):
        """End the given session."""
        if session_id not in self.owned_sessions:
            logger.warning(f'session {session_id} is not owned '
                           'by this instance')
        resp = self.req_sender.send(RequestType.END_SESSION,
                                    dict(session_id=session_id))
        if _check_resp_success(resp, (f'Failed to end session: {session_id}. '
                                      f'Error: {resp.type}.')):
            self.owned_sessions.remove(session_id)

    def cancel(self, session_id: int):
        """Stop current streaming inference."""
        if session_id not in self.owned_sessions:
            logger.warning(f'session {session_id} is not owned '
                           'by this instance')
        resp = self.req_sender.send(RequestType.STOP_SESSION,
                                    dict(session_id=session_id))
        _check_resp_success(resp, (f'Failed to cancel session: {session_id}. '
                                   f'Error: {resp.type}.'))

    def decode(self, prompt_token_ids: List[List[int]]):
        """Return logits of context decoding.

        Args:
            prompt_token_ids: token ids of a batch prompts.

        Returns:
            logits (numpy.ndarray) with shape
                [batch, n_max_token_of_the_batch, vocab_size]
        """
        return self.engine.decode(prompt_token_ids)
