# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from dataclasses import dataclass, field
from typing import List

import torch
from transformers import AutoModelForCausalLM

from lmdeploy.pytorch_poc.config import (CacheConfig, ModelConfig,
                                         SchedulerConfig)
from lmdeploy.pytorch_poc.messages import SchedulerMessage, SchedulerSession
from lmdeploy.pytorch_poc.paging import BlockTable, Scheduler
from lmdeploy.pytorch_poc.patch import patch
from lmdeploy.pytorch_poc.utils import get_gpu_memory
from lmdeploy.utils import get_logger

from .cache_engine import CacheEngine

logger = get_logger('lmdeploy')


class SamplingParam:

    def __init__(self,
                 top_p: float = 0.8,
                 top_k: int = None,
                 temperature: float = 0.8,
                 repetition_penalty: float = 1.0,
                 ignore_eos: bool = False,
                 random_seed: int = None,
                 stop_words: List[int] = None,
                 bad_words: List[int] = None):
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.ignore_eos = ignore_eos
        self.random_seed = random_seed
        self.stop_words = stop_words
        self.bad_words = bad_words


@dataclass
class ModelContext:
    block_tables: List[BlockTable] = field(default_factory=list)
    history_lengths: List[int] = field(default_factory=list)

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

            k_state = k_states[loc:loc + seq_len]
            v_state = v_states[loc:loc + seq_len]

            # fill remain(last non-full block)
            block_id = b_offsets[free_offset]
            fill_token_num = min(block_size - token_offset, seq_len)
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


class Engine:

    def __init__(self, model_path: str) -> None:
        hf_model = AutoModelForCausalLM.from_pretrained(model_path)

        self.context = ModelContext()
        self.patched_model = patch(hf_model, self.context).cuda()
        hf_config = hf_model.config

        scheduler_config = SchedulerConfig(4, 2048)
        cache_config = CacheConfig(block_size=16,
                                   num_cpu_blocks=0,
                                   num_gpu_blocks=0)
        model_config = ModelConfig(hf_config.hidden_size,
                                   hf_config.num_hidden_layers,
                                   hf_config.num_attention_heads,
                                   dtype=torch.float32)

        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.model_config = model_config

        self.init_cache_engine(model_config, cache_config)
        self.scheduler = Scheduler(scheduler_config, cache_config)

    def init_cache_engine(self, model_config: ModelConfig,
                          cache_config: CacheConfig):
        GPU_MEM_PERCENT = 0.5
        SWAP_SPACE = 4 * (1 << 30)
        gpu_mem = get_gpu_memory() * GPU_MEM_PERCENT
        cpu_mem = SWAP_SPACE
        cache_block_size = CacheEngine.get_cache_block_size(
            cache_config.block_size, model_config)
        if cache_config.num_cpu_blocks == 0:
            cache_config.num_cpu_blocks = int(cpu_mem / cache_block_size)
        if cache_config.num_gpu_blocks == 0:
            cache_config.num_gpu_blocks = int(gpu_mem / cache_block_size)
        self.cache_engine = CacheEngine(cache_config, model_config)
        logger.debug(
            f'Initialize cache engine with {cache_config.num_gpu_blocks}'
            f' gpu blocks and {cache_config.num_cpu_blocks} cpu blocks.')

    def create_instance(self, cuda_stream_id=0):
        """Create a turbomind instance.

        Args:
            cuda_stream_id(int): identity of a cuda stream
        Returns:
            EngineInstance: an instance of turbomind
        """
        return EngineInstance(self, cuda_stream_id)

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
        attention_mask = torch.tensor([
            seq_len * [1] + (max_seq_len - seq_len) * [0]
            for seq_len in seq_length
        ]).to(device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids += position_ids.new_tensor(history_lengths).unsqueeze(-1)
        seq_length = torch.tensor(seq_length).to(device)

        block_tables = self.scheduler.get_block_tables(messages)

        return dict(input_ids=input_ids,
                    seq_length=seq_length,
                    attention_mask=attention_mask,
                    block_tables=block_tables,
                    past_key_values=self.cache_engine.gpu_cache,
                    position_ids=position_ids)

    def step(self):
        # TODO: cache manage

        # schedule
        schedule_output = self.scheduler.schedule()

        running = schedule_output.running
        swap_in_map = schedule_output.swap_in_map
        swap_out_map = schedule_output.swap_out_map
        if len(running) == 0:
            return dict()

        sessions = self.scheduler.get_sessions(running)
        session_ids = [msg.session_id for msg in running]
        history_lengths = [sess.history_length for sess in sessions]

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

        # make batch
        inputs = self._make_inputs(running)

        # inference
        with torch.no_grad():
            # setup context
            self.context.block_tables = inputs['block_tables']
            self.context.history_lengths = history_lengths

            # forward
            hf_outputs = self.patched_model(
                input_ids=inputs['input_ids'],
                position_ids=inputs['position_ids'],
                attention_mask=inputs['attention_mask'],
                past_key_values=inputs['past_key_values'],
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False)

        logits = hf_outputs['logits']

        # update scheduler
        self.scheduler.update()

        # gather output
        output_index = inputs['seq_length'].cumsum(0) - 1
        output_token_ids = logits.max(-1)[1]
        next_token_ids = output_token_ids[output_index]

        # update message for next step
        for token, msg in zip(next_token_ids, running):
            msg.token_ids = token.unsqueeze(0)

        outputs = dict(zip(session_ids, next_token_ids))

        return outputs


class EngineInstance:
    """Instance of TurboMind.

    Args:
        engine (Engine): engine
        cuda_stream_id(int): identity of a cuda stream
    """

    def __init__(self, engine, cuda_stream_id=0):
        self.engine = engine
        self.patched_model = engine.patched_model
        self.scheduler = engine.scheduler

    def stream_infer(self,
                     session_id: int,
                     prompt_token_ids: List[int] = None,
                     request_output_len: int = None,
                     step: int = 0,
                     sampling_param: SamplingParam = SamplingParam()):
        pass
