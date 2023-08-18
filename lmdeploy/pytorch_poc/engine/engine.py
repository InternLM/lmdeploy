from typing import List
import torch
from lmdeploy.pytorch_poc.patch import patch
from lmdeploy.pytorch_poc.paging import Scheduler
from lmdeploy.pytorch_poc.config import SchedulerConfig, CacheConfig
from lmdeploy.pytorch_poc.messages import SchedulerMessage, SchedulerSession

from transformers import AutoModelForCausalLM
import itertools


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


class Engine:

    def __init__(self, model_path: str) -> None:
        hf_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.patched_model = patch(hf_model).cuda()

        scheduler_config = SchedulerConfig(4, 2048)
        cache_config = CacheConfig(
            block_size=16, num_cpu_blocks=1024, num_gpu_blocks=1024)
        self.scheduler = Scheduler(scheduler_config, cache_config)

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

    def _make_inputs(self, token_ids, device='cuda'):
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
        seq_length = torch.tensor(seq_length).to(device)

        return dict(
            input_ids=input_ids,
            seq_length=seq_length,
            attention_mask=attention_mask,
            position_ids=position_ids)

    def step(self):
        # TODO: cache manage
        import pdb
        pdb.set_trace()

        # schedule
        running = self.scheduler.schedule()
        if len(running) == 0:
            return dict()

        # make batch
        token_ids = [msg.token_ids for msg in running]
        inputs = self._make_inputs(token_ids)

        # inference
        with torch.no_grad():
            hf_outputs = self.patched_model(
                input_ids=inputs['input_ids'],
                position_ids=inputs['position_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False)

        logits = hf_outputs['logits']
        past_key_values = hf_outputs['past_key_values']

        # update scheduler
        self.scheduler.update()

        # gather output
        output_index = inputs['seq_length'].cumsum(0) - 1
        output_token_ids = logits.max(-1)[1]
        next_token_ids = output_token_ids[output_index]
        session_ids = [msg.session_id for msg in running]

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
