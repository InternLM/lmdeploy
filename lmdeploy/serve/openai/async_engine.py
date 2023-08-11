# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import dataclasses
import os.path as osp
import random
from typing import Literal, Optional

from lmdeploy import turbomind as tm
from lmdeploy.model import MODELS, BaseModel
from lmdeploy.turbomind.tokenizer import Tokenizer


def valid_str(string, coding='utf-8'):
    """decode text according to its encoding type."""
    invalid_chars = [b'\xef\xbf\xbd']
    bstr = bytes(string, coding)
    for invalid_char in invalid_chars:
        bstr = bstr.replace(invalid_char, b'')
    ret = bstr.decode(encoding=coding, errors='ignore')
    return ret


@dataclasses.dataclass
class GenOut:
    response: str
    history_token_len: int
    input_token_len: int
    generate_token_len: int
    finish_reason: Optional[Literal['stop', 'length']] = None


class AsyncEngine:

    def __init__(self, model_path, instance_num=32, tp=1) -> None:
        tokenizer_model_path = osp.join(model_path, 'triton_models',
                                        'tokenizer')
        tokenizer = Tokenizer(tokenizer_model_path)
        self.tm_model = tm.TurboMind(model_path,
                                     eos_id=tokenizer.eos_token_id,
                                     tp=tp)
        self.tokenizer = tokenizer
        self.generators = [
            self.tm_model.create_instance() for i in range(instance_num)
        ]
        self.instance_num = instance_num
        self.model: BaseModel = MODELS.get(self.tm_model.model_name)()
        self.available = [True] * instance_num
        self.starts = [None] * instance_num
        self.steps = {}

    async def get_generator(self, instance_id):
        while self.available[instance_id] is False:
            await asyncio.sleep(0.1)
        return self.generators[instance_id]

    async def generate(
        self,
        messages,
        instance_id,
        stream_response=True,
        renew_session=False,
        request_output_len=512,
        stop=False,
        top_k=40,
        top_p=0.8,
        temperature=0.8,
        repetition_penalty=1.0,
        ignore_eos=False,
    ):
        session_id = instance_id
        instance_id %= self.instance_num
        sequence_start = False
        generator = await self.get_generator(instance_id)
        self.available[instance_id] = False
        if renew_session and str(session_id) in self.steps and self.steps[str(
                session_id)] > 0:  # renew a session
            empty_prompt = self.model.messages2prompt('', False)
            empty_input_ids = self.tokenizer.encode(empty_prompt)
            for outputs in generator.stream_infer(session_id=session_id,
                                                  input_ids=[empty_input_ids],
                                                  request_output_len=512,
                                                  sequence_start=False,
                                                  sequence_end=True):
                pass
            self.steps[str(session_id)] = 0
        if str(session_id) not in self.steps:
            self.steps[str(session_id)] = 0
        if self.steps[str(session_id)] == 0:
            sequence_start = True
        seed = random.getrandbits(64)
        prompt = self.model.messages2prompt(messages, sequence_start)
        input_ids = self.tokenizer.encode(prompt)
        response_size = 0
        async for outputs in generator.async_stream_infer(
                session_id=session_id,
                input_ids=[input_ids],
                stream_output=stream_response,
                request_output_len=request_output_len,
                sequence_start=(sequence_start),
                sequence_end=False,
                step=self.steps[str(session_id)],
                stop=stop,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                ignore_eos=ignore_eos,
                random_seed=seed if sequence_start else None):
            res, tokens = outputs[0]
            # decode res
            response = self.tokenizer.decode(res)[response_size:]
            response = valid_str(response)
            # response out, history token len, input token len, gen token len
            yield GenOut(response, self.steps[str(session_id)], len(input_ids),
                         tokens)
            response_size += len(response)

        # update step
        self.steps[str(session_id)] += len(input_ids) + tokens
        self.available[instance_id] = True
