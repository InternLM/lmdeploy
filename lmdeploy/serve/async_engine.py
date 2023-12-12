# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import dataclasses
import random
from contextlib import contextmanager
from typing import List, Literal, Optional


@dataclasses.dataclass
class GenOut:
    """Pack all response information together."""
    response: str
    history_token_len: int
    input_token_len: int
    generate_token_len: int
    finish_reason: Optional[Literal['stop', 'length']] = None


class AsyncEngine:
    """Async inference engine. Maintaining a bunch of tm_model instances.

    Args:
        model_path (str): the path of the deployed model
        instance_num (int): instance numbers to be created
        tp (int): tensor parallel
    """

    def __init__(self, model_path, instance_num=32, tp=1, **kwargs) -> None:
        from lmdeploy import turbomind as tm
        self.tm_model = tm.TurboMind.from_pretrained(model_path,
                                                     tp=tp,
                                                     **kwargs)
        self.tokenizer = self.tm_model.tokenizer
        self.instance_num = instance_num
        self.model = self.tm_model.model
        self.id2step = {}
        self.id2generator = {}
        self.loop = asyncio.get_event_loop()
        self.special_gen = self.tm_model.create_instance()
        self.gens_set = set()
        for i in range(instance_num):
            self.gens_set.add(self.tm_model.create_instance())

    def stop_session(self, session_id: int):
        """Stop a session by a session_id."""
        input_ids = [self.tm_model.eos_id]
        stop_generator = self.id2generator.get(str(session_id),
                                               self.special_gen)
        for outputs in stop_generator.stream_infer(session_id,
                                                   input_ids,
                                                   request_output_len=0,
                                                   sequence_start=False,
                                                   sequence_end=False,
                                                   stop=True):
            pass
        if str(session_id) in self.id2generator and self.id2generator[str(
                session_id)] not in self.gens_set:
            self.gens_set.add(self.id2generator[str(session_id)])

    def end_session(self, session_id: int):
        """Clear a session by a session_id."""
        input_ids = [self.tm_model.eos_id]
        end_generator = self.id2generator.get(str(session_id),
                                              self.special_gen)
        for outputs in end_generator.stream_infer(session_id,
                                                  input_ids,
                                                  request_output_len=0,
                                                  sequence_start=False,
                                                  sequence_end=True,
                                                  stop=True):
            pass
        self.id2step[str(session_id)] = 0
        if str(session_id) in self.id2generator and self.id2generator[str(
                session_id)] not in self.gens_set:
            self.gens_set.add(self.id2generator[str(session_id)])

    @contextmanager
    def safe_run(self, session_id: Optional[int] = None):
        """A context manager to make sure server's safe running."""
        try:
            yield
        except (Exception, asyncio.CancelledError) as e:  # noqa
            self.stop_session(session_id)
        if str(session_id) in self.id2generator and self.id2generator[str(
                session_id)] not in self.gens_set:
            self.gens_set.add(self.id2generator[str(session_id)])

    async def get_embeddings(self, prompt, do_prerpocess=False):
        if do_prerpocess:
            prompt = self.model.get_prompt(prompt)
        input_ids = self.tokenizer.encode(prompt)
        return input_ids

    async def get_generator(self, stop: bool, session_id: int):
        """Only return the model instance if it is available."""
        if stop:
            return self.id2generator.get(str(session_id), self.special_gen)
        while self.gens_set == set():
            await asyncio.sleep(0)
        return self.gens_set.pop()

    def batch_infer(self,
                    prompts: List[str],
                    request_output_len=512,
                    top_k=40,
                    top_p=0.8,
                    temperature=0.8,
                    repetition_penalty=1.0,
                    ignore_eos=False,
                    do_preprocess=True,
                    **kwargs):
        """Inference a batch of prompts.

        Args:
            prompts (List[str]): a batch of prompts
            request_output_len (int): output token nums
            top_k (int): The number of the highest probability vocabulary
              tokens to keep for top-k-filtering
            top_p (float): If set to float < 1, only the smallest set of most
              probable tokens with probabilities that add up to top_p or higher
            are kept for generation.
            temperature (float): to modulate the next token probability
            repetition_penalty (float): The parameter for repetition penalty.
              1.0 means no penalty
            ignore_eos (bool): indicator for ignoring eos
            do_preprocess (bool): whether pre-process the messages.
        """
        assert isinstance(prompts, List), 'prompts should be a list'
        batch_size = len(prompts)
        outputs = [''] * batch_size
        generators = []
        for i, prompt in enumerate(prompts):
            generators.append(
                self.generate(prompt,
                              i,
                              stream_response=True,
                              sequence_start=True,
                              sequence_end=True,
                              request_output_len=request_output_len,
                              top_k=top_k,
                              top_p=top_p,
                              temperature=temperature,
                              ignore_eos=ignore_eos,
                              repetition_penalty=repetition_penalty,
                              do_preprocess=do_preprocess,
                              **kwargs))

        async def _inner_call(i, generator):
            async for out in generator:
                outputs[i] += out.response

        async def gather():
            await asyncio.gather(
                *[_inner_call(i, generators[i]) for i in range(batch_size)])

        self.loop.run_until_complete(gather())
        return outputs

    async def generate(
            self,
            messages,
            session_id,
            stream_response=True,
            sequence_start=True,
            sequence_end=True,  # no interactive mode by default
            step=0,
            request_output_len=512,
            stop=False,
            top_k=40,
            top_p=0.8,
            temperature=0.8,
            repetition_penalty=1.0,
            ignore_eos=False,
            do_preprocess=True,
            **kwargs):
        """Generate responses.

        Args:
            messages (str | List): chat history or prompt
            session_id (int): the session id
            stream_response (bool): whether return responses streamingly
            request_output_len (int): output token nums
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
            step (int): the offset of the k/v cache
            stop (bool): whether stop inference
            top_k (int): The number of the highest probability vocabulary
              tokens to keep for top-k-filtering
            top_p (float): If set to float < 1, only the smallest set of most
              probable tokens with probabilities that add up to top_p or higher
              are kept for generation.
            temperature (float): to modulate the next token probability
            repetition_penalty (float): The parameter for repetition penalty.
              1.0 means no penalty
            ignore_eos (bool): indicator for ignoring eos
            do_preprocess (bool): whether pre-process the messages.
        """
        if str(session_id) not in self.id2step:
            self.id2step[str(session_id)] = 0
        if step != 0:
            self.id2step[str(session_id)] = step
        seed = random.getrandbits(64)
        prompt = messages
        if do_preprocess:
            prompt = self.model.messages2prompt(prompt, sequence_start)
        input_ids = self.tokenizer.encode(prompt, add_bos=sequence_start)
        finish_reason = 'stop' if stop else None
        if self.id2step[str(session_id)] + len(
                input_ids) + request_output_len >= self.tm_model.session_len:
            finish_reason = 'length'
            yield GenOut('', self.id2step[str(session_id)], len(input_ids), 0,
                         finish_reason)
            if sequence_end is True and sequence_start is False:
                self.end_session(session_id)
        else:
            generator = await self.get_generator(stop, session_id)
            self.id2generator[str(session_id)] = generator
            with self.safe_run(session_id):
                response_size = 0
                async for outputs in generator.async_stream_infer(
                        session_id=session_id,
                        input_ids=[input_ids],
                        stream_output=stream_response,
                        request_output_len=request_output_len,
                        sequence_start=(sequence_start),
                        sequence_end=sequence_end,
                        step=self.id2step[str(session_id)],
                        stop=stop,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        ignore_eos=ignore_eos,
                        random_seed=seed if sequence_start else None):
                    res, tokens = outputs[0]
                    # decode res
                    response = self.tokenizer.decode(res.tolist(),
                                                     offset=response_size)
                    # utf-8 char at the end means it's a potential unfinished
                    # byte sequence, continue to concate it with the next
                    # sequence and decode them together
                    if response.endswith('�'):
                        continue
                    # response, history token len,
                    # input token len, gen token len
                    yield GenOut(response, self.id2step[str(session_id)],
                                 len(input_ids), tokens, finish_reason)
                    response_size = tokens

                # `response_size` might be note updated since
                # ` if response.endswith('�')`
                if response_size != tokens:
                    yield GenOut(response, self.id2step[str(session_id)],
                                 len(input_ids), tokens, finish_reason)
                # update step
                self.id2step[str(session_id)] += len(input_ids) + tokens
                if sequence_end or stop:
                    self.id2step[str(session_id)] = 0
