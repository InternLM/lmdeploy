# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import dataclasses
import random
from argparse import ArgumentError
from contextlib import contextmanager
from typing import List, Literal, Optional, Union

from lmdeploy.messages import EngineGenerationConfig, GenerationConfig
from lmdeploy.model import ChatTemplateConfig, best_match_model
from lmdeploy.pytorch import EngineConfig as PytorchEngineConfig
from lmdeploy.turbomind import EngineConfig as TurbomindEngineConfig
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


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
        model_path (str): the path of a model.
            It could be one of the following options:
                - i) A local directory path of a turbomind model which is
                    converted by `lmdeploy convert` command or download from
                    ii) and iii).
                - ii) The model_id of a lmdeploy-quantized model hosted
                    inside a model repo on huggingface.co, such as
                    "InternLM/internlm-chat-20b-4bit",
                    "lmdeploy/llama2-chat-70b-4bit", etc.
                - iii) The model_id of a model hosted inside a model repo
                    on huggingface.co, such as "InternLM/internlm-chat-7b",
                    "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                    and so on.
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "InternLM/internlm-chat-7b",
            "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat" and so on.
        backend (str): either `turbomind` or `pytorch` backend. Default to
            `turbomind` backend.
        backend_config (EngineConfig): beckend config. Default to none.
        chat_template_config (ChatTemplateConfig): chat template configuration.
            Default to None.
        instance_num (int): instance numbers to be created
        tp (int): tensor parallel
    """

    def __init__(self,
                 model_path: str,
                 model_name: Optional[str] = None,
                 backend: Literal['turbomind', 'pytorch'] = 'turbomind',
                 backend_config: Optional[Union[TurbomindEngineConfig,
                                                PytorchEngineConfig]] = None,
                 chat_template_config: Optional[ChatTemplateConfig] = None,
                 instance_num: int = 32,
                 tp: int = 1,
                 **kwargs) -> None:
        if backend == 'turbomind':
            if backend_config is None:
                backend_config = TurbomindEngineConfig(model_name=model_name,
                                                       tp=tp)
            from lmdeploy import turbomind as tm
            self.engine = tm.TurboMind.from_pretrained(
                model_path,
                engine_config=backend_config,
                chat_template_config=chat_template_config,
                **kwargs)
            self.chat_template = self.engine.model
        elif backend == 'pytorch':
            self.model_name = model_name
            from lmdeploy.pytorch.engine import Engine

            # try fuzzy matching to get a model_name
            if self.model_name is None and (backend_config is None or
                                            backend_config.model_name == ''):
                potential_names = best_match_model(model_path)
                if potential_names is None:
                    raise ArgumentError(
                        'Please set model_name or backend_config.')
                else:
                    self.model_name = potential_names[0]
                    logger.warning(
                        f'Best matched chat template name: {self.model_name}')
            elif self.model_name is not None and backend_config is not None:
                if self.model_name != backend_config.model_name:
                    raise ArgumentError(
                        f'Got different model names from model_name = '
                        f'{self.model_name}, backend_config = {backend_config}'
                    )
            if self.model_name is not None and backend_config is None:
                backend_config = PytorchEngineConfig(self.model_name,
                                                     session_len=2048)
            self.engine = Engine(model_path=model_path,
                                 engine_config=backend_config)
            if chat_template_config is None:
                chat_template_config = ChatTemplateConfig(self.model_name)
            self.chat_template = chat_template_config.chat_template
        else:
            raise ValueError(f'unsupported backend {backend}')
        self.tokenizer = self.engine.tokenizer
        self.instance_num = instance_num
        self.id2step = {}
        self.id2generator = {}
        self.loop = asyncio.get_event_loop()
        self.gens_set = set()
        for i in range(instance_num):
            self.gens_set.add(self.engine.create_instance())

    def __call__(self,
                 prompts: List[str],
                 gen_config: Optional[GenerationConfig] = None,
                 chat_template_config: Optional[ChatTemplateConfig] = None,
                 request_output_len=512,
                 top_k=40,
                 top_p=0.8,
                 temperature=0.8,
                 repetition_penalty=1.0,
                 ignore_eos=False,
                 **kwargs):
        """Inference a batch of prompts.

        Args:
            prompts (List[str]): a batch of prompts
            gen_config (GenerationConfig | None): a instance of
                GenerationConfig. Default to None.
            chat_template_config (ChatTemplateConfig | None):a instance of
                ChatTemplateConfig. Default to None.
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
        """
        return self.batch_infer(prompts,
                                gen_config=gen_config,
                                chat_template_config=chat_template_config,
                                request_output_len=request_output_len,
                                top_k=top_k,
                                top_p=top_p,
                                temperature=temperature,
                                repetition_penalty=repetition_penalty,
                                ignore_eos=ignore_eos,
                                **kwargs)

    def stop_session(self, session_id: int):
        """Stop a session by a session_id."""
        if str(session_id) in self.id2generator:
            self.id2generator[str(session_id)].cancel(session_id)
            if self.id2generator[str(session_id)] not in self.gens_set:
                self.gens_set.add(self.id2generator[str(session_id)])

    def end_session(self, session_id: int):
        """Clear a session by a session_id."""
        if str(session_id) in self.id2generator:
            self.id2generator[str(session_id)].end(session_id)
            self.id2step[str(session_id)] = 0
            if self.id2generator[str(session_id)] not in self.gens_set:
                self.gens_set.add(self.id2generator[str(session_id)])

    @contextmanager
    def safe_run(self, session_id: Optional[int] = None):
        """A context manager to make sure server's safe running."""
        try:
            yield
        except (Exception, asyncio.CancelledError) as e:  # noqa
            self.stop_session(session_id)
            raise e
        if str(session_id) in self.id2generator and self.id2generator[str(
                session_id)] not in self.gens_set:
            self.gens_set.add(self.id2generator[str(session_id)])

    async def get_generator(self, stop: bool, session_id: int):
        """Only return the model instance if it is available."""
        if stop:
            return self.engine.create_instance()
        while self.gens_set == set():
            await asyncio.sleep(0)
        generator = self.gens_set.pop()
        self.id2generator[str(session_id)] = generator
        return generator

    def batch_infer(self,
                    prompts: Union[List[str], str],
                    gen_config: Optional[GenerationConfig] = None,
                    chat_template_config: Optional[ChatTemplateConfig] = None,
                    request_output_len=512,
                    top_k=40,
                    top_p=0.8,
                    temperature=0.8,
                    repetition_penalty=1.0,
                    ignore_eos=False,
                    **kwargs):
        """Inference a batch of prompts.

        Args:
            prompts (List[str] | str): a batch of prompts
            gen_config (GenerationConfig | None): a instance of
                GenerationConfig. Default to None.
            chat_template_config (ChatTemplateConfig | None):a instance of
                ChatTemplateConfig. Default to None.
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
        """
        input_str = isinstance(prompts, str)
        prompts = [prompts] if input_str else prompts
        assert isinstance(prompts, List), 'prompts should be a list'
        batch_size = len(prompts)
        outputs = [''] * batch_size
        generators = []
        for i, prompt in enumerate(prompts):
            generators.append(
                self.generate(prompt,
                              i,
                              gen_config=gen_config,
                              chat_template_config=chat_template_config,
                              stream_response=True,
                              sequence_start=True,
                              sequence_end=True,
                              request_output_len=request_output_len,
                              top_k=top_k,
                              top_p=top_p,
                              temperature=temperature,
                              ignore_eos=ignore_eos,
                              repetition_penalty=repetition_penalty,
                              **kwargs))

        async def _inner_call(i, generator):
            async for out in generator:
                outputs[i] += out.response

        async def gather():
            await asyncio.gather(
                *[_inner_call(i, generators[i]) for i in range(batch_size)])

        self.loop.run_until_complete(gather())
        outputs = outputs[0] if input_str else outputs
        return outputs

    async def generate(
            self,
            messages,
            session_id: int,
            gen_config: Optional[GenerationConfig] = None,
            chat_template_config: Optional[ChatTemplateConfig] = None,
            stream_response: bool = True,
            sequence_start: bool = True,
            sequence_end: bool = True,  # no interactive mode by default
            step: int = 0,
            request_output_len: int = 512,
            stop: bool = False,
            stop_words: Optional[List[str]] = None,
            top_k: int = 40,
            top_p: float = 0.8,
            temperature: float = 0.8,
            repetition_penalty: float = 1.0,
            ignore_eos: bool = False,
            **kwargs):
        """Generate responses.

        Args:
            messages (str | List): chat history or prompt
            session_id (int): the session id
            gen_config (GenerationConfig | None): a instance of
                GenerationConfig. Default to None.
            chat_template_config (ChatTemplateConfig | None):a instance of
                ChatTemplateConfig. Default to None.
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
        """
        if str(session_id) not in self.id2step:
            self.id2step[str(session_id)] = 0
        if step != 0:
            self.id2step[str(session_id)] = step
        seed = random.getrandbits(64)
        prompt = messages
        if chat_template_config is not None:
            chat_template = chat_template_config.chat_template
        else:
            chat_template = self.chat_template
        prompt = chat_template.messages2prompt(prompt, sequence_start)
        input_ids = self.tokenizer.encode(prompt, add_bos=sequence_start)
        finish_reason = None
        if stop is True:
            self.stop_session(session_id)
            yield GenOut('', self.id2step[str(session_id)], len(input_ids), 0,
                         finish_reason)
        elif self.id2step[str(session_id)] + len(
                input_ids) + request_output_len >= self.engine.session_len:
            finish_reason = 'length'
            yield GenOut('', self.id2step[str(session_id)], len(input_ids), 0,
                         finish_reason)
            if sequence_end is True and sequence_start is False:
                self.end_session(session_id)
        else:
            if gen_config is None:
                gen_config = GenerationConfig(
                    max_new_tokens=request_output_len,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    ignore_eos=ignore_eos,
                    random_seed=seed if sequence_start else None,
                    stop_words=stop_words)
            gen_config = EngineGenerationConfig.From(gen_config,
                                                     self.tokenizer)
            generator = await self.get_generator(stop, session_id)
            with self.safe_run(session_id):
                response_size = 0
                async for outputs in generator.async_stream_infer(
                        session_id=session_id,
                        input_ids=input_ids,
                        gen_config=gen_config,
                        stream_output=stream_response,
                        sequence_start=(sequence_start),
                        sequence_end=sequence_end,
                        step=self.id2step[str(session_id)],
                        stop=stop):
                    status, res, tokens = outputs
                    # decode res
                    response = self.tokenizer.decode(res, offset=response_size)
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

                finish_reason = 'length' \
                    if tokens >= request_output_len else 'stop'
                # `response_size` might be note updated since
                # ` if response.endswith('�')`
                if response_size == tokens:
                    response = ''  # avaid returning the last response twice
                yield GenOut(response, self.id2step[str(session_id)],
                             len(input_ids), tokens, finish_reason)
                # update step
                self.id2step[str(session_id)] += len(input_ids) + tokens
                if sequence_end or stop:
                    self.id2step[str(session_id)] = 0
