# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import dataclasses
from argparse import ArgumentError
from contextlib import contextmanager
from typing import Dict, List, Literal, Optional, Union

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
                    on huggingface.co, such as "internlm/internlm-chat-7b",
                    "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                    and so on.
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "internlm/internlm-chat-7b",
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
            self._build_turbomind(model_path=model_path,
                                  model_name=model_name,
                                  backend_config=backend_config,
                                  chat_template_config=chat_template_config,
                                  tp=tp,
                                  **kwargs)
        elif backend == 'pytorch':
            self._build_pytorch(model_path=model_path,
                                model_name=model_name,
                                backend_config=backend_config,
                                chat_template_config=chat_template_config,
                                **kwargs)
        else:
            raise ValueError(f'unsupported backend {backend}')
        self.backend = backend
        self.tokenizer = self.engine.tokenizer
        self.instance_num = instance_num
        self.id2step = {}
        self.id2generator = {}
        self.loop = asyncio.get_event_loop()
        self.gens_set = set()
        for i in range(instance_num):
            self.gens_set.add(self.engine.create_instance())

    def _build_turbomind(
            self,
            model_path: str,
            model_name: Optional[str] = None,
            backend_config: Optional[Union[TurbomindEngineConfig,
                                           PytorchEngineConfig]] = None,
            chat_template_config: Optional[ChatTemplateConfig] = None,
            tp: int = 1,
            **kwargs):
        """Innter build method for turbomind backend."""
        if backend_config is None:
            backend_config = TurbomindEngineConfig(model_name=model_name,
                                                   tp=tp)
        assert isinstance(backend_config, TurbomindEngineConfig), 'Please'\
            ' use EngineConfig imported from lmdeploy.turbomind for ' \
            'turbomind backend'
        from lmdeploy import turbomind as tm
        self.engine = tm.TurboMind.from_pretrained(
            model_path,
            engine_config=backend_config,
            chat_template_config=chat_template_config,
            **kwargs)
        if chat_template_config is None:
            chat_template_config = ChatTemplateConfig(self.engine.model_name)
        self.chat_template = chat_template_config.chat_template
        self.session_len = self.engine.session_len

    def _build_pytorch(
            self,
            model_path: str,
            model_name: Optional[str] = None,
            backend_config: Optional[Union[TurbomindEngineConfig,
                                           PytorchEngineConfig]] = None,
            chat_template_config: Optional[ChatTemplateConfig] = None,
            **kwargs):
        """Innter build method for pytorch backend."""
        self.model_name = model_name
        from lmdeploy.pytorch.engine import Engine

        # try fuzzy matching to get a model_name
        if self.model_name is None and (backend_config is None
                                        or backend_config.model_name == ''):
            potential_names = best_match_model(model_path)
            if potential_names is None:
                raise ArgumentError('Please set model_name or backend_config.')
            else:
                self.model_name = potential_names[0]
                logger.warning(
                    f'Best matched chat template name: {self.model_name}')
        elif self.model_name is not None and backend_config is not None:
            if self.model_name != backend_config.model_name:
                raise ArgumentError(
                    f'Got different model names from model_name = '
                    f'{self.model_name}, backend_config = {backend_config}')
        if self.model_name is not None and backend_config is None:
            backend_config = PytorchEngineConfig(self.model_name,
                                                 session_len=2048)
        assert isinstance(backend_config, PytorchEngineConfig), 'Please '\
            'use EngineConfig imported from lmdeploy.pytorch for pytorch' \
            ' backend'
        self.engine = Engine(model_path=model_path,
                             engine_config=backend_config)
        if chat_template_config is None:
            chat_template_config = ChatTemplateConfig(self.model_name)
        self.chat_template = chat_template_config.chat_template
        if self.engine.session_len is None:
            self.session_len = self.chat_template.session_len
        else:
            self.session_len = self.engine.session_len

    def __call__(self,
                 prompts: List[str],
                 gen_config: Optional[GenerationConfig] = None,
                 request_output_len=512,
                 top_k: int = 40,
                 top_p: float = 0.8,
                 temperature: float = 0.8,
                 repetition_penalty: float = 1.0,
                 ignore_eos: bool = False,
                 do_preprocess: bool = True,
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
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
        """
        if gen_config is None:
            gen_config = GenerationConfig(
                max_new_tokens=request_output_len,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                ignore_eos=ignore_eos)
        return self.batch_infer(prompts,
                                gen_config=gen_config,
                                do_preprocess=do_preprocess,
                                **kwargs)

    def stop_session(self, session_id: int):
        """Stop a session by a session_id."""
        if str(session_id) in self.id2generator:
            self.id2generator[str(session_id)].cancel(session_id)
            self.gens_set.add(self.id2generator[str(session_id)])

    def end_session(self, session_id: int):
        """Clear a session by a session_id."""
        if str(session_id) in self.id2generator:
            self.id2generator[str(session_id)].end(session_id)
            self.id2step.pop(str(session_id))
            self.gens_set.add(self.id2generator[str(session_id)])

    @contextmanager
    def safe_run(self, session_id: Optional[int] = None):
        """A context manager to make sure server's safe running."""
        try:
            yield
        except (Exception, asyncio.CancelledError) as e:  # noqa
            self.stop_session(session_id)
            raise e
        if str(session_id) in self.id2generator:
            self.gens_set.add(self.id2generator[str(session_id)])

    async def get_generator(self, stop: bool, session_id: int):
        """Only return the model instance if it is available."""
        if stop:
            return self.engine.create_instance()
        while self.gens_set == set():
            await asyncio.sleep(0)
        if str(session_id) in self.id2generator:
            # pytorch engine instance is bind to session
            generator = self.id2generator[str(session_id)]
            if generator in self.gens_set:
                self.gens_set.remove(generator)
        else:
            generator = self.gens_set.pop()
            self.id2generator[str(session_id)] = generator
        return generator

    def batch_infer(self,
                    prompts: Union[List[str], str],
                    gen_config: Optional[Union[GenerationConfig,
                                               EngineGenerationConfig]] = None,
                    do_preprocess: bool = True,
                    **kwargs):
        """Inference a batch of prompts.

        Args:
            prompts (List[str] | str): a batch of prompts
            gen_config (GenerationConfig | None): a instance of
                GenerationConfig. Default to None.
            chat_template_config (ChatTemplateConfig | None):a instance of
                ChatTemplateConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
        """
        need_list_wrap = isinstance(prompts, str) or isinstance(
            prompts[0], Dict)
        prompts = [prompts] if need_list_wrap else prompts
        assert isinstance(prompts, List), 'prompts should be a list'
        if gen_config is None:
            gen_config = GenerationConfig()
        if type(gen_config) is GenerationConfig:
            gen_config = EngineGenerationConfig.From(gen_config,
                                                     self.tokenizer)
        prompt_num = len(prompts)
        outputs = [''] * prompt_num
        for j in range(0, prompt_num, self.instance_num):
            batch_prompts = prompts[j:j + self.instance_num]
            generators = []
            for i, prompt in enumerate(batch_prompts):
                generators.append(
                    self.generate(prompt,
                                  i,
                                  gen_config=gen_config,
                                  stream_response=True,
                                  sequence_start=True,
                                  sequence_end=True,
                                  do_preprocess=do_preprocess,
                                  **kwargs))

            async def _inner_call(i, generator):
                async for out in generator:
                    outputs[i + j] += out.response

            async def gather():
                await asyncio.gather(*[
                    _inner_call(i, generators[i])
                    for i in range(len(batch_prompts))
                ])

            self.loop.run_until_complete(gather())
        outputs = outputs[0] if need_list_wrap else outputs
        return outputs

    async def generate(
            self,
            messages,
            session_id: int,
            gen_config: Optional[Union[GenerationConfig,
                                       EngineGenerationConfig]] = None,
            stream_response: bool = True,
            sequence_start: bool = True,
            sequence_end: bool = True,  # no interactive mode by default
            step: int = 0,
            stop: bool = False,
            do_preprocess: bool = True,
            **kwargs):
        """Generate responses.

        Args:
            messages (str | List): chat history or prompt
            session_id (int): the session id
            gen_config (GenerationConfig | None): a instance of
                GenerationConfig. Default to None.
            stream_response (bool): whether return responses streamingly
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
            step (int): the offset of the k/v cache
            stop (bool): whether stop inference
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
        """
        if str(session_id) not in self.id2step:
            self.id2step[str(session_id)] = 0
        if step != 0:
            self.id2step[str(session_id)] = step
        if gen_config is None:
            gen_config = GenerationConfig()
        if type(gen_config) is GenerationConfig:
            gen_config = EngineGenerationConfig.From(gen_config,
                                                     self.tokenizer)
        prompt = messages
        if do_preprocess:
            prompt = self.chat_template.messages2prompt(prompt, sequence_start)
        input_ids = self.tokenizer.encode(prompt, add_bos=sequence_start)
        finish_reason = None
        if stop is True:
            self.stop_session(session_id)
            yield GenOut('', self.id2step[str(session_id)], len(input_ids), 0,
                         finish_reason)
        elif self.id2step[str(session_id)] + len(
                input_ids) + gen_config.max_new_tokens >= self.session_len:
            finish_reason = 'length'
            yield GenOut('', self.id2step[str(session_id)], len(input_ids), 0,
                         finish_reason)
            if sequence_end is True and sequence_start is False:
                self.end_session(session_id)
        else:
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
                    if tokens >= gen_config.max_new_tokens else 'stop'
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
                # manually end pytorch session
                # TODO modify pytorch or turbomind api
                if self.backend == 'pytorch' and sequence_end:
                    self.end_session(session_id)
