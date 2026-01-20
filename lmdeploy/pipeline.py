# Copyright (c) OpenMMLab. All rights reserved.
import os
from contextlib import closing
from typing import TYPE_CHECKING, Dict, Iterator, List, Tuple

import tqdm
from typing_extensions import deprecated

from .archs import autoget_backend_config, get_task
from .messages import GenerationConfig, PytorchEngineConfig, SpeculativeConfig, TurbomindEngineConfig
from .model import ChatTemplateConfig
from .serve.processors import MultimodalProcessor
from .utils import get_logger, get_model

if TYPE_CHECKING:
    from PIL.Image import Image

    from .serve.managers import Session

logger = get_logger('lmdeploy')


class Pipeline:
    """Pipeline - User-facing API layer for inference."""

    def __init__(self,
                 model_path: str,
                 backend_config: TurbomindEngineConfig | PytorchEngineConfig | None = None,
                 chat_template_config: ChatTemplateConfig | None = None,
                 log_level: str = 'WARNING',
                 max_log_len: int | None = None,
                 speculative_config: SpeculativeConfig | None = None,
                 **kwargs):
        """Initialize Pipeline.

        Args:
            model_path: Path to the model.
            backend_config: Backend configuration.
            chat_template_config: Chat template configuration.
            log_level: Log level.
            max_log_len: Max number of prompt characters or prompt tokens being printed in log.
            speculative_config: Speculative decoding configuration.
            **kwargs: Additional keyword arguments.
        """

        os.environ.setdefault('TM_LOG_LEVEL', log_level)
        logger.setLevel(log_level)

        # Download model if the path does not exist locally
        if not os.path.exists(model_path):
            download_dir = backend_config.download_dir if backend_config else None
            revision = backend_config.revision if backend_config else None
            model_path = get_model(model_path, download_dir, revision)

        # Download speculative model if the path does not exist locally
        if speculative_config and speculative_config.model and not os.path.exists(speculative_config.model):
            download_dir = backend_config.download_dir if backend_config else None
            speculative_config.model = get_model(speculative_config.model, download_dir)

        # Create inference engine
        _, pipeline_class = get_task(model_path)
        backend, backend_config = autoget_backend_config(model_path, backend_config)
        self.async_engine = pipeline_class(model_path,
                                           backend=backend,
                                           backend_config=backend_config,
                                           chat_template_config=chat_template_config,
                                           max_log_len=max_log_len,
                                           speculative_config=speculative_config,
                                           **kwargs)
        self.session_mgr = self.async_engine.session_mgr
        self.backend_config = self.async_engine.backend_config

    def infer(self,
              prompts: List[str] | str | List[Dict] | List[List[Dict]] | Tuple | List[Tuple],
              session_id: List[int] | int | None = None,
              gen_config: GenerationConfig | List[GenerationConfig] | None = None,
              do_preprocess: bool = True,
              adapter_name: str | None = None,
              use_tqdm: bool = False,
              **kwargs):
        """Inference prompts.

        Args:
            prompts: Prompts to inference. It can be a single prompt, a list of prompts, a list of tuples, or a tuple.
                Tuple can be (prompt, image or [images]) or (image or [images], prompt).
            session_id(List[int] | int | None): Session ID(s).
            gen_config(GenerationConfig | List[GenerationConfig] | None): Generation configuration(s).
            do_preprocess(bool): Whether to pre-process messages.
            adapter_name(str | None): Adapter name.
            use_tqdm(bool): Whether to use progress bar.
            **kwargs(dict): Additional keyword arguments.
        """
        is_single = self._is_single(prompts)
        # format prompts to openai message format, which is a list of dicts
        prompts = MultimodalProcessor.format_prompts(prompts)
        pbar = tqdm.tqdm(total=len(prompts)) if use_tqdm else None
        outputs = []
        try:
            requests = self._request_generator(prompts,
                                               session_id=session_id,
                                               gen_config=gen_config,
                                               do_preprocess=do_preprocess,
                                               adapter_name=adapter_name,
                                               stream_response=False,
                                               **kwargs)
            for g in self.async_engine._infer(requests, multiplex=False, pbar=pbar):
                res = None
                for out in g:
                    res = res.extend(out) if res else out
                outputs.append(res)
        finally:
            if pbar: pbar.close()  # noqa
        if is_single:
            return outputs[0]
        return outputs

    @deprecated('This method is deprecated. Please use "Pipeline.infer" instead.')
    def batch_infer(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def stream_infer(self,
                     prompts: List[str] | str | List[Dict] | List[List[Dict]] | Tuple | List[Tuple],
                     session_id: int | List[int] | None = None,
                     gen_config: GenerationConfig | List[GenerationConfig] | None = None,
                     do_preprocess: bool = True,
                     adapter_name: str | None = None,
                     stream_response: bool = True,
                     **kwargs):
        """Stream inference.

        Args:
            prompts(List[str] | str | List[Dict] | List[List[Dict]] | Tuple | List[Tuple]): Prompts to inference.
                It can be a single prompt, a list of prompts, a list of tuples, or a tuple.
                Tuple can be (prompt, image or [images]) or (image or [images], prompt).
            session_id(int | List[int] | None): Session ID.
            gen_config(GenerationConfig | List[GenerationConfig] | None): Generation configuration(s).
            do_preprocess(bool): Whether to pre-process messages.
            adapter_name(str | None): Adapter name.
            stream_response(bool): Whether to stream the response. If True, the generator will stream the response.
                Otherwise, the generator will run until finish and return the final response. This argument
                is introduced to support the streaming and non-streaming modes of Pipeline.chat.
            **kwargs(dict): Additional keyword arguments.

        Returns:
            Generator: A generator that yields the output (i.e. instance of class `Response`) of the inference.
        """
        prompts = MultimodalProcessor.format_prompts(prompts)
        requests = self._request_generator(prompts,
                                           session_id=session_id,
                                           gen_config=gen_config,
                                           do_preprocess=do_preprocess,
                                           adapter_name=adapter_name,
                                           stream_response=stream_response,
                                           **kwargs)
        return self.async_engine._infer(requests, multiplex=True)

    def close(self):
        """Close the pipeline."""
        self.async_engine.close()

    def chat(self,
             prompt: str | Tuple[str, 'Image' | List['Image']],
             session=None,
             gen_config: GenerationConfig | None = None,
             stream_response=False,
             adapter_name=None,
             **kwargs) -> 'Session' | Iterator:
        """Chat.

        Args:
            prompt (str): prompt
            session (Session): the chat session
            gen_config (GenerationConfig | None): a instance of
                GenerationConfig. Default to None.
            stream_response (bool): whether to stream the response.
            adapter_name (str): adapter name.
            **kwargs (dict): additional keyword arguments.
        """
        if session is None:
            session = self.session_mgr.get()
        session.update(prompt=prompt, response=None)

        prompt = MultimodalProcessor.format_prompts(prompt)

        sequence_start = session.step == 0
        generator = self.stream_infer(prompt,
                                      session_id=session.session_id,
                                      gen_config=gen_config,
                                      stream_response=stream_response,
                                      adapter_name=adapter_name,
                                      multiplex=True,
                                      sequence_start=sequence_start,
                                      sequence_end=False,
                                      step=session.step)

        def _gen():
            resp = None
            try:
                for out in generator:
                    resp = resp.extend(out) if resp else out
                    yield out
            except:  # noqa
                self.async_engine._run(coro=self.session_mgr.async_abort(session)).result()
                raise
            else:
                session.response = resp
                session.step += resp.generate_token_len + resp.input_token_len
                session.history.append((session.prompt, resp.text))

        if stream_response:
            return _gen()
        else:
            # run the generator until finish
            with closing(_gen()) as gen:
                for _ in gen:
                    pass
            session.generator = None

        return session

    def open_session(self) -> 'Session':
        """Open a new session."""
        return self.session_mgr.get()

    def stop_session(self, session: 'Session'):
        """Stop a session."""
        self.async_engine._run(coro=self.session_mgr.async_abort(session)).result()

    def end_session(self, session: 'Session'):
        """End a session."""
        self.async_engine._run(coro=self.session_mgr.async_end(session)).result()

    def get_ppl(self, input_ids: List[int] | List[List[int]]) -> List[float]:
        """Get perplexity scores given a list of input tokens that have to be
        of the same length.

        Args:
            input_ids (List[int] | List[List[int]]): the batch of
                input token ids

        Returns:
            List[float]: A list of perplexity scores.
        """
        return self.async_engine.get_ppl(input_ids)

    def __call__(self,
                 prompts: List[str] | str | List[Dict] | List[List[Dict]],
                 gen_config: GenerationConfig | List[GenerationConfig] | None = None,
                 **kwargs):
        return self.infer(prompts, gen_config=gen_config, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @deprecated('This method is deprecated. Please use "AsyncEngine.generate" instead.')
    async def generate(self, *args, **kwargs):
        """Generate responses as an async generator.

        This method delegates to async_engine.generate and forwards all yielded values.
        """
        async for item in self.async_engine.generate(*args, **kwargs):
            yield item

    @staticmethod
    def _is_single(prompts):
        """Check if prompts is a single prompt."""
        return (isinstance(prompts, str) or (isinstance(prompts, tuple) and len(prompts) == 2)
                or (isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], Dict)))

    def _request_generator(self,
                           prompts: List[str] | str | List[Dict] | List[List[Dict]],
                           session_id: List[int] | int | None = None,
                           gen_config: GenerationConfig | List[GenerationConfig] | None = None,
                           **kwargs):
        """Generate requests."""
        is_single = self._is_single(prompts)
        prompts = [prompts] if is_single else prompts

        if session_id is None:
            session_ids = [self.session_mgr.reserve() for _ in prompts]
        elif isinstance(session_id, list):
            session_ids = session_id
        else:
            session_ids = [session_id]

        if len(prompts) != len(session_ids):
            raise ValueError(f'prompts and session_ids should have the same length. '
                             f'Got {len(prompts)} prompts and {len(session_ids)} session_ids')

        if gen_config is None:
            gen_configs = [GenerationConfig()] * len(prompts)
        elif isinstance(gen_config, list):
            gen_configs = gen_config
        else:
            gen_configs = [gen_config] * len(prompts)

        if len(prompts) != len(gen_configs):
            raise ValueError(f'input gen_config length differs from the length of prompts. '
                             f'Got {len(prompts)} prompts and {len(gen_configs)} gen_configs')

        for prompt, gen_cfg, sid in zip(prompts, gen_configs, session_ids):
            yield dict(session_id=sid, messages=prompt, gen_config=gen_cfg, **kwargs)
