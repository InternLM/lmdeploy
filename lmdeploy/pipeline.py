# Copyright (c) OpenMMLab. All rights reserved.
import os
from contextlib import closing
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union

import tqdm

from .archs import autoget_backend_config, get_task
from .messages import GenerationConfig, PytorchEngineConfig, SpeculativeConfig, TurbomindEngineConfig
from .model import ChatTemplateConfig
from .utils import get_logger, get_model

if TYPE_CHECKING:
    from .serve.session_manager import Session

logger = get_logger('lmdeploy')


class Pipeline:
    """Pipeline - User-facing API layer for inference."""

    def __init__(self,
                 model_path: str,
                 backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig]] = None,
                 chat_template_config: Optional[ChatTemplateConfig] = None,
                 log_level: str = 'WARNING',
                 max_log_len: int = None,
                 speculative_config: Optional[SpeculativeConfig] = None,
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
        if not isinstance(backend_config, PytorchEngineConfig):
            backend_config = autoget_backend_config(model_path, backend_config)
        backend = 'pytorch' if isinstance(backend_config, PytorchEngineConfig) else 'turbomind'

        self.async_engine = pipeline_class(model_path,
                                           backend=backend,
                                           backend_config=backend_config,
                                           chat_template_config=chat_template_config,
                                           max_log_len=max_log_len,
                                           speculative_config=speculative_config,
                                           **kwargs)

    def infer(self,
              prompts: Union[List[str], str, List[Dict], List[List[Dict]]],
              gen_config: Optional[Union[GenerationConfig, List[GenerationConfig]]] = None,
              do_preprocess: bool = True,
              adapter_name: Optional[str] = None,
              stream_response: bool = False,
              use_tqdm: bool = False,
              **kwargs):
        """Inference prompts.

        Args:
            prompts: Prompts to inference.
            gen_config: Generation configuration(s).
            do_preprocess: Whether to pre-process messages.
            adapter_name: Adapter name.
            stream_response: Whether to stream response.
            multiplex: Whether to multiplex responses.
            pbar: Progress bar.
        """
        pbar = tqdm.tqdm(total=len(prompts)) if use_tqdm else None
        requests = self._request_generator(prompts,
                                           gen_config,
                                           do_preprocess=do_preprocess,
                                           adapter_name=adapter_name,
                                           stream_response=stream_response,
                                           **kwargs)
        return self.async_engine._infer(requests, multiplex=False, pbar=pbar)

    def stream_infer(self,
                     prompts: Union[List[str], str, List[Dict], List[List[Dict]]],
                     gen_config: Optional[Union[GenerationConfig, List[GenerationConfig]]] = None,
                     do_preprocess: bool = True,
                     adapter_name: Optional[str] = None,
                     stream_response: bool = True,
                     **kwargs):
        """Stream inference."""
        requests = self._request_generator(prompts,
                                           gen_config,
                                           do_preprocess=do_preprocess,
                                           adapter_name=adapter_name,
                                           stream_response=stream_response,
                                           **kwargs)
        return self.async_engine._infer(requests, multiplex=True)

    def close(self):
        """Close the pipeline."""
        self.async_engine.close()

    def chat(self,
             prompt: str,
             session=None,
             gen_config: Optional[GenerationConfig] = None,
             stream_response=False,
             adapter_name=None,
             **kwargs) -> Union['Session', Iterator]:
        """Chat.

        Args:
            prompt (str): prompt
            session (Session): the chat session
            gen_config (GenerationConfig | None): a instance of
                GenerationConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            **kwargs (dict): ad hoc parametrization of `gen_config
        """
        if session is None:
            session = self.session_mgr.create()

        # sync & init
        session.update(prompt=prompt, response=None)

        sequence_start = session.step == 0

        generator = self.infer(prompt,
                               gen_config,
                               adapter_name=adapter_name,
                               sequence_start=sequence_start,
                               sequence_end=False,
                               session_id=session.session_id,
                               stream_response=stream_response,
                               multiplex=True,
                               step=session.step)

        def _gen():
            resp = None
            try:
                for out in generator:
                    resp = resp.extend(out) if resp else out
                    yield out
            except:  # noqa
                self._run(coro=self.stop_session(session.session_id)).result()
                raise
            else:
                session._response = resp
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

    def session(self):
        """Create a new session."""
        return self.session_mgr.create()

    def open_session(self):
        """Open a new session."""
        return self.session_mgr.create()

    def close_session(self, session: 'Session'):
        """Close a session."""
        self.async_engine.run(coro=self.async_engine.end_session(session.session_id))
        self.session_mgr.end(session)
        self.session_mgr.sessions.pop(session.session_id)

    def end_session(self, session: 'Session'):
        """End a session."""
        self.session_mgr.end(session)
        self.session_mgr.sessions.pop(session.session_id)

    def __call__(self,
                 prompts: Union[List[str], str, List[Dict], List[List[Dict]]],
                 gen_config: Optional[GenerationConfig] = None,
                 do_preprocess: bool = True,
                 adapter_name: Optional[str] = None,
                 use_tqdm: bool = False,
                 **kwargs):
        """Inference a batch of prompts.

        Args:
            prompts: A batch of prompts.
            gen_config: Generation configuration.
            do_preprocess: Whether to pre-process the messages.
            adapter_name: Adapter name for slora.
            use_tqdm: Whether to use progress bar.
        """
        return self.infer(prompts,
                          gen_config=gen_config,
                          do_preprocess=do_preprocess,
                          adapter_name=adapter_name,
                          use_tqdm=use_tqdm,
                          **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _is_single(prompts):
        """Check if prompts is a single prompt."""
        return isinstance(prompts, str) or (isinstance(prompts, list) and len(prompts) > 0
                                            and isinstance(prompts[0], Dict))

    def _request_generator(self,
                           prompts: Union[List[str], str, List[Dict], List[List[Dict]]],
                           gen_config: Optional[Union[GenerationConfig, List[GenerationConfig]]] = None,
                           **kwargs):
        """Generate requests."""
        prompts = [prompts] if self._is_single(prompts) else prompts
        assert isinstance(prompts, List), 'prompts should be a list'
        gen_config = gen_config or GenerationConfig()
        if not isinstance(gen_config, List):
            gen_config = [gen_config] * len(prompts)
        assert len(prompts) == len(gen_config), 'input gen_config length differs from the length of prompts'
        for prompt, gen_cfg in zip(prompts, gen_config):
            session = self.session()
            yield dict(session_id=session.session_id, messages=prompt, gen_config=gen_cfg, **kwargs)
