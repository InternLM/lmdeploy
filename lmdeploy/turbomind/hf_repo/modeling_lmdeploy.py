# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import count
from queue import Queue
from typing import List, Optional, Tuple, Union

from huggingface_hub import snapshot_download
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from lmdeploy.turbomind import TurboMind
from lmdeploy.turbomind.utils import get_gen_param

from .configuration_lmdeploy import LmdeployConfig

logger = logging.get_logger(__name__)


@dataclass
class Session:
    _count = count()
    _session_id: int = None
    _message: List[Tuple[str, str]] = field(default_factory=list)
    _step: int = 0
    _nth_round: int = 0
    _error: int = 0

    def __init__(self):
        self._session_id = next(Session._count)
        self._message = []
        self._step = 0
        self._nth_round = 0

    @property
    def session_id(self):
        return self._session_id

    @property
    def message(self):
        return self._message

    @property
    def step(self):
        return self._step

    @property
    def nth_round(self):
        return self._nth_round

    @property
    def error(self):
        return self._error


class LmdeployForCausalLM(PreTrainedModel):
    config_class = LmdeployConfig

    def __init__(self,
                 config: LmdeployConfig,
                 *inputs,
                 model_path: str = None,
                 **kwargs):
        super().__init__(config)
        self.tm_model = TurboMind.from_pretrained(model_path, **kwargs)
        que = Queue()
        for _ in range(config.turbomind['max_batch_size']):
            que.put(self.tm_model.create_instance())
        self.que = que

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        *model_args,
                        config: Optional[Union[PretrainedConfig, str,
                                               os.PathLike]] = None,
                        cache_dir: Optional[Union[str, os.PathLike]] = None,
                        force_download: bool = False,
                        local_files_only: bool = False,
                        token: Optional[Union[str, bool]] = None,
                        revision: str = 'main',
                        **kwargs):
        """Instantiate a LM model with turbomind backend."""

        resume_download = kwargs.pop('resume_download', True)
        proxies = kwargs.pop('proxies', None)

        if os.path.isdir(pretrained_model_name_or_path):
            local_folder = pretrained_model_name_or_path
        else:
            local_folder = snapshot_download(
                pretrained_model_name_or_path,
                revision=revision,
                cache_dir=cache_dir,
                proxies=proxies,
                resume_download=resume_download,
                force_download=force_download,
                token=token,
                local_files_only=local_files_only,
            )

        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else local_folder
            kwargs.pop('return_unused_kwargs')
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path, return_unused_kwargs=True, **kwargs)
        else:
            model_kwargs = kwargs

        model = cls(config,
                    *model_args,
                    model_path=local_folder,
                    **model_kwargs)

        generation_config = model.tm_model.model.sampling_param
        for k, v in dataclasses.asdict(generation_config).items():
            if hasattr(model.generation_config, k):
                base_value = getattr(model.generation_config, k)
                setattr(generation_config, k, base_value)
            if k in kwargs:
                setattr(generation_config, k, v)
        model.generation_config = generation_config

        return model

    @contextmanager
    def managed_generator(self, session: Session):
        generator = self.que.get()
        try:
            yield generator
        except:  # noqa E722
            for _ in generator.stream_infer(session.session_id, [0],
                                            request_output_len=0,
                                            sequence_start=False,
                                            sequence_end=False,
                                            stop=True):
                pass
            session._error = 1
        finally:
            self.que.put(generator)

    def generate(
        self,
        input_ids: List[int],
        session: Session,
        **kwargs,
    ):
        """Generates sequences of token ids for models with a language modeling
        head.

        Args:
            input_ids (List(int)): list of input token ids
            session (Session) session information
            kwargs (dict): hoc parametrization of generation
        """
        with self.managed_generator(session) as generator:
            for outputs in generator.stream_infer(
                    session_id=session.session_id,
                    input_ids=[input_ids],
                    **kwargs,
            ):
                res, tokens = outputs[0]
                yield res, tokens

    def chat(
        self,
        query: str,
        session: Optional[Session] = None,
        cap: str = 'chat',
        request_output_len: int = 512,
        stream_output: bool = False,
        ignore_eos=False,
        random_seed: Optional[int] = None,
        **kwargs,
    ) -> Tuple[str, Session]:
        """chat."""

        if session is None:
            session = Session()
        assert session._error == 0, 'An error occurred before, ' \
            'please start a new session.'

        session._message.append([query, ''])

        prompt = self.tm_model.model.get_prompt(query, session.nth_round == 0)
        input_ids = self.tm_model.tokenizer.encode(prompt)

        if len(
                input_ids
        ) + session.step + request_output_len >= self.tm_model.session_len:
            logger.error(
                f'session_length exceeded {self.tm_model.session_len}')
            session._error = 1
            yield '', session
        else:
            gen_param = get_gen_param(cap, self.generation_config,
                                      session.nth_round + 1, session.step,
                                      request_output_len, **kwargs)
            gen_kwargs = dataclasses.asdict(gen_param)
            gen_kwargs.update(
                random_seed=random_seed if session.nth_round == 0 else None,
                stream_output=stream_output,
                ignore_eos=ignore_eos,
                **kwargs)

            _step = session._step
            _nth_round = session._nth_round
            response_size = 0

            for res, tokens in self.generate(input_ids,
                                             session=session,
                                             **gen_kwargs):
                response = self.tm_model.tokenizer.decode(res.tolist(),
                                                          offset=response_size)
                if response.endswith('�'):
                    continue
                response_size = tokens

                session._message[-1][-1] += response
                session._nth_round = _nth_round + 1
                session._step = _step + response_size

                yield response, session
