# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig, VisionConfig
from lmdeploy.utils import get_logger
from lmdeploy.vl.model.builder import load_vl_model

logger = get_logger('lmdeploy')


def _raise_exception_on_finish(task: asyncio.Task) -> None:
    """Raise exception on finish."""
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as e:
        raise e


def _accepts_arg(func, arg_name: str) -> bool:
    """Check if a function accepts a specific keyword argument."""
    return arg_name in inspect.signature(func).parameters


class ImageEncoder:
    """Image encoder."""

    def __init__(
        self,
        model_path: str,
        backend: str,
        vision_config: VisionConfig = None,
        backend_config: TurbomindEngineConfig | PytorchEngineConfig | None = None,
    ):
        self.model = load_vl_model(model_path, backend, backend_config=backend_config)
        if vision_config is None:
            vision_config = VisionConfig()
        self.vision_config = vision_config
        self.max_batch_size = vision_config.max_batch_size
        self.executor = ThreadPoolExecutor(max_workers=1)
        torch.cuda.empty_cache()

    def apply_chat_template(self, messages, chat_template, sequence_start, chat_template_kwargs=None):
        return self.model.apply_chat_template(
            messages, chat_template, sequence_start, chat_template_kwargs
        )

    async def preprocess(self,
                         messages: list[dict],
                         input_prompt: str | list[int],
                         mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:
        """Preprocess multimodal data in the messages."""
        if _accepts_arg(self.model.preprocess, 'mm_processor_kwargs'):
            future = asyncio.get_event_loop().run_in_executor(self.executor,
                                                              self.model.preprocess,
                                                              messages,
                                                              input_prompt,
                                                              mm_processor_kwargs)
        else:
            future = asyncio.get_event_loop().run_in_executor(self.executor,
                                                              self.model.preprocess,
                                                              messages,
                                                              input_prompt)
        future.add_done_callback(_raise_exception_on_finish)
        outputs = await future
        return outputs

    async def async_infer(self, messages: list[dict]) -> list[dict]:
        """Get multimodal embedding.

        Args:
            messages (list[dict]): a list of message, which is the output
            of `preprocess()`
        """
        future = asyncio.get_event_loop().run_in_executor(self.executor, self.model.forward, messages,
                                                          self.max_batch_size)
        future.add_done_callback(_raise_exception_on_finish)
        outputs = await future
        return outputs

    async def wrap_for_turbomind(
        self,
        messages: list[dict],
        chat_template,
        tokenizer,
        sequence_start,
        tools: list[object] | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> dict:
        """
        Args:
            messages (list[dict]): a list of message, which is supposed to be
                the output of `async_infer`

        Returns:
            dict: a dict passed to turbomind engine_instance's forward.
                The dict has the following structure::

                    {
                        'prompt': 'the prompt after applying chat template',
                        'input_ids': [],
                        'input_embeddings': list[torch.Tensor],
                        'input_embedding_ranges': list[torch.Tensor],
                        ...
                    }
        """
        result = self.model.to_turbomind(messages,
                                         chat_template,
                                         tokenizer,
                                         sequence_start,
                                         tools=tools,
                                         chat_template_kwargs=chat_template_kwargs)
        # clear data
        for i, message in enumerate(messages):
            if isinstance(message['content'], list):
                messages[i]['preprocess'] = None
                messages[i]['forward'] = None
        return result
