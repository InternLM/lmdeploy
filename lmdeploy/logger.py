# Copyright (c) OpenMMLab. All rights reserved.
# modify from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/logger.py  # noqa
from typing import List, Optional

from .messages import GenerationConfig
from .utils import get_logger

logger = get_logger('lmdeploy')


class RequestLogger:
    """A class responsible for logging requests, ensuring that logs do not
    exceed a specified maximum length.

    Args:
        max_log_len (Optional[int]): The maximum length of the log entries.
            If None, no maximum length is enforced.
    """

    def __init__(self, max_log_len: Optional[int]) -> None:
        self.max_log_len = max_log_len

    def log_prompt(self, session_id: int, prompt: str) -> None:
        if self.max_log_len is not None:
            if prompt is not None:
                prompt = prompt[:self.max_log_len]
        logger.info(f'session_id={session_id}, '
                    f'prompt={prompt!r}')

    def log_inputs(self, session_id: int, prompt: Optional[str],
                   prompt_token_ids: Optional[List[int]],
                   gen_config: GenerationConfig, adapter_name: str) -> None:
        max_log_len = self.max_log_len
        if max_log_len is not None:
            if prompt is not None:
                prompt = prompt[:max_log_len]

            if prompt_token_ids is not None:
                prompt_token_ids = prompt_token_ids[:max_log_len]

        logger.info(f'session_id={session_id}, '
                    f'prompt={prompt!r}, '
                    f'gen_config={gen_config}, '
                    f'prompt_token_id={prompt_token_ids}, '
                    f'adapter_name={adapter_name}.')
