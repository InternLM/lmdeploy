# Copyright (c) OpenMMLab. All rights reserved.
import logging
import re

from transformers import PreTrainedTokenizerFast

from .base import BasicAdapterFast

logger = logging.getLogger(__name__)

B_INST, E_INST = '[INST]', '[/INST]'
B_SYS, E_SYS = '<<SYS>>\n', '\n<</SYS>>\n\n'
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""   # noqa: E501


class Llama2Adapter(BasicAdapterFast):
    """Adapter for llama2.

    Llama2 use the following template and the first user prompt
    should contain a system prompt.

    User can specify the system prompt using a <<SYS>> tag otherwise
    the default system prompt is prepended to user's input.

        <bos>
        [INST]<space>
        <<SYS>>\n
        SYSTEM_PROMPT\n
        <</SYS>>\n\n
        {user_prompt_1}<space>
        [/INST]<space>
        {answer_1}<space>
        <eos>

        <bos>
        [INST]<space>
        {user_prompt_2}<space>
        [/INST]<space>
        {answer_2}<space>
        <eos>

        <bos>
        [INST]<space>
        {user_prompt_2}(no space here)
        ...
    """

    start_ids = []
    sep_ids = []

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        super().__init__(tokenizer)
        self.prev_round = 0

    def encode_and_decorate(self, prompt):
        r"""Encode prompt and decorate with template."""

        if self.prev_round == 0:
            res = re.search(r'<<SYS>>(.*?)<</SYS>>(.*)', prompt)
            if res:
                prompt = B_SYS + res.group(1).strip() + \
                    E_SYS + res.group(2).strip()
            else:
                prompt = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS + prompt

        prompt = f'{B_INST} {prompt.strip()} {E_INST}'

        logger.debug(f'decorated prompt: {repr(prompt)}')

        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            return_tensors='pt',
        )

        self.prev_round += 1
        return input_ids
