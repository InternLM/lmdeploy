# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import numpy as np
import torch

from lmdeploy.utils import get_logger
from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel
from lmdeploy.vl.model.qwen3 import Qwen3VLModel

logger = get_logger('lmdeploy')



@VISION_MODELS.register_module()
class InternS1ProVisionModel(Qwen3VLModel):
    """InternS1Pro model.

    Basically the same preprocessing as Qwen3VL, but with Time Series support.
    """

    _arch = ['InternS1_1_ForConditionalGeneration', 'InternS1ProForConditionalGeneration']

    def build_preprocessor(self, trust_remote_code: bool = False):
        super().build_preprocessor(trust_remote_code=trust_remote_code)

        # time series tokens
        self.ts_token = getattr(self.processor, 'ts_token', None)
        self.ts_token_id = getattr(self.processor, 'ts_token_id', None)

    def _preprocess_time_series(self,
                                data: list[Any],
                                params: dict[str, Any],
                                mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:

        ts_input = data
        sr = params.get('sampling_rate') if params is not None else None

        if not isinstance(ts_input, np.ndarray):
            ts_input = np.array(ts_input, dtype=np.float32)

        mean = ts_input.mean(axis=0, keepdims=True)
        std = ts_input.std(axis=0, keepdims=True)
        ts_input = (ts_input - mean) / (std + 1e-8)

        # truncate to 240k to avoid OOM
        max_ts_len = 240000
        if len(ts_input) > max_ts_len:
            ts_input = ts_input[:max_ts_len]

        if ts_input.ndim == 1:
            ts_input = ts_input[:, None]  # [T,C]

        ts_len = ts_input.shape[0]

        # set the default value to ts_len / 4 if sr is not provided or invalid
        if sr is None or sr <= 0:
            sr = max(ts_len / 4, 1.0)

        # compute num ts tokens
        stride = np.floor(160 / ((1 + np.exp(-sr / 100))**6))
        patch_size = stride * 2
        embed_length = (np.ceil((ts_len - patch_size) / stride) + 1)
        ts_tokens = int((embed_length // 2 + 1) // 2)

        return dict(ts_values=[ts_input],
                    ts_sr=[sr],
                    ts_lens=[ts_len],
                    ts_tokens=[ts_tokens],
                    ts_token_id=self.ts_token_id)

    def preprocess(self, messages: list[dict], mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:
        """Refer to `super().preprocess()` for spec."""
        outputs = []
        self.contains_video_input = False
        self.contains_ts_input = False

        mm_items = self.collect_multimodal_items(messages)
        for modality, data, params in mm_items:
            result = {}
            if modality == Modality.IMAGE:
                result = self._preprocess_image(data, params, mm_processor_kwargs)
            elif modality == Modality.VIDEO:
                self.contains_video_input = True
                result = self._preprocess_video(data, params, mm_processor_kwargs)
            elif modality == Modality.TIME_SERIES:
                self.contains_ts_input = True
                result = self._preprocess_time_series(data, params, mm_processor_kwargs)

            result.update(modality=modality)
            outputs.append(result)

        messages.append(dict(role='preprocess', content=outputs))
        return messages

    def proc_messages(self,
                      messages,
                      chat_template,
                      sequence_start,
                      tools: list[object] | None = None,
                      chat_template_kwargs=None):
        """Apply chat template to get the prompt."""
        chat_template_kwargs = chat_template_kwargs or {}
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        messages = [x for x in messages if x['role'] not in ['preprocess', 'forward']]
        if VisionModel.IMAGE_TOKEN_included(messages):
            # backward compatibility
            for message in messages:
                role, content = message['role'], message['content']
                if role != 'user' or isinstance(content, str):
                    prompt_messages.append(message)
                    continue
                content = [x['text'] for x in content if x['type'] == 'text']
                prompt = ''.join(content)
                prompt = prompt.replace(IMAGE_TOKEN, f'<|vision_start|>{self.image_token}<|vision_end|>')
                prompt_messages.append(dict(role='user', content=prompt))
        else:
            prompt_messages = messages

        # time series input requires enabling_thinking = False
        if self.contains_ts_input:
            chat_template_kwargs['enable_thinking'] = False

        prompt = chat_template.messages2prompt(prompt_messages, sequence_start, tools=tools, **chat_template_kwargs)
        return prompt, None

    def to_pytorch_aux_ts(self, messages, prompt, TS_TOKEN, tokenizer, sequence_start):
        """Pack the time series input to the compatible format with pytorch
        engine."""
        # collect all preprocessing result from messages
        preps = [x['content'] for x in messages if x['role'] == 'preprocess']
        assert len(preps) == 1
        preps = preps[0]

        # split prompt into segments and validate data
        segs = prompt.split(TS_TOKEN)
        assert len(segs) == len(preps) + 1, (f'the number of {TS_TOKEN} is not equal '
                                             f'to input time series data, {len(segs) - 1} vs {len(preps)}')

        input_ids = []
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(preps):
                preps[i - 1].update(offset=len(input_ids))
                ts_tokens = preps[i - 1]['ts_tokens']

                ts_tokens = ts_tokens[0]
                ts_array = np.array(preps[i - 1]['ts_values'])

                preps[i - 1].update(ts_tokens=ts_tokens)
                preps[i - 1].update(ts_values=torch.from_numpy(ts_array).to(dtype=torch.bfloat16))
                preps[i - 1].update(ts_lens=torch.tensor(preps[i - 1]['ts_lens']))
                preps[i - 1].update(ts_sr=torch.tensor(preps[i - 1]['ts_sr']))

                assert self.ts_token_id == preps[i - 1]['ts_token_id']
                input_ids.extend([self.ts_token_id] * ts_tokens)
            token_ids = tokenizer.encode(seg, add_bos=((i == 0) and sequence_start))
            input_ids.extend(token_ids)

        return dict(prompt=prompt, input_ids=input_ids, multimodal=preps)

    def to_pytorch(self,
                   messages,
                   chat_template,
                   tokenizer,
                   sequence_start,
                   tools: list[object] | None = None,
                   chat_template_kwargs: dict | None = None,
                   **kwargs):
        """Return to the information needed by pytorch engine."""
        prompt, _ = self.proc_messages(messages,
                                       chat_template,
                                       sequence_start,
                                       tools=tools,
                                       chat_template_kwargs=chat_template_kwargs)

        if self.contains_video_input:
            return self.to_pytorch_aux_video(messages, prompt, self.video_token, tokenizer, sequence_start)
        elif self.contains_ts_input:
            return self.to_pytorch_aux_ts(messages, prompt, self.ts_token, tokenizer, sequence_start)
        else:
            return self.to_pytorch_aux(messages, prompt, self.image_token, tokenizer, sequence_start)
