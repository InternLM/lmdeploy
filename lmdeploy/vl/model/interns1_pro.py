# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import numpy as np
import torch

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, MultimodalSpecialTokens
from lmdeploy.vl.model.qwen3 import Qwen3VLModel

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class InternS1ProVisionModel(Qwen3VLModel):
    """InternS1Pro model.

    Basically the same preprocessing as Qwen3VL, but with Time Series support.
    """

    _arch = ['InternS1_1_ForConditionalGeneration', 'InternS1ProForConditionalGeneration']

    def build_preprocessor(self):
        super().build_preprocessor()

        # time series tokens
        self.ts_token = getattr(self.processor, 'ts_token', None)
        self.ts_token_id = getattr(self.processor, 'ts_token_id', None)
        self.ts_start_token = getattr(self.processor, 'ts_start_token', None)
        self.ts_end_token = getattr(self.processor, 'ts_end_token', None)

        # special tokens
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_token,
            video_token=self.video_token,
            ts_token=self.ts_token,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            ts_token_id=self.ts_token_id
        )

    def time_series_processor(self,
                              text: str,
                              time_series: list[Any],
                              sampling_rate: float | None = None,
                              **kwargs):

        ts_input = time_series[0] if isinstance(time_series, list) else time_series
        sampling_rate = sampling_rate[0] if isinstance(sampling_rate, list) else sampling_rate

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
        if sampling_rate is None or sampling_rate <= 0:
            sampling_rate = max(ts_len / 4, 1.0)

        # compute num ts tokens
        stride = np.floor(160 / ((1 + np.exp(-sampling_rate / 100))**6))
        patch_size = stride * 2
        embed_length = (np.ceil((ts_len - patch_size) / stride) + 1)
        ts_tokens = int((embed_length // 2 + 1) // 2)

        # generate text with ts tokens
        for i in range(len(text)):
            if f'{self.ts_start_token}{self.ts_token}{self.ts_end_token}' in text[i]:
                ts_placeholder = self.ts_start_token + self.ts_token * ts_tokens + self.ts_end_token
                text[i] = text[i].replace(
                    f'{self.ts_start_token}{self.ts_token}{self.ts_end_token}', ts_placeholder, 1
                )
            elif self.ts_token in text[i]:
                text[i] = text[i].replace(self.ts_token, self.ts_token * ts_tokens)

        input_ids = self.tokenizer(text, add_special_tokens=False, **kwargs)['input_ids']

        ts_input = torch.from_numpy(np.array([ts_input])).to(dtype=torch.bfloat16)
        ts_sr = torch.tensor([sampling_rate])
        ts_lens = torch.tensor([ts_len])
        return dict(input_ids=input_ids,
                    ts_values=ts_input,
                    ts_sr=ts_sr,
                    ts_lens=ts_lens,
                    ts_token_id=self.ts_token_id)
