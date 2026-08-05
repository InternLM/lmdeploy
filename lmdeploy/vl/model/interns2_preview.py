# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import numpy as np
import torch

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, MultimodalSpecialTokens
from lmdeploy.vl.model.qwen3_5 import Qwen3_5Model, check_transformers
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class InternS2PreviewVisionModel(Qwen3_5Model):
    """InternS2Preview vision model with time-series preprocessing."""

    _arch = [
        'InternS2PreviewForConditionalGeneration',
        'InternS2PreviewForCausalLM',
    ]
    _turbomind_native_vision = True

    def build_preprocessor(self, trust_remote_code: bool = False):
        super().build_preprocessor(trust_remote_code=trust_remote_code)

        self.ts_token = getattr(self.processor, 'ts_token', None)
        self.ts_token_id = getattr(self.processor, 'ts_token_id', None)
        self.ts_start_token = getattr(self.processor, 'ts_start_token', None)
        self.ts_end_token = getattr(self.processor, 'ts_end_token', None)

        self.mm_tokens = MultimodalSpecialTokens(image_token=self.image_token,
                                                 video_token=self.video_token,
                                                 ts_token=self.ts_token,
                                                 image_token_id=self.image_token_id,
                                                 video_token_id=self.video_token_id,
                                                 ts_token_id=self.ts_token_id)

        self.ts_signals_do_normalize = getattr(self.processor, 'ts_signals_do_normalize', True)
        self.ts_signals_do_truncate = getattr(self.processor, 'ts_signals_do_truncate', True)

    def time_series_processor(self,
                              text: list[str],
                              time_series: list[Any],
                              sampling_rate: float | None = None,
                              **kwargs):
        ts_input = time_series[0] if isinstance(time_series, list) else time_series
        sampling_rate = sampling_rate[0] if isinstance(sampling_rate, list) else sampling_rate

        if not isinstance(ts_input, np.ndarray):
            ts_input = np.array(ts_input, dtype=np.float32)

        if self.ts_signals_do_normalize:
            mean = ts_input.mean(axis=0, keepdims=True)
            std = ts_input.std(axis=0, keepdims=True)
            ts_input = (ts_input - mean) / (std + 1e-8)

        max_ts_len = 240000
        if self.ts_signals_do_truncate and len(ts_input) > max_ts_len:
            ts_input = ts_input[:max_ts_len]

        if ts_input.ndim == 1:
            ts_input = ts_input[:, None]

        ts_len = ts_input.shape[0]
        ts_channel = ts_input.shape[1]

        if sampling_rate is None or sampling_rate <= 0:
            sampling_rate = max(ts_len / 4, 1.0)

        # newer checkpoints with TS-forecast compute ts tokens differently
        if getattr(self.hf_config, 'ts_forecaster_config', None) is not None:
            chunk_size = getattr(self.processor, 'chunk_size', 12800)
            num_query = getattr(self.processor, 'num_query', 2)
            subrate = max(ts_len / 500, 1.0)
            stride = subrate * num_query
            patch_size = np.ceil(stride)
            chunk_num = ts_len // chunk_size
            tail_len = ts_len - chunk_size * chunk_num
            full_chunk_tokens = (np.ceil((chunk_size - patch_size) / stride + 1) * num_query + 1) // 2
            tail_tokens = (np.ceil((tail_len - patch_size) / stride + 1) * num_query + 1) // 2
            ts_tokens = int(chunk_num * full_chunk_tokens + tail_tokens)
        else:
            stride = np.floor(160 / ((1 + np.exp(-sampling_rate / 100))**6))
            patch_size = stride * 2
            embed_length = (np.ceil((ts_len - patch_size) / stride) + 1)
            ts_tokens = int((embed_length // 2 + 1) // 2)

        for i in range(len(text)):
            ts_placeholder = f'{self.ts_start_token}{self.ts_token}{self.ts_end_token}'
            if ts_placeholder in text[i]:
                expanded_placeholder = self.ts_start_token + self.ts_token * ts_tokens + self.ts_end_token
                text[i] = text[i].replace(ts_placeholder, expanded_placeholder, 1)
            elif self.ts_token in text[i]:
                text[i] = text[i].replace(self.ts_token, self.ts_token * ts_tokens)

        input_ids = self.tokenizer(text, add_special_tokens=False, **kwargs)['input_ids']

        ts_input = torch.from_numpy(np.array([ts_input])).to(dtype=torch.bfloat16)
        ts_sr = torch.tensor([sampling_rate])
        ts_lens = torch.tensor([ts_len])
        ts_channels = torch.tensor([ts_channel])
        return dict(input_ids=input_ids,
                    ts_values=ts_input,
                    ts_sr=ts_sr,
                    ts_lens=ts_lens,
                    ts_channels=ts_channels,
                    ts_token_id=self.ts_token_id)

    def build_model(self, trust_remote_code: bool = False):
        check_transformers()
        arch = self.hf_config.architectures[0]
        if arch in self._arch:
            from transformers import AutoModelForImageTextToText as AutoModelCls
        else:
            raise ValueError(f'Unsupported arch={arch}')

        if self.with_llm:
            self.vl_model = AutoModelCls.from_pretrained(self.model_path,
                                                         device_map='cpu',
                                                         trust_remote_code=trust_remote_code)
        else:
            from accelerate import init_empty_weights
            with init_empty_weights():
                config = self.hf_config
                config.tie_word_embeddings = False
                if hasattr(config, 'text_config'):
                    config.text_config.tie_word_embeddings = False

                model = AutoModelCls.from_config(config, trust_remote_code=trust_remote_code)
                model.visual = model.model.visual
                model.time_series = model.model.time_series
                del model.model
                del model.lm_head
                model.half()

            from accelerate import load_checkpoint_and_dispatch
            with disable_logging():
                load_checkpoint_and_dispatch(model=model,
                                             checkpoint=self.model_path,
                                             device_map='auto' if not self.with_llm else {'': 'cpu'},
                                             max_memory=self.max_memory,
                                             no_split_module_classes=[
                                                 'InternS2PreviewDecoderLayer',
                                                 'InternS2PreviewVisionBlock',
                                             ],
                                             dtype=torch.half)
            self.model = model.eval()
