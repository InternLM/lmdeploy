# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import numpy as np
import torch

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, MultimodalSpecialTokens
from lmdeploy.vl.model.qwen3 import Qwen3VLModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


def check_transformers():
    try:
        # import config instead of model to avoid import error on windows
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config  # noqa: F401
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig  # noqa: F401
    except ImportError:
        raise ImportError('please install latest transformers by '
                          'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class Qwen3_5Model(Qwen3VLModel):
    """Qwen3_5 model."""

    _arch = [
        'Qwen3_5ForConditionalGeneration',
        'Qwen3_5MoeForConditionalGeneration',
        'InternS2PreviewForConditionalGeneration',
        'InternS2PreviewForCausalLM',
    ]
    _turbomind_native_vision = True

    def build_preprocessor(self, trust_remote_code: bool = False):
        check_transformers()
        super().build_preprocessor(trust_remote_code=trust_remote_code)

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
                              text: list[str],
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
        ts_channel = ts_input.shape[1]

        # set the default value to ts_len / 4 if sr is not provided or invalid
        if sampling_rate is None or sampling_rate <= 0:
            sampling_rate = max(ts_len / 4, 1.0)

        # compute num ts tokens
        if getattr(self.hf_config, 'model_type', None) == 'intern_s2_preview':
            chunk_size = getattr(self.processor, 'chunk_size', 12800)
            patch = max(ts_len / 500, 1.0)
            chunk_num = ts_len // chunk_size
            full_chunk_tokens = (np.ceil(chunk_size / patch) + 1) // 2
            tail_tokens = (np.ceil((ts_len - chunk_size * chunk_num) / patch) + 1) // 2
            ts_tokens = int(chunk_num * full_chunk_tokens + tail_tokens)
        else:
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
        if arch == 'Qwen3_5ForConditionalGeneration':
            from transformers import Qwen3_5ForConditionalGeneration as AutoModelCls
        elif arch == 'Qwen3_5MoeForConditionalGeneration':
            from transformers import Qwen3_5MoeForConditionalGeneration as AutoModelCls
        elif arch in ['InternS2PreviewForConditionalGeneration', 'InternS2PreviewForCausalLM']:
            from transformers import AutoModelForImageTextToText as AutoModelCls
        else:
            raise ValueError(f'Unsupported arch={arch}')

        if self.with_llm:
            if arch in ['Qwen3_5ForConditionalGeneration', 'Qwen3_5MoeForConditionalGeneration']:
                self.vl_model = AutoModelCls.from_pretrained(self.model_path, device_map='cpu')
            else:
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

                if arch in ['Qwen3_5ForConditionalGeneration', 'Qwen3_5MoeForConditionalGeneration']:
                    model = AutoModelCls._from_config(config)
                    model.visual = model.model.visual
                    del model.model
                    del model.lm_head
                elif arch in ['InternS2PreviewForConditionalGeneration', 'InternS2PreviewForCausalLM']:
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
                                                'Qwen3_5VisionBlock',
                                                'Qwen3_5MoeVisionBlock',
                                                'InternS2PreviewDecoderLayer',
                                                'InternS2PreviewVisionBlock'
                                            ],
                                             dtype=torch.half)
            self.model = model.eval()
