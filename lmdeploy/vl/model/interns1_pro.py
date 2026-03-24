# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel

logger = get_logger('lmdeploy')


def check_transformers():
    try:
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError('please install latest transformers by '
                          'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class InternS1ProVisionModel(VisionModel):
    """InternS1Pro model.

    Basically the same preprocessing as Qwen3VL, but with Time Series support.
    """

    _arch = ['InternS1ProForConditionalGeneration', 'InternS1_1_ForConditionalGeneration']

    def build_preprocessor(self):
        check_transformers()
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

        # image tokens
        self.image_token = self.processor.image_token
        self.image_token_id = self.processor.image_token_id

        # video tokens
        self.video_token = self.processor.video_token
        self.video_token_id = self.processor.video_token_id

        # time series tokens
        self.ts_token = getattr(self.processor, 'ts_token', None)
        self.ts_token_id = getattr(self.processor, 'ts_token_id', None)

        # vision start and end tokens
        self.vision_start_token = self.processor.vision_start_token
        self.vision_end_token = self.processor.vision_end_token

    def get_processor_args(self, mm_processor_kwargs: dict[str, Any] | None = None):
        min_pixels = self.processor.image_processor.size['shortest_edge']
        max_pixels = self.processor.image_processor.size['longest_edge']

        if mm_processor_kwargs is None:
            return min_pixels, max_pixels

        input_min_pixels = mm_processor_kwargs.get('min_pixels', None)
        input_max_pixels = mm_processor_kwargs.get('max_pixels', None)

        # boundary check for min_pixels and max_pixels
        if input_min_pixels is None:
            if input_max_pixels is not None:
                # only max_pixels is given in the input
                if input_max_pixels < min_pixels:
                    logger.warning(
                        f'input max_pixels {input_max_pixels} < default min_pixels {min_pixels}, fall back to default.')
                    return min_pixels, max_pixels
                max_pixels = input_max_pixels
        else:
            if input_max_pixels is None:
                # only min_pixels is given in the input
                if input_min_pixels > max_pixels:
                    logger.warning(
                        f'input min_pixels {input_min_pixels} > default max_pixels {max_pixels}, fall back to default.')
                    return min_pixels, max_pixels
            else:
                if input_min_pixels > input_max_pixels:
                    logger.warning(
                        f'input min_pixels {input_min_pixels} > max_pixels {input_max_pixels}, fall back to default.')
                    return min_pixels, max_pixels
                max_pixels = input_max_pixels
            min_pixels = input_min_pixels

        return min_pixels, max_pixels

    def check_time_series_input(self, messages):
        has_time_series_input = any(
            isinstance(message['content'], list) and any(item['type'] == 'time_series' for item in message['content'])
            for message in messages)
        self.has_time_series_input = has_time_series_input

    def _preprocess_image(self,
                          data: list[Any],
                          params: dict[str, Any],
                          mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:

        image = data.convert('RGB')
        min_pixels, max_pixels = self.get_processor_args(mm_processor_kwargs)

        result = self.processor.image_processor(images=image,
                                                size={
                                                    'shortest_edge': min_pixels,
                                                    'longest_edge': max_pixels
                                                },
                                                return_tensors='pt')
        merge_length = self.processor.image_processor.merge_size**2
        image_tokens = result['image_grid_thw'].prod(dim=1) // merge_length
        result.update(dict(image_size=image.size, image_tokens=image_tokens, image_token_id=self.image_token_id))
        return result

    def _preprocess_video(self,
                          data: list[Any],
                          params: dict[str, Any],
                          mm_processor_kwargs: dict[str, Any] | None = None) -> list[dict]:

        # TODO: zhouxinyu, apply transformers smart_resize using per-request kwargs
        metadata = params['video_metadata']
        video_kwargs = dict(return_metadata=True,
                            do_resize=True,
                            do_sample_frames=False,
                            video_metadata=metadata,
                            return_tensors='pt')
        result = self.processor.video_processor(videos=data, **video_kwargs)
        video_grid_thw = result['video_grid_thw']

        merge_length = self.processor.video_processor.merge_size**2
        if metadata.get('fps') is None:
            logger.warning_once('Qwen3VL: fps not found, defaulting to 24.')
            metadata['fps'] = metadata['fps'] or 24

        # if timestamps are not provided, calculate them
        curr_timestamp = self.processor._calculate_timestamps(
            metadata['frames_indices'],
            metadata['fps'],
            self.processor.video_processor.merge_size,
        )

        frame_seqlen = video_grid_thw[0][1:].prod() // merge_length
        result.update(curr_timestamp=curr_timestamp, frame_seqlen=frame_seqlen, video_token_id=self.video_token_id)
        return result

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

    def to_pytorch_aux_video(self, messages, prompt, VIDEO_TOKEN, tokenizer, sequence_start):
        """Pack the video input to the compatible format with pytorch
        engine."""

        # collect all preprocessing result from messages
        preps = [x['content'] for x in messages if x['role'] == 'preprocess']
        assert len(preps) == 1
        preps = preps[0]

        # split prompt into segments and validate data
        segs = prompt.split(self.vision_start_token + self.video_token + self.vision_end_token)
        assert len(segs) == len(preps) + 1, (f'the number of {self.video_token} is not equal '
                                             f'to input videos, {len(segs) - 1} vs {len(preps)}')

        # calculate the video token offset for each video
        input_ids = []
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(preps):
                preps[i - 1].update(offset=len(input_ids))
                frame_seqlen = preps[i - 1]['frame_seqlen']
                assert self.video_token_id == preps[i - 1]['video_token_id']

                video_grid_thw = preps[i - 1]['video_grid_thw']
                curr_timestamp = preps[i - 1]['curr_timestamp']

                # update prompt with timestamp index tokens and video pad tokens
                video_placeholder = ''
                for frame_idx in range(video_grid_thw[0][0]):
                    curr_time = curr_timestamp[frame_idx]
                    video_placeholder += f'<{curr_time:.1f} seconds>'
                    video_placeholder += (self.vision_start_token + '<|placeholder|>' * frame_seqlen +
                                          self.vision_end_token)

                video_placeholder = video_placeholder.replace('<|placeholder|>', self.video_token)
                video_token_ids = tokenizer.encode(video_placeholder)
                input_ids.extend(video_token_ids)

                preps[i - 1].update(video_tokens=len(video_token_ids))

            token_ids = tokenizer.encode(seg, add_bos=((i == 0) and sequence_start))
            input_ids.extend(token_ids)

        return dict(prompt=prompt, input_ids=input_ids, multimodal=preps)

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

    def build_model(self):
        # TODO: implement for turbomind
        pass

    @torch.no_grad()
    def forward(self, messages: list[dict], max_batch_size: int = 1) -> list[dict]:
        # TODO: implement for turbomind
        pass

    def to_turbomind(self,
                     messages,
                     chat_template,
                     tokenizer,
                     sequence_start,
                     chat_template_kwargs: dict | None = None,
                     **kwargs):
        # TODO: implement for turbomind
        pass
